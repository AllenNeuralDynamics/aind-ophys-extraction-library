"""Computation functions for ophys ROI extraction.

Covers data handling (virtual datasets, memory maps), ROI analysis
(contours, centre-of-mass), trace processing (neuropil correction),
CaImAn CNMF helpers, and visualisation (summary images, contour video).

Imports heavy deps (caiman, suite2p, cellpose, h5py, sparse, cv2, …) at
module level; this module is omitted from coverage because it requires a
full GPU-capable CO environment to exercise.
"""

import json
import logging
import os
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import caiman
import cv2
import h5py
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import skimage
import sparse
import suite2p
from aind_ophys_utils.array_utils import downsample_array
from aind_ophys_utils.segmentation_utils import roi_probabilities
from aind_ophys_utils.summary_images import mean_image
from aind_pophys_metadata.paths import find_one
from caiman.source_extraction.cnmf import cnmf, params
from cellpose.models import Cellpose
from scipy.sparse import coo_matrix, hstack, linalg

from .settings import ExtractionSettings

logger = logging.getLogger(
    "Source extraction using a combination of Cellpose, Suite2p, and CaImAn"
)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def make_output_directory(output_dir: Path, experiment_id: str) -> Path:
    """Create and return the per-plane extraction output directory.

    Parameters
    ----------
    output_dir : Path
        Root results directory.
    experiment_id : str
        Plane/FOV identifier.

    Returns
    -------
    Path
        ``output_dir / experiment_id / "extraction"``, created if absent.
    """
    plane_dir = output_dir / experiment_id / "extraction"
    plane_dir.mkdir(exist_ok=True, parents=True)
    return plane_dir


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------


def create_virtual_dataset(
    h5_file: Path, frame_locations: list, frames_length: int, temp_dir: Path
) -> Path:
    """Create a virtual dataset from a list of frame locations.

    Parameters
    ----------
    h5_file : Path
        Path to the source HDF5 file.
    frame_locations : list
        List of (start, end) frame index pairs (inclusive).
    frames_length : int
        Total number of frames in the virtual dataset.
    temp_dir : Path
        Directory for the temporary VDS file.

    Returns
    -------
    Path
        Path to the created virtual dataset.
    """
    with h5py.File(h5_file, "r") as f:
        data_shape = f["data"].shape
        dtype = f["data"].dtype
        vsource = h5py.VirtualSource(f["data"])
        layout = h5py.VirtualLayout(shape=(frames_length, *data_shape[1:]), dtype=dtype)
        start = 0
        for loc in frame_locations:
            layout[start : start + loc[1] - loc[0] + 1] = vsource[loc[0] : loc[1] + 1]
            start += loc[1] - loc[0] + 1
        h5_file = temp_dir / h5_file.name
        with h5py.File(h5_file, "w") as f:
            f.create_virtual_dataset("data", layout)

    return h5_file


def bergamo_segmentation(
    motion_corr_fp: Path, valid_epoch_stems: Sequence[str], temp_dir: Path
) -> Path:
    """Create a virtual dataset for Bergamo data, excluding photostimulation frames.

    Parameters
    ----------
    motion_corr_fp : Path
        Path to the motion-corrected HDF5 file.
    valid_epoch_stems : sequence of str
        TIFF stems of the stimulus epochs to keep, in acquisition order
        (see ``utils.metadata_utils.get_valid_epoch_stems``).
    temp_dir : Path
        Directory for the temporary virtual dataset.

    Returns
    -------
    Path
        Path to the filtered virtual dataset.
    """
    motion_dir = motion_corr_fp.parent
    epoch_loc_fp = find_one(motion_dir, "epoch_locations.json", recursive=False)
    with open(epoch_loc_fp, "r") as j:
        epoch_locations = json.load(j)
    frame_locations = [epoch_locations[i] for i in valid_epoch_stems]
    frames_length = sum([(i[1] - i[0] + 1) for i in frame_locations])

    return create_virtual_dataset(
        motion_corr_fp, frame_locations, frames_length, temp_dir
    )


def create_chunk_vds(start: int, chunksize: int, input_fn: str, tmp_dir: str) -> str:
    """Create a Virtual Dataset for a specific chunk of the input data.

    Parameters
    ----------
    start : int
        Starting frame index of the chunk.
    chunksize : int
        Number of frames in the chunk.
    input_fn : str
        Path to the input HDF5 file.
    tmp_dir : str
        Directory for the temporary VDS chunk file.

    Returns
    -------
    str
        Path to the created VDS chunk file.
    """
    with h5py.File(input_fn, "r") as fin:
        data = fin["data"]
        end = min(start + chunksize, data.shape[0])
        layout = h5py.VirtualLayout(
            shape=(end - start, *data.shape[1:]), dtype=data.dtype
        )
        vsource = h5py.VirtualSource(input_fn, "data", shape=data.shape)
        layout[:] = vsource[start:end]
        vds_file = os.path.join(tmp_dir, f"chunk_{start}.h5")
        with h5py.File(vds_file, "w") as fout:
            fout.create_virtual_dataset("data", layout)
    return vds_file


def create_mmap_file(
    input_fn: str,
    unique_id: str,
    tmp_dir: str,
    chunksize: int = 500,
    n_chunks: int = 100,
) -> str:
    """Create a CaImAn memory-mapped file from the input HDF5 data.

    Parameters
    ----------
    input_fn : str
        Path to the input HDF5 file.
    unique_id : str
        Unique identifier used for the output mmap filename.
    tmp_dir : str
        Directory for temporary chunk files and the final mmap.
    chunksize : int, optional
        Frames per chunk when splitting large datasets.
    n_chunks : int, optional
        Number of chunks to use when combining the mmap file.

    Returns
    -------
    str
        Path to the created memory-mapped file.
    """
    with h5py.File(input_fn, "r") as fin:
        data = fin["data"]
        if data.nbytes < 1e9:
            logging.info("Data is small, saving directly as a memory-mapped file.")
            fname_new = caiman.save_memmap(
                [str(input_fn)],
                var_name_hdf5="data",
                order="C",
                base_name=unique_id,
            )
        else:
            logging.info("Data is large, splitting into chunks.")
            with Pool() as pool:
                try:
                    chunkfiles = pool.starmap(
                        create_chunk_vds,
                        [
                            (start, chunksize, input_fn, tmp_dir)
                            for start in range(0, data.shape[0], chunksize)
                        ],
                    )
                    logging.info(f"Created {len(chunkfiles)} chunk files.")
                    fname_new = caiman.save_memmap(
                        chunkfiles,
                        var_name_hdf5="data",
                        order="C",
                        dview=pool,
                        base_name=unique_id,
                        n_chunks=n_chunks,
                    )
                    logging.info("Memory-mapped file created successfully.")
                finally:
                    logging.info("Cleaning up temporary chunk files.")
                    pool.map(os.remove, chunkfiles)
    return fname_new


# ---------------------------------------------------------------------------
# ROI analysis
# ---------------------------------------------------------------------------


def com(rois: Union[np.ndarray, sparse.COO]) -> np.ndarray:
    """Calculate centre-of-mass for spatial components.

    Parameters
    ----------
    rois : np.ndarray or sparse.COO
        Spatial components tensor (K x height x width).

    Returns
    -------
    np.ndarray
        Centre-of-mass coordinates (K x 2).
    """
    d1, d2 = rois.shape[1:]
    Coor = np.array(
        list(map(np.ravel, np.meshgrid(np.arange(d2), np.arange(d1)))), dtype=rois.dtype
    )
    A = rois.reshape((rois.shape[0], d1 * d2)).tocsc()
    return (A / A.sum(axis=1)).dot(Coor.T)


def get_contours(
    rois: Union[np.ndarray, sparse.COO], thr: float = 0.2, thr_method: str = "max"
) -> list:
    """Get contour coordinates of spatial components.

    Parameters
    ----------
    rois : np.ndarray or sparse.COO
        Spatial components tensor (K x height x width).
    thr : float, optional
        Threshold fraction for computing contours.
    thr_method : str, optional
        Thresholding method: ``'max'`` or ``'nrg'``.

    Returns
    -------
    list of dict
        Per-component dicts with keys ``coordinates``, ``CoM``, ``neuron_id``.
    """
    nr, dims = rois.shape[0], rois.shape[1:]
    d1, d2 = dims[:2]
    d = np.prod(dims)
    x, y = np.mgrid[0:d1:1, 0:d2:1]  # noqa: F841

    coordinates = []
    cm = com(rois)
    A = rois.T.reshape((d, nr)).tocsc()

    for i in range(nr):
        pars = dict()
        patch_data = A.data[A.indptr[i] : A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == "nrg":
            cumEn = np.cumsum(patch_data[indx] ** 2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != "max":
                logging.warning("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i] : A.indptr[i + 1]]] = (
                patch_data / patch_data.max()
            )

        Bmat = np.reshape(Bvec, dims, order="F")
        pars["coordinates"] = []
        for B in Bmat if len(dims) == 3 else [Bmat]:
            vertices = skimage.measure.find_contours(B.T, thr)
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
            pars["coordinates"] = v if len(dims) == 2 else (pars["coordinates"] + [v])
        pars["CoM"] = np.squeeze(cm[i, :])
        pars["neuron_id"] = i + 1
        coordinates.append(pars)
    return coordinates


def estimate_gSig(diameter: float, img: np.ndarray, fac: float = 2.35482) -> float:
    """Estimate Gaussian sigma for CaImAn based on cell diameter.

    Parameters
    ----------
    diameter : float
        Cell diameter in pixels; if 0, estimated via Cellpose.
    img : np.ndarray
        Mean image used when ``diameter == 0``.
    fac : float, optional
        FWHM-to-sigma conversion factor (default: 2√(2 ln 2)).

    Returns
    -------
    float
        Estimated Gaussian sigma.
    """
    if diameter == 0:
        diameter = Cellpose().sz.eval(img)[0]
        logger.info(
            f"'diameter' set to 0 — automatically estimated with Cellpose as {diameter:.3f}."
        )
    gSig = diameter / fac
    logger.info(f"Setting gSig to {gSig:.3f}.")
    return gSig


# ---------------------------------------------------------------------------
# Trace processing
# ---------------------------------------------------------------------------


def get_r_from_min_mi(
    raw_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    resolution: float = 0.01,
    r_test_range: list = [0, 2],
) -> tuple:
    """Find the neuropil coefficient r that minimises mutual information.

    Parameters
    ----------
    raw_trace : np.ndarray
        1D raw fluorescence trace.
    neuropil_trace : np.ndarray
        1D neuropil trace.
    resolution : float, optional
        Step size for r search.
    r_test_range : list of float, optional
        Inclusive [min, max] range for r search.

    Returns
    -------
    tuple
        ``(r_best, mi_iters, r_iters)``
    """
    r_iters = np.arange(r_test_range[0], r_test_range[1] + resolution, resolution)
    mi_iters = np.zeros(len(r_iters))
    neuropil_trace[np.isnan(neuropil_trace)] = 0
    raw_trace[np.isnan(raw_trace)] = 0
    for r_i, r_temp in enumerate(r_iters):
        Fc = raw_trace - r_temp * neuropil_trace
        mi_iters[r_i] = skimage.metrics.normalized_mutual_information(
            Fc, neuropil_trace
        )
    min_ind = np.argmin(mi_iters)
    r_best = r_iters[min_ind]
    return r_best, mi_iters, r_iters


def get_FC_from_r(
    raw_trace: np.ndarray, neuropil_trace: np.ndarray, min_r_count: int = 5
) -> tuple:
    """Compute neuropil-corrected traces using per-ROI optimal r values.

    Parameters
    ----------
    raw_trace : np.ndarray
        2D raw fluorescence traces (ROIs x time).
    neuropil_trace : np.ndarray
        2D neuropil traces (ROIs x time).
    min_r_count : int, optional
        Minimum valid r values (<1) required to calculate the mean; uses 0.8 otherwise.

    Returns
    -------
    tuple
        ``(FCs, r_values, raw_r)``
    """
    r_values = np.zeros(raw_trace.shape[0])
    FCs = np.zeros_like(raw_trace)
    for roi in range(raw_trace.shape[0]):
        r_values[roi], _, _ = get_r_from_min_mi(raw_trace[roi], neuropil_trace[roi])
    mean_r = np.mean(r_values[r_values < 1])
    if len(np.where(r_values < 1)[0]) < min_r_count:
        mean_r = 0.8
    raw_r = r_values.copy()
    r_values[r_values >= 1] = mean_r
    for roi in range(raw_trace.shape[0]):
        FCs[roi] = raw_trace[roi] - r_values[roi] * neuropil_trace[roi]
    return FCs, r_values, raw_r


# ---------------------------------------------------------------------------
# CaImAn helpers
# ---------------------------------------------------------------------------


def build_CNMFParams(
    args: ExtractionSettings,
    ops: dict,
    cnmfe: bool,
    Ain: Optional[np.ndarray] = None,
    dims: Optional[tuple] = None,
) -> params.CNMFParams:
    """Build CaImAn CNMF parameter object.

    Parameters
    ----------
    args : ExtractionSettings
        Extraction settings.
    ops : dict
        Summary images dict (must contain ``meanImg``).
    cnmfe : bool
        True to use CNMF-E (ring model); False for standard CNMF.
    Ain : sparse matrix, optional
        Initial spatial components; when provided, ROI detection is skipped.
    dims : tuple, optional
        FOV dimensions; required when ``Ain`` is provided.

    Returns
    -------
    params.CNMFParams
        CaImAn parameter object.
    """
    gSig = estimate_gSig(args.diameter, ops["meanImg"])

    params_dict = {
        "p": 1,
        "nb": 0 if cnmfe else args.nb,
        "only_init": cnmfe,
        "gSig": (gSig, gSig),
        "gSiz": (int(round(gSig * (4 if cnmfe else 2) + 1)),) * 2,
        "ssub_B": args.ssub_B,
        "normalize_init": not cnmfe,
        "center_psf": cnmfe,
        "ring_size_factor": 1.5 if cnmfe else None,
        "min_SNR": args.snr_thr,
        "rval_thr": args.rval_thr,
        "use_cnn": args.cnn_thr > 0,
        "min_cnn_thr": args.cnn_thr,
    }

    if Ain is None:
        params_dict.update(
            {
                "K": args.k_components,
                "method_init": args.init,
                "rf": args.rf,
                "stride": args.stride,
                "ssub": args.ssub,
                "tsub": args.tsub,
                "min_corr": args.min_corr,
                "min_pnr": args.min_pnr,
                "merge_thr": args.merge_thr,
            }
        )
    else:
        params_dict.update(
            {
                "K": None,
                "method_init": "corr_pnr" if cnmfe else "greedy_roi",
                "ssub": 1,
                "tsub": 1,
                "min_corr": 0,
                "min_pnr": 0,
                "merge_thr": 1,
                "init_iter": 1,
            }
        )
        if Ain is not None and dims is not None:
            params_dict["seed_method"] = caiman.base.rois.com(Ain, *dims)

    return params.CNMFParams(params_dict=params_dict)


def run_caiman_extraction(
    input_fn: Union[str, Path],
    unique_id: str,
    args: ExtractionSettings,
    ops: dict,
    Ain: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CaImAn CNMF to extract neural activity traces.

    Parameters
    ----------
    input_fn : str or Path
        Input HDF5 file.
    unique_id : str
        Plane identifier used for mmap naming.
    args : ExtractionSettings
        Extraction settings.
    ops : dict
        Summary images dict (must contain ``meanImg``).
    Ain : sparse matrix, optional
        Initial spatial components; if None, performs ROI detection.
    n_jobs : int, optional
        Number of parallel workers.

    Returns
    -------
    tuple
        ``(traces_corrected, traces_neuropil, traces_roi, data, coords, iscell)``
    """
    logger.info(f"running CaImAn v{caiman.__version__}")
    cnmfe = args.neuropil == "cnmf-e"
    fname_new = create_mmap_file(input_fn, unique_id, str(args.tmp_dir))
    Yr, dims, T = caiman.load_memmap(fname_new)
    movie = np.reshape(Yr.T, [T] + list(dims), order="F")
    opts = build_CNMFParams(args, ops, cnmfe, Ain, dims)
    Ain_processed = None if cnmfe or Ain is None else (Ain > 0).toarray()
    with Pool(n_jobs) as pool:
        cnm = cnmf.CNMF(
            n_processes=pool._processes,
            dview=pool,
            params=opts,
            Ain=Ain_processed,
        )
        cnm.fit(movie)
        if not cnmfe and Ain is None:
            cnm = cnm.refit(movie, dview=pool)
        cnm.estimates.dims = dims
        cnm.params.init["gSig"] = tuple(map(int, cnm.params.init["gSig"]))
        cnm.estimates.evaluate_components(movie, cnm.params, dview=pool)
    return format_caiman_output(cnm.estimates, cnmfe, Yr)


def format_caiman_output(
    e: Any,
    cnmfe: bool,
    Yr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Format CaImAn CNMF estimates into the standard extraction output.

    Parameters
    ----------
    e : caiman.source_extraction.cnmf.estimates.Estimates
        CaImAn estimates object.
    cnmfe : bool
        True if CNMF-E was used.
    Yr : np.ndarray
        Flattened data array (pixels x time).

    Returns
    -------
    tuple
        ``(traces_corrected, traces_neuropil, traces_roi, data, coords, iscell)``
    """
    assert np.allclose(linalg.norm(e.A, 2, 0), 1)
    traces_corrected = (e.C + e.YrA).astype("f4")
    if cnmfe:
        Atb0 = e.A.T.dot(e.b0)[:, None]
        traces_corrected += Atb0
        ssub_B = np.round(np.sqrt(Yr.shape[0] / e.W.shape[0])).astype(int)
        if ssub_B == 1:
            AtW = e.A.T.dot(e.W)
            traces_neuropil = (
                Atb0 + AtW.dot(Yr) - AtW.dot(e.A).dot(e.C) - AtW.dot(e.b0)[:, None]
            ).astype("f4")
        else:
            ds_mat = caiman.source_extraction.cnmf.utilities.decimation_matrix(
                e.dims, ssub_B
            )
            Ads = ds_mat.dot(e.A)
            b0ds = ds_mat.dot(e.b0)
            AtW = Ads.T.dot(e.W)
            traces_neuropil = (
                Atb0
                + ssub_B**2
                * (
                    AtW.dot(ds_mat).dot(Yr)
                    - AtW.dot(Ads).dot(e.C)
                    - AtW.dot(b0ds)[:, None]
                )
            ).astype("f4")
    else:
        traces_neuropil = e.A.T.dot(e.b).dot(e.f).astype("f4")
        traces_corrected += (
            0.8 * traces_neuropil
        )  # TODO: check factor on groundtruth data
    traces_roi = (e.C + e.YrA + traces_neuropil).astype("f4")
    data = []
    coords = []
    for i in range(e.A.shape[1]):
        roi = coo_matrix(e.A[:, i].reshape(e.dims, order="F").toarray(), dtype="f4")
        data.append(roi.data)
        coords.append(
            np.array([i * np.ones(len(roi.data)), roi.row, roi.col], dtype="i2")
        )
    if len(data):
        data = np.concatenate(data)
        coords = np.hstack(coords)
    iscell = np.zeros((e.A.shape[1], 2), dtype="f4")
    iscell[e.idx_components, 0] = 1
    iscell[:, 1] = e.cnn_preds if hasattr(e, "cnn_preds") else np.nan
    return traces_corrected, traces_neuropil, traces_roi, data, coords, iscell


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def save_summary_images_with_rois(
    output_dir: Path,
    unique_id: str,
    rois: sparse.COO,
    iscell: np.ndarray,
    ops: dict,
    corr_img: np.ndarray,
) -> None:
    """Save summary images with ROI contours overlaid.

    Parameters
    ----------
    output_dir : Path
        Directory for output PNGs.
    unique_id : str
        Plane identifier used in output filenames.
    rois : sparse.COO
        Spatial components (K x height x width).
    iscell : np.ndarray
        Cell classification array (K x 2).
    ops : dict
        Suite2p ops dict containing ``meanImg`` and ``max_proj``.
    corr_img : np.ndarray
        Correlation image.
    """
    cm_pts = com(rois)
    coordinates = get_contours(rois)
    dims = rois.shape[1:]
    x_size = 17 * max(dims[1] / dims[0], 0.4)
    fix, ax = plt.subplots(1, 3, figsize=(x_size, 6))
    lw = min(512 / max(*dims), 3)
    for i, img in enumerate((ops["meanImg"], ops["max_proj"], corr_img)):
        vmin, vmax = np.nanpercentile(img, (1, 99))
        ax[i].imshow(img, interpolation=None, cmap="gray", vmin=vmin, vmax=vmax)
        for c, good in zip(coordinates, iscell[:, 0]):
            ax[i].plot(*c["coordinates"].T, c="orange" if good else "r", lw=lw)
        ax[i].axis("off")
        ax[i].set_title(
            ("mean image", "max image", "correlation image")[i],
            fontsize=min(24, 2.4 + 2 * x_size),
        )
    plt.tight_layout(pad=0.1)
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    for i in (0, 1, 2):
        for k in range(rois.shape[0]):
            ax[i].text(
                *cm_pts[k],
                str(k),
                color="orange" if iscell[k, 0] else "r",
                fontsize=8 * lw,
            )
    plt.savefig(
        output_dir / f"{unique_id}_detected_ROIs_withIDs.png",
        bbox_inches="tight",
        pad_inches=0.02,
    )


def contour_video(
    output_path: str,
    data: Union[h5py.Dataset, np.ndarray],
    rois: Union[sparse.COO, np.ndarray],
    traces: np.ndarray,
    downscale: int = 10,
    fs: float = 30,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.9975,
    only_raw: bool = False,
    n_jobs: Optional[int] = (
        None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)
    ),
    bitrate: str = "0",
    crf: int = 20,
    cpu_used: int = 4,
) -> None:
    """Create a VP9-encoded contour overlay video.

    Parameters
    ----------
    output_path : str
        Output path for the encoded video.
    data : h5py.Dataset or np.ndarray
        Source movie (frames x height x width).
    rois : sparse.COO or np.ndarray
        Spatial components (K x height x width).
    traces : np.ndarray
        Temporal components (K x time).
    downscale : int, optional
        Temporal decimation factor.
    fs : float, optional
        Frame rate for the encoded video.
    lower_quantile : float, optional
        Lower normalisation quantile.
    upper_quantile : float, optional
        Upper normalisation quantile.
    only_raw : bool, optional
        Produce raw-data-only video (no reconstruction/residual tiles).
    n_jobs : int, optional
        Parallel workers for downsampling.
    bitrate : str, optional
        Target bitrate (``"0"`` = constant quality).
    crf : int, optional
        Constant-quality factor (0–63; lower = better quality).
    cpu_used : int, optional
        VP9 speed/quality trade-off (0–5; higher = faster).
    """
    dims = data.shape[1:]
    img_contours = np.zeros(dims + (3,), np.uint8)
    rgb = (255, 127, 14)
    for m in rois:
        if isinstance(m, sparse.COO):
            m = m.todense()
        ret, thresh = cv2.threshold(
            (m > m.max() / 10).astype(np.uint8), 0, 1, cv2.THRESH_BINARY
        )
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
        for contour in contours:
            cv2.drawContours(img_contours, contour, -1, rgb, max(max(dims) // 200, 1))
    mov = downsample_array(data, downscale, 1, n_jobs=n_jobs)
    minmov, maxmov = np.nanquantile(
        mov[:: max(1, len(mov) // 100)], (lower_quantile, upper_quantile)
    )

    def scale(m: np.ndarray) -> np.ndarray:
        """Normalise and clip to uint8."""
        return np.array(
            ThreadPool(n_jobs).map(
                lambda frame: np.clip(
                    255 * (frame - minmov) / (maxmov - minmov), 0, 255
                ).astype(np.uint8),
                m,
            )
        )

    if only_raw:
        mov = scale(mov)
    else:
        img_contours = np.repeat(img_contours[..., None], 3, 0).reshape(
            dims[0], 3 * dims[1], -1
        )
        reconstructed = np.tensordot(
            downsample_array(traces.T, downscale, 1, n_jobs=n_jobs).astype("f4"),
            rois,
            1,
        )
        residual = scale(mov - reconstructed)
        mov = scale(mov)
        reconstructed = scale(reconstructed)
        mov = np.concatenate([mov, reconstructed, residual], 2)
        del reconstructed
        del residual
    font = cv2.FONT_HERSHEY_SIMPLEX
    magnify = max(500 // dims[0], 1)
    h, w = dims[0] * magnify, dims[1] * magnify
    canvas_size = (
        int(np.ceil(h * 1.08 / 16)) * 16,
        int(np.ceil((w if only_raw else 3 * w) / 16)) * 16,
    )
    pad = (canvas_size[1] - 3 * w) // 2
    canvas = np.zeros(canvas_size + (3,), np.uint8)
    for i in range(1 if only_raw else 3):
        text = ("Original", "ROI Activity", "Residual")[i]
        fontscale = min(h / 600, w / 190)
        textsize = cv2.getTextSize(text, font, fontscale, max(h // 200, 1))[0]
        cv2.putText(
            canvas,
            text,
            (int(w * (0.49, 1.5, 2.51)[i] + pad - textsize[0] / 2), h // 25),
            font,
            fontscale,
            (255, 255, 255),
            max(h // 200, 1),
            cv2.LINE_4,
        )
    writer = imageio_ffmpeg.write_frames(
        output_path,
        canvas_size[::-1],
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libvpx-vp9",
        fps=fs,
        bitrate=bitrate,
        output_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-cpu-used",
            str(cpu_used),
        ],
    )
    writer.send(None)
    if magnify > 1:
        img_contours = cv2.resize(img_contours, (0, 0), fx=magnify, fy=magnify)
    is_contours = img_contours != 0
    for frame in mov:
        if magnify > 1:
            frame = cv2.resize(frame, (0, 0), fx=magnify, fy=magnify)
        frame = np.repeat(frame[..., None], 3, 2)
        frame[is_contours] = img_contours[is_contours]
        canvas[-h:, -(w if only_raw else 3 * w) :] = frame
        writer.send(canvas)
    writer.close()


# ---------------------------------------------------------------------------
# Suite2p orchestration helpers (called by job.py)
# ---------------------------------------------------------------------------


def run_suite2p_extraction(
    motion_corrected_fn: Path,
    input_fn: Path,
    args: ExtractionSettings,
    frame_rate: float,
    instrument_id: Optional[str],
) -> dict:
    """Run Suite2p ROI detection and return raw outputs.

    Handles the ``init`` modes that use Suite2p (all modes except
    ``greedy_roi``/``corr_pnr``).  Returns a result dict consumed by
    :func:`job.run`.

    Parameters
    ----------
    motion_corrected_fn : Path
        Motion-corrected (or Bergamo-filtered) HDF5 input.
    input_fn : Path
        Original registered HDF5 (used for Bergamo full-movie extraction).
    args : ExtractionSettings
        Extraction settings.
    frame_rate : float
        Resolved frame rate.
    instrument_id : str, optional
        Instrument identifier; ``"Bergamo"`` substring triggers special handling.

    Returns
    -------
    dict
        Keys: ``traces_corrected``, ``traces_neuropil``, ``traces_roi``,
        ``data``, ``coords``, ``neuropil_coords``, ``iscell``,
        ``input_args``, ``nframes``, ``keys``.
    """
    suite2p_args = suite2p.default_ops()
    suite2p_args["diameter"] = args.diameter
    if args.diameter == 0 and args.init == "sourcery":
        with h5py.File(str(motion_corrected_fn), "r") as open_vid:
            suite2p_args["diameter"] = round(
                Cellpose().sz.eval(mean_image(open_vid["data"]))[0]
            )
        logger.info(
            "'diameter' set to 0 — automatically estimated with Cellpose "
            f"as {suite2p_args['diameter']:.0f}."
        )
    suite2p_args["anatomical_only"] = {
        "max/mean": 1,
        "mean": 2,
        "enhanced_mean": 3,
        "max": 4,
    }.get(args.init, 0)
    suite2p_args["cellprob_threshold"] = args.cellprob_threshold
    suite2p_args["flow_threshold"] = args.flow_threshold
    suite2p_args["spatial_hp_cp"] = args.spatial_hp_cp
    suite2p_args["pretrained_model"] = args.pretrained_model
    suite2p_args["denoise"] = args.denoise
    suite2p_args["save_path0"] = str(args.tmp_dir)
    suite2p_args["functional_chan"] = args.functional_chan
    suite2p_args["threshold_scaling"] = args.threshold_scaling
    suite2p_args["max_overlap"] = args.max_overlap
    suite2p_args["soma_crop"] = args.soma_crop
    suite2p_args["allow_overlap"] = args.allow_overlap
    suite2p_args["sparse_mode"] = args.init == "sparsery"
    suite2p_args["h5py"] = str(motion_corrected_fn)
    suite2p_args["data_path"] = []
    suite2p_args["roidetect"] = True
    suite2p_args["do_registration"] = 0
    suite2p_args["spikedetect"] = False
    suite2p_args["fs"] = frame_rate
    suite2p_args["neuropil_extract"] = True
    suite2p_args["bin_duration"] = 3.7
    with h5py.File(suite2p_args["h5py"], "r") as f:
        nframes = f["data"].shape[0]
    bin_size = suite2p_args["bin_duration"] * suite2p_args["fs"]
    suite2p_args["nbinned"] = int(nframes / bin_size)
    logger.info(
        f"Movie has {nframes} frames collected at "
        f"{suite2p_args['fs']} Hz. "
        "To get a bin duration of "
        f"{suite2p_args['bin_duration']} "
        f"seconds, setting nbinned to "
        f"{suite2p_args['nbinned']}."
    )

    reserved_suite2p_keys = {
        "anatomical_only",
        "allow_overlap",
        "bin_duration",
        "cellprob_threshold",
        "data_path",
        "denoise",
        "diameter",
        "do_registration",
        "flow_threshold",
        "fs",
        "functional_chan",
        "h5py",
        "max_overlap",
        "nbinned",
        "neuropil_extract",
        "pretrained_model",
        "roidetect",
        "save_path0",
        "soma_crop",
        "sparse_mode",
        "spatial_hp_cp",
        "spikedetect",
        "threshold_scaling",
    }

    if args.suite2p_ops is not None:
        ops_path = args.suite2p_ops
        if not ops_path.is_file():
            raise ValueError(f"'suite2p_ops' file not found: {ops_path}")
        loaded = np.load(ops_path, allow_pickle=True)
        import copy
        ops_dict = loaded[()] if isinstance(loaded, np.ndarray) else loaded
        if not isinstance(ops_dict, dict):
            raise ValueError(
                f"'suite2p_ops' did not contain a dict (got {type(ops_dict).__name__})"
            )
        applied = {k: v for k, v in ops_dict.items() if k not in reserved_suite2p_keys}
        dropped = sorted(set(ops_dict) - set(applied))
        for k, v in applied.items():
            suite2p_args[k] = copy.deepcopy(v)
        logger.warning(
            f"Loaded {len(applied)} Suite2p parameter(s) from "
            f"'suite2p_ops' ({ops_path}). Applied: {sorted(applied)}. "
            f"Dropped {len(dropped)} reserved key(s): {dropped}"
        )

    if args.suite2p_params:
        try:
            user_suite2p_params = json.loads(args.suite2p_params)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse 'suite2p_params' as JSON: {e}"
            ) from e
        if not isinstance(user_suite2p_params, dict):
            raise ValueError(
                "'suite2p_params' must be a JSON object mapping parameter names to values."
            )
        import copy
        applied_params = {
            k: v
            for k, v in user_suite2p_params.items()
            if k not in reserved_suite2p_keys
        }
        dropped_params = sorted(set(user_suite2p_params) - set(applied_params))
        for k, v in applied_params.items():
            suite2p_args[k] = copy.deepcopy(v)
        logger.warning(
            f"Overriding {len(applied_params)} Suite2p parameter(s) from "
            f"'suite2p_params': {sorted(applied_params)}. "
            f"Dropped {len(dropped_params)} reserved key(s): {dropped_params}"
        )

    logger.info(f"running Suite2P v{suite2p.version}")
    try:
        input_args = {**vars(args), **suite2p_args}
        suite2p.run_s2p(suite2p_args)
    except IndexError:
        pass

    with h5py.File(str(motion_corrected_fn), "r") as open_vid:
        dims = open_vid["data"][0].shape

    tmp_dir = args.tmp_dir.resolve()
    suite2p_stat_path = find_one(tmp_dir, "stat.npy", required=False)
    if suite2p_stat_path is None:
        traces_roi, traces_neuropil, traces_corrected = [
            np.empty((0, nframes), dtype=np.float32)
        ] * 3
        r_values, data, coords, neuropil_coords, iscell = [[]] * 5
        if args.neuropil == "mutualinfo":
            raw_r = []
        keys = []
        return {
            "traces_corrected": traces_corrected,
            "traces_neuropil": traces_neuropil,
            "traces_roi": traces_roi,
            "data": data,
            "coords": coords,
            "neuropil_coords": neuropil_coords,
            "iscell": iscell,
            "input_args": input_args,
            "nframes": nframes,
            "keys": keys,
            "dims": dims,
        }

    suite2p_stats = np.load(str(suite2p_stat_path), allow_pickle=True)

    if args.neuropil in ("mutualinfo", "suite2p"):
        if instrument_id and "Bergamo" in instrument_id:
            (
                stat,
                traces_roi,
                traces_neuropil,
                _,
                _,
            ) = suite2p.extraction.extraction_wrapper(
                suite2p_stats, h5py.File(input_fn)["data"], ops=suite2p_args
            )
        else:
            suite2p_f_path = str(find_one(tmp_dir, "F.npy"))
            suite2p_fneu_path = str(find_one(tmp_dir, "Fneu.npy"))
            traces_roi = np.load(suite2p_f_path, allow_pickle=True)
            traces_neuropil = np.load(suite2p_fneu_path, allow_pickle=True)
        iscell = np.load(str(find_one(tmp_dir, "iscell.npy")))
        if args.neuropil == "suite2p":
            traces_corrected = (
                traces_roi - suite2p_args["neucoeff"] * traces_neuropil
            )
            r_values = suite2p_args["neucoeff"] * np.ones(traces_roi.shape[0])
        else:
            traces_corrected, r_values, raw_r = get_FC_from_r(
                traces_roi, traces_neuropil
            )
        data = []
        coords = []
        neuropil_coords = []
        for i, roi in enumerate(suite2p_stats):
            data.append(roi["lam"])
            coords.append(
                np.array(
                    [i * np.ones(len(roi["lam"])), roi["ypix"], roi["xpix"]],
                    dtype=np.int16,
                )
            )
            neuropil_coords.append(
                np.array(
                    [
                        i * np.ones(len(roi["neuropil_mask"])),
                        roi["neuropil_mask"] // dims[1],
                        roi["neuropil_mask"] % dims[1],
                    ],
                    dtype=np.int16,
                )
            )
        keys = list(suite2p_stats[0].keys())
        for k in ("ypix", "xpix", "lam", "neuropil_mask"):
            keys.remove(k)
        stat = {}
        for k in keys:
            stat[k] = [s[k] for s in suite2p_stats]
        data = np.concatenate(data)
        coords = np.hstack(coords)
        neuropil_coords = np.hstack(neuropil_coords)
        stat["soma_crop"] = np.concatenate(stat["soma_crop"])
        stat["overlap"] = np.concatenate(stat["overlap"])
    else:
        Ain = hstack(
            [
                coo_matrix((roi["lam"], (roi["ypix"], roi["xpix"])), shape=dims)
                .reshape((-1, 1), order="F")
                .tocsc()
                for roi in suite2p_stats
            ]
        )
        ops_path = find_one(tmp_dir, "ops.npy")
        ops = np.load(str(ops_path), allow_pickle=True)[()]
        n_jobs = None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)
        (
            traces_corrected,
            traces_neuropil,
            traces_roi,
            data,
            coords,
            iscell,
        ) = run_caiman_extraction(
            input_fn, args.tmp_dir.resolve().name, args, ops, Ain=Ain, n_jobs=n_jobs
        )
        neuropil_coords, keys = [], []

    return {
        "traces_corrected": traces_corrected,
        "traces_neuropil": traces_neuropil,
        "traces_roi": traces_roi,
        "data": data,
        "coords": coords,
        "neuropil_coords": neuropil_coords,
        "iscell": iscell,
        "input_args": input_args,
        "nframes": nframes,
        "keys": keys,
        "dims": dims,
        "stat": stat if args.neuropil in ("mutualinfo", "suite2p") else {},
        "r_values": r_values if args.neuropil in ("mutualinfo", "suite2p") else [],
        "raw_r": raw_r if args.neuropil == "mutualinfo" else [],
    }


def write_extraction_h5(
    output_dir: Path,
    unique_id: str,
    suite2p_result: dict,
    args: ExtractionSettings,
    motion_corrected_fn: Path,
) -> Path:
    """Write the extraction output HDF5 file.

    Parameters
    ----------
    output_dir : Path
        Per-plane extraction output directory.
    unique_id : str
        Plane identifier.
    suite2p_result : dict
        Result dict from :func:`run_suite2p_extraction` or the CaImAn path.
    args : ExtractionSettings
        Extraction settings.
    motion_corrected_fn : Path
        Motion-corrected movie path (for correlation image).

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    traces_corrected = suite2p_result["traces_corrected"]
    traces_neuropil = suite2p_result["traces_neuropil"]
    traces_roi = suite2p_result["traces_roi"]
    data = suite2p_result["data"]
    coords = suite2p_result["coords"]
    neuropil_coords = suite2p_result["neuropil_coords"]
    iscell = suite2p_result["iscell"]
    keys = suite2p_result.get("keys", [])
    dims = suite2p_result["dims"]
    stat = suite2p_result.get("stat", {})
    r_values = suite2p_result.get("r_values", [])
    raw_r = suite2p_result.get("raw_r", [])

    tmp_dir = args.tmp_dir.resolve()
    cellpose_path = find_one(tmp_dir, "cellpose.npz", required=False)
    ops_path = find_one(tmp_dir, "ops.npy", required=False)
    ops = np.load(str(ops_path), allow_pickle=True)[()] if ops_path else {}

    out_fp = output_dir / f"{unique_id}_extraction.h5"
    shape = np.array([len(traces_roi), *dims], dtype=np.int16)
    with h5py.File(out_fp, "w") as f:
        f.create_dataset("traces/corrected", data=traces_corrected, compression="gzip")
        f.create_dataset("traces/neuropil", data=traces_neuropil, compression="gzip")
        f.create_dataset("traces/roi", data=traces_roi, compression="gzip")
        if args.neuropil in ("mutualinfo", "suite2p"):
            f.create_dataset("traces/neuropil_rcoef", data=r_values)
            if args.neuropil == "mutualinfo":
                f.create_dataset(
                    "traces/raw_neuropil_rcoef_mutualinfo", data=raw_r
                )
        for k in keys:
            dtype = np.array(stat[k]).dtype
            if dtype != "bool":
                dtype = "i2" if np.issubdtype(dtype, np.integer) else "f4"
            if k in ("skew", "std"):
                f.create_dataset(f"traces/{k}", data=stat[k], dtype=dtype)
            else:
                f.create_dataset(f"rois/{k}", data=stat[k], dtype=dtype)
        f.create_dataset("rois/coords", data=coords, compression="gzip")
        f.create_dataset("rois/data", data=data, compression="gzip")
        f.create_dataset("rois/shape", data=shape)
        if len(neuropil_coords) > 0:
            f.create_dataset(
                "rois/neuropil_coords", data=neuropil_coords, compression="gzip"
            )
        if cellpose_path:
            with np.load(str(cellpose_path)) as cp:
                for k in cp.keys():
                    f.create_dataset(f"cellpose/{k}", data=cp[k], compression="gzip")
                roi_masks = sparse.COO(coords, data, shape).todense()
                roi_probs = roi_probabilities(roi_masks, cp["cellprob"])
                dset = f.create_dataset(
                    "rois/cellpose_soma_probability",
                    data=roi_probs,
                    dtype="f4",
                )
                dset.attrs["description"] = (
                    "Mean of per-pixel sigmoid(cellpose cellprob) over "
                    "each ROI footprint; probability in [0, 1], in "
                    "suite2p ROI order (aligned with rois/coords)."
                )
        else:
            logging.warning("No cellpose output found.")
        f.create_dataset("iscell", data=iscell, dtype="f4")
        if ops:
            f.create_dataset("meanImg", data=ops["meanImg"], compression="gzip")
            f.create_dataset("maxImg", data=ops["max_proj"], compression="gzip")

    return out_fp
