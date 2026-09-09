"""Top-level extraction orchestration.

``run()`` is the library entry point called by the thin runtime wrapper:
read metadata (v1 or v2) -> configure logging -> locate the input movie
(the decrosstalked one when present, else the registered one) -> run
Suite2p/CaImAn -> write output HDF5 + diagnostics -> emit a full
``processing.json`` and ``quality_control.json``.

The shared ``stage_guard`` owns the ``processing.json`` lifecycle, so a
failed run still leaves a document describing what was attempted, with the
error recorded in ``notes``.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import sparse
from aind_ophys_utils.summary_images import max_corr_image, mean_image, max_image
from aind_pophys_metadata import find_one, relative_to_root
from aind_pophys_metadata.processing import collect_upstream_process_names
from aind_pophys_metadata.runner import stage_guard
from log_schema.main import setup_logging

from . import qc
from .extraction import (
    bergamo_segmentation,
    make_output_directory,
    run_caiman_extraction,
    run_suite2p_extraction,
    save_summary_images_with_rois,
    contour_video,
    write_extraction_h5,
)
from .settings import ExtractionSettings
from .utils import metadata_utils as mu

logger = logging.getLogger("Ophys extraction")

PROCESS_NAME = "aind-ophys-extraction"

#: A prior run's output mounted alongside the input marks this plane done.
EXISTING_OUTPUT_DIR = "output"


def _locate_input_movie(input_dir: Path) -> Path:
    """Return the decrosstalked movie if present, else the registered one."""
    decrosstalk = find_one(input_dir, "*decrosstalk.h5", required=False)
    if decrosstalk is not None:
        return decrosstalk
    return find_one(input_dir, "*registered.h5")


def _run_detection(
    settings: ExtractionSettings,
    metadata: mu.ExtractionMetadata,
    motion_corrected_fn: Path,
    input_fn: Path,
    unique_id: str,
) -> dict:
    """Run the CaImAn-only or Suite2p detection path and return its result."""
    n_jobs = None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)

    if settings.init not in ("greedy_roi", "corr_pnr"):
        return run_suite2p_extraction(
            motion_corrected_fn=motion_corrected_fn,
            input_fn=input_fn,
            args=settings,
            frame_rate=metadata.frame_rate,
            instrument_id=metadata.instrument_id,
        )

    with h5py.File(str(motion_corrected_fn), "r") as open_vid:
        dims = open_vid["data"][0].shape
        ops = {
            "meanImg": mean_image(open_vid["data"]),
            "max_proj": max_image(open_vid["data"]),
        }
    (
        traces_corrected,
        traces_neuropil,
        traces_roi,
        data,
        coords,
        iscell,
    ) = run_caiman_extraction(
        input_fn, unique_id, settings, ops, Ain=None, n_jobs=n_jobs
    )
    with h5py.File(str(motion_corrected_fn), "r") as f:
        nframes = f["data"].shape[0]
    return {
        "traces_corrected": traces_corrected,
        "traces_neuropil": traces_neuropil,
        "traces_roi": traces_roi,
        "data": data,
        "coords": coords,
        "neuropil_coords": [],
        "iscell": iscell,
        "input_args": settings.model_dump(),
        "nframes": nframes,
        "keys": [],
        "dims": dims,
        "stat": {},
        "r_values": [],
        "raw_r": [],
    }


def _build_rois(result: dict) -> sparse.COO:
    """Assemble the sparse ROI mask array from the detection result."""
    data_arr = result["data"]
    coords_arr = result["coords"]
    shape = np.array(
        [len(result["traces_roi"]), *result["dims"]], dtype=np.int16
    )
    if len(data_arr) > 0 and len(coords_arr) > 0:
        return sparse.COO(coords_arr, data_arr, shape)
    return sparse.COO(np.zeros((3, 0), dtype=np.int16), np.zeros(0), shape)


def _load_ops(tmp_dir: Path, motion_corrected_fn: Path) -> dict:
    """Load Suite2p ops, computing summary images for the CaImAn path."""
    ops_path = find_one(tmp_dir, "ops.npy", required=False)
    if ops_path is not None:
        return np.load(str(ops_path), allow_pickle=True)[()]
    with h5py.File(str(motion_corrected_fn), "r") as open_vid:
        return {
            "meanImg": mean_image(open_vid["data"]),
            "max_proj": max_image(open_vid["data"]),
        }


def _write_visualisations(
    settings: ExtractionSettings,
    output_dir: Path,
    unique_id: str,
    motion_corrected_fn: Path,
    result: dict,
    rois: sparse.COO,
    frame_rate: float,
) -> None:
    """Write summary images with ROI overlays and the optional contour video."""
    ops = _load_ops(settings.tmp_dir.resolve(), motion_corrected_fn)
    with h5py.File(str(motion_corrected_fn), "r") as f:
        corr_img = max_corr_image(f["data"])

    if rois.shape[0] > 0:
        save_summary_images_with_rois(
            output_dir, unique_id, rois, result["iscell"], ops, corr_img
        )

    if settings.contour_video and rois.shape[0] > 0:
        with h5py.File(str(motion_corrected_fn), "r") as f:
            contour_video(
                output_dir / f"{unique_id}_ROI_contours_overlay.webm",
                f["data"],
                rois,
                result["traces_corrected"],
                fs=frame_rate,
            )


def _write_quality_control(
    results_root: Path, output_dir: Path, unique_id: str, num_rois: int
) -> None:
    """Emit one full quality_control.json for this plane."""
    reference = relative_to_root(
        results_root, output_dir / f"{unique_id}_detected_ROIs_withIDs.png"
    )
    metric = qc.build_detected_rois_metric(
        unique_id=unique_id,
        num_rois=num_rois,
        reference=reference,
    )
    qc.write_quality_control_json(qc.build_quality_control([metric]), output_dir)


def _run_extraction(
    settings: ExtractionSettings,
    metadata: mu.ExtractionMetadata,
    input_fn: Path,
    output_dir: Path,
    unique_id: str,
) -> dict:
    """Detect ROIs and write every output for one plane.

    Parameters
    ----------
    settings : ExtractionSettings
        Parsed settings.
    metadata : ExtractionMetadata
        Metadata view for this asset.
    input_fn : Path
        Located input movie.
    output_dir : Path
        Per-plane extraction output directory.
    unique_id : str
        Plane/FOV id.

    Returns
    -------
    dict
        Summary statistics for ``DataProcess.output_parameters``.
    """
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["CAIMAN_TEMP"] = str(settings.tmp_dir)

    results_root = settings.output_dir.resolve()
    tmp_dir = settings.tmp_dir.resolve()

    if mu.is_bergamo(metadata.instrument_id):
        motion_corrected_fn = bergamo_segmentation(
            input_fn, metadata.valid_epoch_stems, temp_dir=tmp_dir
        )
    else:
        motion_corrected_fn = input_fn

    result = _run_detection(
        settings,
        metadata,
        motion_corrected_fn,
        input_fn,
        unique_id,
    )

    write_extraction_h5(
        output_dir=output_dir,
        unique_id=unique_id,
        suite2p_result=result,
        args=settings,
        motion_corrected_fn=motion_corrected_fn,
    )

    rois = _build_rois(result)
    _write_visualisations(
        settings,
        output_dir,
        unique_id,
        motion_corrected_fn,
        result,
        rois,
        metadata.frame_rate,
    )
    _write_quality_control(
        results_root, output_dir, unique_id, rois.shape[0]
    )
    return mu.build_output_parameters(
        result, rois.shape[0], metadata.frame_rate
    )


def run(settings: Optional[ExtractionSettings] = None) -> None:
    """Run ophys ROI extraction end to end.

    Parameters
    ----------
    settings : ExtractionSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or ExtractionSettings()
    input_dir = settings.input_dir.resolve()
    results_root = settings.output_dir.resolve()

    metadata = mu.load_extraction_metadata(
        input_dir, frame_rate_override=settings.frame_rate
    )
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )

    # Already-processed guard: a prior run's output is mounted alongside the
    # input. Skip cleanly (no processing.json for a no-op run).
    existing_output = find_one(
        input_dir, EXISTING_OUTPUT_DIR, recursive=False, required=False
    )
    if existing_output is not None:
        logger.info(
            "Found %s in the input directory; this plane is already "
            "processed. Exiting without running extraction.",
            existing_output,
        )
        return

    input_fn = _locate_input_movie(input_dir)
    unique_id = input_fn.parent.parent.name
    output_dir = make_output_directory(results_root, unique_id)
    code = mu.build_extraction_code(
        parameters=settings.model_dump(),
        input_data=[relative_to_root(input_dir, input_fn)],
    )

    with stage_guard(
        output_dir,
        mu.PROCESS_TYPE,
        code,
        plane_id=unique_id,
        logger=logger,
        output_path=relative_to_root(results_root, output_dir),
    ) as ctx:
        ctx.upstream_names = collect_upstream_process_names(
            input_dir, exclude=mu.process_name(unique_id)
        )
        # Which cellpose files this run will load, and which of them are
        # already local, is only observable before detection starts; their
        # digests only exist afterwards. Recorded in a finally so a failed
        # run still names the model bytes it got as far as loading.
        model_plan = mu.plan_cellpose_models(
            settings.init, settings.diameter, settings.pretrained_model
        )
        try:
            ctx.output_parameters = _run_extraction(
                settings, metadata, input_fn, output_dir, unique_id
            )
        finally:
            if model_plan is not None:
                ctx.parameters[mu.CELLPOSE_MODELS_PARAMETER] = (
                    mu.build_cellpose_provenance(model_plan)
                )
