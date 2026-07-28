"""Top-level extraction orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> locate the registered
movie -> run Suite2p/CaImAn -> write output HDF5 + diagnostics ->
emit a full ``processing.json`` and ``quality_control.json``.
"""

import logging
import os
import sys
from datetime import datetime as dt
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import sparse
from aind_ophys_utils.summary_images import max_corr_image, mean_image, max_image
from aind_pophys_metadata import io
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

logger = logging.getLogger(
    "Source extraction using a combination of Cellpose, Suite2p, and CaImAn"
)

PROCESS_NAME = "aind-ophys-extraction"
_BERGAMO_INSTRUMENT = "Bergamo"


def _load_session_raw(input_dir: Path) -> dict:
    """Load the raw v1 session dict for Bergamo photostim-frame filtering.

    Returns an empty dict when no ``session.json`` is present (v2-only inputs).
    """
    session_fp = io.find(input_dir, "session.json")
    return io.load_json(session_fp) if session_fp else {}


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects."""
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json(
        (Path(output_dir) / "processing.json").read_text()
    )
    QualityControl.model_validate_json(
        (Path(output_dir) / "quality_control.json").read_text()
    )


def run(settings: Optional[ExtractionSettings] = None) -> None:
    """Run ophys ROI extraction end to end.

    Parameters
    ----------
    settings : ExtractionSettings, optional
        Parsed settings; constructed from CLI/env when omitted.
    """
    settings = settings or ExtractionSettings()
    start_time = dt.now()

    metadata = mu.load_extraction_metadata(settings.input_dir)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info("stage_start: extraction")

    # Env overrides for linear algebra thread counts (CaImAn requirement)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["CAIMAN_TEMP"] = str(settings.tmp_dir)

    output_dir = settings.output_dir.resolve()
    input_dir = settings.input_dir.resolve()
    tmp_dir = settings.tmp_dir.resolve()

    if next(input_dir.glob("output"), ""):
        sys.exit()

    # Locate input movie
    if next(input_dir.rglob("*decrosstalk.h5"), ""):
        input_fn = next(input_dir.rglob("*decrosstalk.h5"))
    else:
        input_fn = next(input_dir.rglob("*registered.h5"))

    unique_id = input_fn.parent.parent.name

    # Bergamo: filter out photostimulation frames via a virtual dataset
    session_raw = _load_session_raw(input_dir)
    if metadata.instrument_id and _BERGAMO_INSTRUMENT in metadata.instrument_id:
        motion_corrected_fn = bergamo_segmentation(
            input_fn, session_raw, temp_dir=tmp_dir
        )
    else:
        motion_corrected_fn = input_fn

    # Resolve frame rate
    frame_rate = metadata.frame_rate
    if frame_rate is None:
        frame_rate = settings.frame_rate
    if frame_rate is None:
        raise ValueError(
            "frame_rate could not be resolved from metadata or --frame_rate"
        )

    output_dir = make_output_directory(output_dir, unique_id)
    n_jobs = None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)

    if settings.init in ("greedy_roi", "corr_pnr"):
        # CaImAn-only path
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
        result = {
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
    else:
        result = run_suite2p_extraction(
            motion_corrected_fn=motion_corrected_fn,
            input_fn=input_fn,
            args=settings,
            frame_rate=frame_rate,
            instrument_id=metadata.instrument_id,
        )

    write_extraction_h5(
        output_dir=output_dir,
        unique_id=unique_id,
        suite2p_result=result,
        args=settings,
        motion_corrected_fn=motion_corrected_fn,
    )

    # Write metadata outputs
    input_args = result.get("input_args", settings.model_dump())
    data_process = mu.build_data_process(
        parameters=input_args,
        start_time=start_time,
        end_time=dt.now(),
        input_data=[metadata.dataset_name] if metadata.dataset_name else None,
        output_path=str(output_dir),
    )
    mu.write_processing_json(mu.build_processing([data_process]), output_dir)

    # Visualisation
    data_arr = result["data"]
    coords_arr = result["coords"]
    dims = result["dims"]
    iscell = result["iscell"]
    shape = np.array([len(result["traces_roi"]), *dims], dtype=np.int16)

    if len(data_arr) > 0 and len(coords_arr) > 0:
        rois = sparse.COO(coords_arr, data_arr, shape)
    else:
        rois = sparse.COO(
            np.zeros((3, 0), dtype=np.int16),
            np.zeros(0),
            shape,
        )

    # Load ops for summary images (may be empty for CaImAn path)
    tmp_dir = settings.tmp_dir.resolve()
    ops_path = str(next(tmp_dir.rglob("ops.npy"), ""))
    if ops_path:
        ops = np.load(ops_path, allow_pickle=True)[()]
    else:
        with h5py.File(str(motion_corrected_fn), "r") as open_vid:
            ops = {
                "meanImg": mean_image(open_vid["data"]),
                "max_proj": max_image(open_vid["data"]),
            }

    with h5py.File(str(motion_corrected_fn), "r") as f:
        corr_img = max_corr_image(f["data"])

    if rois.shape[0] > 0:
        save_summary_images_with_rois(output_dir, unique_id, rois, iscell, ops, corr_img)

    if settings.contour_video and rois.shape[0] > 0:
        with h5py.File(str(motion_corrected_fn), "r") as f:
            contour_video(
                output_dir / f"{unique_id}_ROI_contours_overlay.webm",
                f["data"],
                rois,
                result["traces_corrected"],
                fs=frame_rate,
            )

    # QC
    reference = str(
        (output_dir / f"{unique_id}_detected_ROIs_withIDs.png").relative_to(
            settings.output_dir.resolve()
        )
    )
    metric = qc.build_detected_rois_metric(
        unique_id=unique_id,
        num_rois=rois.shape[0],
        reference=reference,
    )
    qc.write_quality_control_json(qc.build_quality_control([metric]), output_dir)

    logger.info("stage_complete: extraction")
