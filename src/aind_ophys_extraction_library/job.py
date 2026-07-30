"""Top-level extraction orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> locate the registered
movie -> run Suite2p/CaImAn -> write output HDF5 + diagnostics ->
emit a full ``processing.json`` and ``quality_control.json``.

``processing.json`` is written from a ``finally`` block so a failed run
still leaves a document describing what was attempted, with the error
recorded in ``notes``.
"""

import logging
import os
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
from .utils.paths import relative_to_root

logger = logging.getLogger("Ophys extraction")

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


def _locate_input_movie(input_dir: Path) -> Path:
    """Return the decrosstalked movie if present, else the registered one."""
    decrosstalk = next(input_dir.rglob("*decrosstalk.h5"), None)
    if decrosstalk is not None:
        return decrosstalk
    return next(input_dir.rglob("*registered.h5"))


def _resolve_frame_rate(
    metadata: mu.ExtractionMetadata, settings: ExtractionSettings
) -> float:
    """Resolve the frame rate from metadata, falling back to the CLI value."""
    frame_rate = metadata.frame_rate
    if frame_rate is None:
        frame_rate = settings.frame_rate
        logger.warning("Using CLI fallback frame rate: %s", settings.frame_rate)
    if frame_rate is None:
        raise ValueError(
            "frame_rate could not be resolved from metadata or --frame_rate"
        )
    return frame_rate


def _run_detection(
    settings: ExtractionSettings,
    metadata: mu.ExtractionMetadata,
    motion_corrected_fn: Path,
    input_fn: Path,
    unique_id: str,
    frame_rate: float,
) -> dict:
    """Run the CaImAn-only or Suite2p detection path and return its result."""
    n_jobs = None if (tmp := os.environ.get("CO_CPUS")) is None else int(tmp)

    if settings.init not in ("greedy_roi", "corr_pnr"):
        return run_suite2p_extraction(
            motion_corrected_fn=motion_corrected_fn,
            input_fn=input_fn,
            args=settings,
            frame_rate=frame_rate,
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
    ops_path = str(next(tmp_dir.rglob("ops.npy"), ""))
    if ops_path:
        return np.load(ops_path, allow_pickle=True)[()]
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


def _write_processing_metadata(
    settings: ExtractionSettings,
    ctx: dict,
    start_time: dt,
    error: Optional[BaseException],
) -> None:
    """Best-effort writer for processing.json. Never raises.

    Failures here must not mask an extraction failure already propagating
    out of ``run()``. Falls back to the results root when the run failed
    before the per-plane output directory was created.
    """
    results_root = settings.output_dir.resolve()
    output_dir = ctx.get("output_dir") or results_root
    try:
        data_process = mu.build_data_process(
            parameters=ctx.get("parameters") or settings.model_dump(),
            start_time=start_time,
            end_time=dt.now(),
            input_data=ctx.get("input_data"),
            output_path=relative_to_root(results_root, output_dir),
            error=error,
        )
        mu.write_processing_json(mu.build_processing([data_process]), output_dir)
    except Exception as proc_exc:
        logger.error(f"processing.json write failed (non-fatal): {proc_exc}")


def _run_extraction(
    settings: ExtractionSettings,
    metadata: mu.ExtractionMetadata,
    ctx: dict,
) -> None:
    """Locate the input movie, detect ROIs, and write all outputs.

    ``ctx`` is populated as the run progresses so the ``processing.json``
    writer in ``run()``'s ``finally`` can report partial provenance even
    when this raises.
    """
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["CAIMAN_TEMP"] = str(settings.tmp_dir)

    results_root = settings.output_dir.resolve()
    input_dir = settings.input_dir.resolve()
    tmp_dir = settings.tmp_dir.resolve()

    input_fn = _locate_input_movie(input_dir)
    unique_id = input_fn.parent.parent.name
    ctx["input_data"] = [relative_to_root(input_dir, input_fn)]

    session_raw = _load_session_raw(input_dir)
    if metadata.instrument_id and _BERGAMO_INSTRUMENT in metadata.instrument_id:
        motion_corrected_fn = bergamo_segmentation(
            input_fn, session_raw, temp_dir=tmp_dir
        )
    else:
        motion_corrected_fn = input_fn

    frame_rate = _resolve_frame_rate(metadata, settings)
    output_dir = make_output_directory(results_root, unique_id)
    ctx["output_dir"] = output_dir

    result = _run_detection(
        settings,
        metadata,
        motion_corrected_fn,
        input_fn,
        unique_id,
        frame_rate,
    )
    ctx["parameters"] = result.get("input_args", settings.model_dump())

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
        frame_rate,
    )
    _write_quality_control(
        results_root, output_dir, unique_id, rois.shape[0]
    )


def run(settings: Optional[ExtractionSettings] = None) -> None:
    """Run ophys ROI extraction end to end.

    Parameters
    ----------
    settings : ExtractionSettings, optional
        Parsed settings; constructed from CLI/env when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
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
    logger.info("Starting extraction", extra={"event_type": "stage_start"})

    # Already-processed guard: a prior run's output is mounted alongside the
    # input. Skip cleanly (no processing.json for a no-op run).
    existing_output = next(settings.input_dir.resolve().glob("output"), None)
    if existing_output is not None:
        logger.info(
            "Found %s in the input directory; this plane is already "
            "processed. Exiting without running extraction.",
            existing_output,
            extra={"event_type": "stage_complete"},
        )
        return

    ctx: dict = {}
    run_error: Optional[BaseException] = None
    try:
        _run_extraction(settings, metadata, ctx)
        logger.info(
            "Extraction complete", extra={"event_type": "stage_complete"}
        )
    except BaseException as exc:
        run_error = exc
        logger.error("Extraction failed", extra={"event_type": "stage_error"})
        logger.exception("Unhandled exception")
        raise
    finally:
        _write_processing_metadata(settings, ctx, start_time, run_error)

    if settings.verify:
        _verify_outputs(ctx["output_dir"])
