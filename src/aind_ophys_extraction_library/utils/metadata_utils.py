"""aind-data-schema v1/v2 metadata reads for ophys extraction.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The v1/v2 dict paths, schema-version dispatch, path helpers and the
``Processing`` builders live in the shared library; this module only assembles
the extraction ``ExtractionMetadata`` view, the extraction ``Code`` block, and
the output summary written to ``DataProcess.output_parameters``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing

#: Distribution name of this backing library. The shared builder derives both
#: ``Code.url`` and ``Code.version`` from it.
LIBRARY_NAME = "aind-ophys-extraction-library"

CODE_NAME = "aind-ophys-extraction"

PROCESS_TYPE = ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION

#: Substring identifying a Bergamo rig in ``instrument_id``.
BERGAMO_INSTRUMENT = "Bergamo"

#: Stimulus epochs with this name are photostimulation and are excluded from
#: the movie Bergamo extraction runs on.
PHOTOSTIM_STIMULUS_NAME = "2p photostimulation"

#: v1 carried the TIFF stem on ``stimulus_epochs[*].output_parameters``. The
#: v1 -> v2 upgrade drops that field, but keeps the same value on the
#: correspondingly-timed data stream's ``notes``, as ``tiff_stem:<stem>``.
TIFF_STEM_NOTE_PREFIX = "tiff_stem:"


def is_bergamo(instrument_id: Optional[str]) -> bool:
    """Return whether ``instrument_id`` names a Bergamo rig.

    Parameters
    ----------
    instrument_id : str or None
        Instrument/rig identifier from the core acquisition file.

    Returns
    -------
    bool
        ``True`` when the identifier contains the Bergamo marker.
    """
    return bool(instrument_id) and BERGAMO_INSTRUMENT in instrument_id


@dataclass(frozen=True)
class ExtractionMetadata:
    """Minimal metadata the extraction capsule consumes.

    ``frame_rate`` drives Suite2p temporal binning and is validated at load
    time, so a rate-less asset fails with a message naming the file and schema
    version rather than partway through ``run()``. ``instrument_id`` is used to
    detect Bergamo rigs; ``valid_epoch_stems`` carries the Bergamo TIFF stems
    that survive photostimulation filtering, resolved from whichever schema
    version the input carries. The remaining fields are identifiers used for
    logging and output naming.
    """

    frame_rate: float
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    valid_epoch_stems: Tuple[str, ...] = ()


def _v1_epoch_stems(core_raw: dict, core_file: Path) -> Tuple[str, ...]:
    """Return non-photostimulation TIFF stems from a v1 ``session.json``.

    Parameters
    ----------
    core_raw : dict
        Raw ``session.json`` dict.
    core_file : Path
        Source file path, named in the error message.

    Returns
    -------
    tuple of str
        TIFF stems in stimulus-epoch order.

    Raises
    ------
    ValueError
        If an included epoch carries no ``output_parameters.tiff_stem``.
    """
    stems: List[str] = []
    for epoch in core_raw.get("stimulus_epochs") or []:
        if epoch.get("stimulus_name") == PHOTOSTIM_STIMULUS_NAME:
            continue
        stem = (epoch.get("output_parameters") or {}).get("tiff_stem")
        if not stem:
            raise ValueError(
                f"Stimulus epoch {epoch.get('stimulus_name')!r} in "
                f"{Path(core_file).name} has no "
                "output_parameters.tiff_stem, so its frames cannot be "
                "located in epoch_locations.json."
            )
        stems.append(stem)
    return tuple(stems)


def _v2_stems_by_start_time(core_raw: dict) -> Dict[Any, str]:
    """Index v2 data-stream TIFF stems by stream start time.

    Parameters
    ----------
    core_raw : dict
        Raw ``acquisition.json`` dict.

    Returns
    -------
    dict
        Mapping of ``stream_start_time`` to TIFF stem.
    """
    stems: Dict[Any, str] = {}
    for stream in core_raw.get("data_streams") or []:
        notes = stream.get("notes") or ""
        if notes.startswith(TIFF_STEM_NOTE_PREFIX):
            stems[stream.get("stream_start_time")] = notes[
                len(TIFF_STEM_NOTE_PREFIX):
            ]
    return stems


def _v2_epoch_stems(core_raw: dict, core_file: Path) -> Tuple[str, ...]:
    """Return non-photostimulation TIFF stems from a v2 ``acquisition.json``.

    The v1 -> v2 upgrade drops ``stimulus_epochs[*].output_parameters``
    entirely, so the stem is recovered from the data stream covering the same
    time span, whose ``notes`` the upgrade preserves verbatim.

    Parameters
    ----------
    core_raw : dict
        Raw ``acquisition.json`` dict.
    core_file : Path
        Source file path, named in the error message.

    Returns
    -------
    tuple of str
        TIFF stems in stimulus-epoch order.

    Raises
    ------
    ValueError
        If no data stream shares an included epoch's start time.
    """
    by_start_time = _v2_stems_by_start_time(core_raw)
    stems: List[str] = []
    for epoch in core_raw.get("stimulus_epochs") or []:
        if epoch.get("stimulus_name") == PHOTOSTIM_STIMULUS_NAME:
            continue
        start_time = epoch.get("stimulus_start_time")
        stem = by_start_time.get(start_time)
        if not stem:
            raise ValueError(
                f"Stimulus epoch {epoch.get('stimulus_name')!r} starting "
                f"{start_time} in {Path(core_file).name} has no data stream "
                f"with a {TIFF_STEM_NOTE_PREFIX!r} note at the same start "
                "time, so its frames cannot be located in "
                "epoch_locations.json."
            )
        stems.append(stem)
    return tuple(stems)


def get_valid_epoch_stems(
    version: str, core_raw: dict, core_file: Path
) -> Tuple[str, ...]:
    """TIFF stems of the epochs Bergamo extraction keeps, either version.

    Parameters
    ----------
    version : str
        Schema version (:data:`aind_pophys_metadata.io.SCHEMA_V1` or
        ``SCHEMA_V2``).
    core_raw : dict
        Raw ``session.json`` (v1) or ``acquisition.json`` (v2) dict.
    core_file : Path
        Source file path, named in any error message.

    Returns
    -------
    tuple of str
        TIFF stems in stimulus-epoch order, photostimulation excluded.
    """
    if version == io.SCHEMA_V1:
        return _v1_epoch_stems(core_raw, core_file)
    return _v2_epoch_stems(core_raw, core_file)


def load_extraction_metadata(
    input_dir: Path, frame_rate_override: Optional[float] = None
) -> ExtractionMetadata:
    """Read v1/v2 metadata for extraction from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). ``frame_rate`` falls back to ``platform.json``
    and then to ``frame_rate_override``, and is fatal here rather than deep in
    the run. ``subject_id`` is read from the core file (canonical) and falls
    back to ``subject.json``. ``dataset_name`` comes from
    ``data_description.json``.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    frame_rate_override : float, optional
        Operator-supplied frame rate, used only when the metadata carries
        none.

    Returns
    -------
    ExtractionMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input), or if no source
        supplies a frame rate.
    """
    input_dir = Path(input_dir)
    common = io.load_common(input_dir)
    instrument_id = fields.get_instrument_id(common.version, common.core_raw)
    return ExtractionMetadata(
        frame_rate=fields.resolve_frame_rate(
            common.version,
            common.core_raw,
            common.platform_raw,
            core_file=common.core_path,
            input_dir=input_dir,
            cli_override=frame_rate_override,
            required=True,
        ),
        instrument_id=instrument_id,
        subject_id=fields.get_subject_id(
            common.core_raw, common.subject_raw
        ),
        dataset_name=fields.get_dataset_name(common.data_description_raw),
        valid_epoch_stems=(
            get_valid_epoch_stems(
                common.version, common.core_raw, common.core_path
            )
            if is_bergamo(instrument_id)
            else ()
        ),
    )


def process_name(plane_id: str) -> str:
    """Return the ``DataProcess.name`` this capsule emits for a plane.

    Mirrors the composition
    :func:`aind_pophys_metadata.processing.build_data_process` performs, so
    the name can be excluded from the upstream dependency scan before the
    process itself is built.

    Parameters
    ----------
    plane_id : str
        Plane/FOV id this run processed.

    Returns
    -------
    str
        The qualified process name.
    """
    return f"{plane_id}: {PROCESS_TYPE.value}"


def build_extraction_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the v2 ``Code`` block for extraction.

    Parameters
    ----------
    parameters : dict
        Capsule configuration parameters (from ``ExtractionSettings``). This
        is the record of what the step was *asked* to do; output statistics
        belong in ``output_parameters`` instead.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance, relative to the
        input root.

    Returns
    -------
    Code
        The populated code block. Url and version both come from the
        installed backing library, so the two always describe one artifact.
    """
    return shared_processing.build_code(
        CODE_NAME,
        library_name=LIBRARY_NAME,
        parameters=parameters,
        input_data=input_data,
    )


def _count_cells(iscell) -> Optional[int]:
    """Return how many ROIs Suite2p/CaImAn classified as cells.

    Parameters
    ----------
    iscell : sequence
        Per-ROI classification, either a sequence of scalars or the
        ``(n_rois, 2)`` array Suite2p and CaImAn both emit.

    Returns
    -------
    int or None
        The number of ROIs flagged as cells, or ``None`` when the structure
        is not recognised.
    """
    try:
        return sum(
            int(bool(entry[0] if hasattr(entry, "__len__") else entry))
            for entry in iscell
        )
    except (TypeError, IndexError, KeyError, ValueError):
        return None


def build_output_parameters(
    result: dict, num_rois: int, frame_rate: float
) -> Dict[str, Any]:
    """Summarize this step's outputs for ``DataProcess.output_parameters``.

    The convention this fills, shared across the pophys fleet: scalar summary
    statistics describing the outputs this step *produced*, chosen so a run
    can be triaged from the metadata alone without opening the data. It is
    explicitly not a place for input parameters, which belong in
    ``Code.parameters``.

    Parameters
    ----------
    result : dict
        Detection result from the Suite2p or CaImAn path.
    num_rois : int
        Number of ROI masks written to the output HDF5.
    frame_rate : float
        Frame rate the traces were extracted at, in Hz.

    Returns
    -------
    dict
        Scalar summary statistics. Keys whose value could not be determined
        are omitted rather than recorded as ``None``.
    """
    dims = result.get("dims") or ()
    summary: Dict[str, Any] = {
        "n_rois": int(num_rois),
        "n_traces": len(result.get("traces_corrected") or []),
        "n_frames": int(result.get("nframes") or 0),
        "frame_rate_hz": float(frame_rate),
    }
    num_cells = _count_cells(result.get("iscell") or [])
    if num_cells is not None:
        summary["n_rois_classified_as_cells"] = num_cells
    if len(dims) == 2:
        summary["movie_height_px"] = int(dims[0])
        summary["movie_width_px"] = int(dims[1])
    return summary
