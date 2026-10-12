"""Acquisition metadata reads for ophys extraction.

A thin, extraction-specific layer over the shared ``aind_pophys_metadata``
library. Every schema-version difference is resolved by
:class:`~aind_pophys_metadata.core.CoreMetadata`, which reads one located
document regardless of the schema it is written in; this module assembles the
extraction ``ExtractionMetadata`` view, the extraction ``Code`` block, and the
output summary written to ``DataProcess.output_parameters``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata import (
    CoreMetadata,
)


logger = logging.getLogger("Ophys extraction")

#: Distribution name of this backing library. The shared builder derives
#: ``Code.version`` from it.
LIBRARY_NAME = "aind-ophys-extraction-library"

#: Repository this code lives in, recorded as ``Code.url``. Only this repo
#: knows where its own code lives, so the shared builder is told.
REPOSITORY_URL = (
    "https://github.com/AllenNeuralDynamics/aind-ophys-extraction-library"
)

CODE_NAME = "aind-ophys-extraction"

PROCESS_TYPE = ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION

#: Substring identifying a Bergamo rig in ``instrument_id``.
BERGAMO_INSTRUMENT = "Bergamo"

#: Stimulus epochs with this name are photostimulation and are excluded from
#: the movie Bergamo extraction runs on.
PHOTOSTIM_STIMULUS_NAME = "2p photostimulation"


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
    """Minimal metadata the extraction step consumes.

    ``frame_rate`` drives Suite2p temporal binning and is validated at load
    time, so a rate-less asset fails with a message naming the file rather
    than partway through ``run()``. ``instrument_id`` is used to detect
    Bergamo rigs; ``valid_epoch_stems`` carries the Bergamo TIFF stems that
    survive photostimulation filtering. The remaining fields are identifiers
    used for logging and output naming.
    """

    frame_rate: float
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    valid_epoch_stems: Tuple[str, ...] = ()


def valid_epoch_stems(core: CoreMetadata) -> Tuple[str, ...]:
    """TIFF stems of the stimulus epochs Bergamo extraction keeps.

    Photostimulation epochs are excluded: their frames are not part of the
    movie extraction runs on. The filter matches on the stimulus name, which
    is what production has always keyed on; the epoch's ``stimulus_modalities``
    is the same epoch's second signal but is deliberately not the criterion here.

    Order is exactly the order the acquisition metadata records, which is not
    chronological in real assets. It must not be sorted: each stem is paired
    positionally with a frame range in ``epoch_locations.json``.

    Parameters
    ----------
    core : aind_pophys_metadata.CoreMetadata
        Located core acquisition metadata.

    Returns
    -------
    tuple of str
        TIFF stems in recorded epoch order, photostimulation excluded.

    Raises
    ------
    ValueError
        If a kept epoch carries no TIFF stem, since its frames then cannot be
        located in ``epoch_locations.json``.
    """
    stems: List[str] = []
    for epoch in core.get_epoch_records():
        if epoch["stimulus_name"] == PHOTOSTIM_STIMULUS_NAME:
            continue
        if not epoch["tiff_stem"]:
            raise ValueError(
                f"Stimulus epoch {epoch['stimulus_name']!r} in "
                f"{core.core_path.name} carries no TIFF stem in the "
                "acquisition metadata, so its frames cannot be located in "
                "epoch_locations.json."
            )
        stems.append(epoch["tiff_stem"])
    return tuple(stems)


def load_extraction_metadata(
    input_dir: Path, frame_rate_override: Optional[float] = None
) -> ExtractionMetadata:
    """Read the acquisition metadata extraction consumes from ``input_dir``.

    The shared loader locates and reads the core file; a missing frame rate
    is fatal here rather than deep in the run.

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
        The minimal fields this step consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If the input is ambiguous or malformed, or if no source supplies a
        frame rate.
    """
    core = CoreMetadata.load(Path(input_dir))
    instrument_id = core.get_instrument_id()
    return ExtractionMetadata(
        frame_rate=core.get_frame_rate(
            cli_override=frame_rate_override,
            required=True,
        ),
        instrument_id=instrument_id,
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
        valid_epoch_stems=(
            valid_epoch_stems(core) if is_bergamo(instrument_id) else ()
        ),
    )


def process_name(plane_id: str) -> str:
    """Return the ``DataProcess.name`` this step emits for a plane.

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
        Configuration parameters (from ``ExtractionSettings``). This
        is the record of what the step was *asked* to do; output statistics
        belong in ``output_parameters`` instead.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance, relative to the
        input root.

    Returns
    -------
    Code
        The populated code block. The version comes from the installed
        backing library, whose repository :data:`REPOSITORY_URL` names.
    """
    return shared_processing.build_code(
        CODE_NAME,
        library_name=LIBRARY_NAME,
        url=REPOSITORY_URL,
        parameters=parameters,
        input_data=input_data,
    )


def _count_cells(iscell: Any) -> Optional[int]:
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
    # `result` holds numpy arrays, so `x or default` is unusable here: it
    # calls __bool__ on the array and raises for anything longer than one
    # element. Test against None explicitly.
    dims = result.get("dims")
    dims = () if dims is None else tuple(dims)
    traces = result.get("traces_corrected")
    nframes = result.get("nframes")
    summary: Dict[str, Any] = {
        "n_rois": int(num_rois),
        "n_traces": 0 if traces is None else len(traces),
        "n_frames": 0 if nframes is None else int(nframes),
        "frame_rate_hz": float(frame_rate),
    }
    num_cells = _count_cells(result.get("iscell"))
    if num_cells is not None:
        summary["n_rois_classified_as_cells"] = num_cells
    if len(dims) == 2:
        summary["movie_height_px"] = int(dims[0])
        summary["movie_width_px"] = int(dims[1])
    return summary
