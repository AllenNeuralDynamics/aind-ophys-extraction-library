"""Acquisition metadata reads for ophys extraction.

A thin, extraction-specific layer over the shared ``aind_pophys_metadata``
library. Every schema-version difference is resolved by
:class:`~aind_pophys_metadata.core.CoreMetadata`, which reads one located
document regardless of the schema it is written in; this module assembles the
extraction ``ExtractionMetadata`` view, the extraction ``Code`` block, and the
output summary written to ``DataProcess.output_parameters``.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata import (
    MODEL_SOURCE_ASSET,
    MODEL_SOURCE_NETWORK,
    RAW_FORMAT_TIFF_STACK,
    CoreMetadata,
    model_provenance,
)

from ..settings import ANATOMICAL_INIT_MODES, CAIMAN_INIT_MODES

logger = logging.getLogger("Ophys extraction")

PROCESS_TYPE = ProcessName.VIDEO_ROI_TIMESERIES_EXTRACTION

#: Environment variable cellpose itself reads to locate its model files
#: (``cellpose.models.MODEL_DIR``). A cellpose convention, not a Code Ocean
#: one: the runtime wrapper decides which directory it points at, and this
#: library only reports whatever the loader used.
CELLPOSE_MODELS_PATH_ENV = "CELLPOSE_LOCAL_MODELS_PATH"

#: Directory cellpose falls back to when the variable is unset.
CELLPOSE_DEFAULT_MODELS_DIR = Path.home() / ".cellpose" / "models"

#: ``Code.parameters`` key holding the cellpose model provenance records.
CELLPOSE_MODELS_PARAMETER = "cellpose_models"

#: ``init`` modes that estimate the cell diameter from a bare ``Cellpose()``
#: when ``diameter == 0`` and otherwise never load a model: Suite2p's
#: ``sourcery`` plus the CaImAn modes.
AUTO_DIAMETER_INIT_MODES = ("sourcery",) + CAIMAN_INIT_MODES

#: Model a bare ``Cellpose()`` resolves in cellpose 3.x.
AUTO_DIAMETER_MODEL = "cyto3"

#: Models whose files cellpose names with a ``torch`` infix; every other name
#: is used verbatim. Mirrors ``cellpose.models.model_path`` /
#: ``size_model_path``.
_TORCH_NAMED_MODELS = ("cyto", "cyto2", "nuclei")


def is_tiff_stack(raw_format: Optional[str]) -> bool:
    """Return whether the movie is a stitched single-plane TIFF stack.

    A TIFF-stack movie carries ``epoch_locations.json``, so segmentation can
    leave out its photostimulation epochs.

    Parameters
    ----------
    raw_format : str or None
        Raw movie layout from the core acquisition file.

    Returns
    -------
    bool
        ``True`` for ``tiff_stack``.
    """
    return raw_format == RAW_FORMAT_TIFF_STACK


@dataclass(frozen=True)
class ExtractionMetadata:
    """Minimal metadata the extraction step consumes.

    ``frame_rate`` drives Suite2p temporal binning and is validated at load
    time, so a rate-less asset fails with a message naming the file rather
    than partway through ``run()``. ``raw_format`` selects the TIFF-stack
    path; ``valid_epoch_stems`` carries its TIFF stems that survive
    photostimulation filtering. The remaining fields are identifiers
    used for logging and output naming.
    """

    frame_rate: float
    raw_format: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None
    valid_epoch_stems: Tuple[str, ...] = ()


def valid_epoch_stems(core: CoreMetadata) -> Tuple[str, ...]:
    """TIFF stems of the stimulus epochs TIFF-stack extraction keeps.

    Photostimulation epochs are excluded: their frames are not part of the
    movie segmentation runs on. The filter reads each epoch's ``photostim``
    annotation, which ``aind_pophys_metadata`` derives from the stimulus name
    for v1 and v2 documents.

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
        if epoch["photostim"]:
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
    raw_format = core.get_raw_format()
    return ExtractionMetadata(
        frame_rate=core.get_frame_rate(
            cli_override=frame_rate_override,
            required=True,
        ),
        raw_format=raw_format,
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
        valid_epoch_stems=(
            valid_epoch_stems(core) if is_tiff_stack(raw_format) else ()
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


def cellpose_model_files(model_type: str) -> Tuple[str, str]:
    """Return the two file basenames cellpose loads for ``model_type``.

    Both are needed: ``Cellpose.__init__`` builds the diameter-estimating
    ``SizeModel`` unconditionally, before ``diameter`` is considered, so the
    size model is resolved even on a run that never estimates a diameter.

    Parameters
    ----------
    model_type : str
        Built-in cellpose model name, for example ``"cyto"`` or ``"cyto3"``.

    Returns
    -------
    tuple of str
        The segmentation net basename and the size-model basename.
    """
    if model_type in _TORCH_NAMED_MODELS:
        return (f"{model_type}torch_0", f"size_{model_type}torch_0.npy")
    return (model_type, f"size_{model_type}.npy")


def cellpose_models_dir() -> Path:
    """Return the directory cellpose resolves model files in.

    Read from the environment rather than assumed, so the deployment - not
    this library - owns where the bytes live.

    Returns
    -------
    Path
        :data:`CELLPOSE_MODELS_PATH_ENV` when set, else cellpose's own
        per-user cache directory.
    """
    value = os.environ.get(CELLPOSE_MODELS_PATH_ENV)
    return Path(value) if value else CELLPOSE_DEFAULT_MODELS_DIR


@dataclass(frozen=True)
class CellposeModelPlan:
    """The cellpose files one run loads, and which were already local.

    ``present_before`` is captured before detection runs, which is the only
    moment the distinction is observable: a file absent then and present
    afterwards was downloaded by the loader. Policy is local-first with a
    network fallback, so both outcomes are legitimate and the record only
    states which one happened.
    """

    model_type: str
    directory: Path
    filenames: Tuple[str, ...]
    present_before: Tuple[str, ...]

    @property
    def source_name(self) -> str:
        """Stable logical label for the source, never a filesystem path."""
        return self.directory.name or self.model_type


def plan_cellpose_models(
    init: str, diameter: int, pretrained_model: str
) -> Optional[CellposeModelPlan]:
    """Plan the cellpose provenance record for one extraction run.

    Extraction reaches cellpose two ways, and they are mutually exclusive.
    The anatomical ``init`` modes segment a summary image with Suite2p's
    ``pretrained_model``; the functional modes load nothing unless
    ``diameter == 0``, in which case a bare ``Cellpose()`` estimates it from
    cellpose 3.x's default model. Every other combination - ``sparsery``, or
    a functional mode with an operator-supplied diameter - never loads a
    model, and yields ``None`` rather than a speculative record.

    Parameters
    ----------
    init : str
        Resolved ``ExtractionSettings.init``.
    diameter : int
        Resolved ``ExtractionSettings.diameter``; ``0`` requests estimation.
    pretrained_model : str
        Resolved ``ExtractionSettings.pretrained_model``.

    Returns
    -------
    CellposeModelPlan or None
        The files to describe, or ``None`` when this run loads no model or
        loads one whose files cannot be named.
    """
    if init in ANATOMICAL_INIT_MODES:
        if os.sep in pretrained_model or Path(pretrained_model).exists():
            logger.warning(
                "pretrained_model %r is a custom model file, whose provenance "
                "this library cannot name; omitting cellpose provenance.",
                pretrained_model,
            )
            return None
        model_type = pretrained_model
    elif init in AUTO_DIAMETER_INIT_MODES and diameter == 0:
        model_type = AUTO_DIAMETER_MODEL
    else:
        return None

    directory = cellpose_models_dir()
    filenames = cellpose_model_files(model_type)
    return CellposeModelPlan(
        model_type=model_type,
        directory=directory,
        filenames=filenames,
        present_before=tuple(
            name for name in filenames if (directory / name).is_file()
        ),
    )


def build_cellpose_provenance(
    plan: CellposeModelPlan,
) -> List[Dict[str, str]]:
    """Describe the cellpose model bytes a run loaded.

    Provenance only: nothing is compared against an expected digest and no
    mismatch can fail this call. A file still absent after the run is omitted
    by the shared helper rather than asserted.

    Parameters
    ----------
    plan : CellposeModelPlan
        Plan built by :func:`plan_cellpose_models` before detection ran.

    Returns
    -------
    list of dict
        One record per file found, each with ``filename``, ``sha256``,
        ``source_name`` and ``source``.
    """
    records: List[Dict[str, str]] = []
    for source, names in (
        (MODEL_SOURCE_ASSET, plan.present_before),
        (
            MODEL_SOURCE_NETWORK,
            tuple(n for n in plan.filenames if n not in plan.present_before),
        ),
    ):
        records.extend(
            model_provenance(
                plan.directory,
                names,
                source_name=plan.source_name,
                source=source,
            )
        )
    return records


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
        The populated code block; name, version and url are this library's
        installed package metadata.
    """
    return shared_processing.build_code(
        __name__,
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
