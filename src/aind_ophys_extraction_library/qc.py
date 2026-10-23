"""quality_control.json generation for ophys extraction.

Extraction-specific metric content over the shared ``aind_pophys_metadata``
QC builders (the ``QualityControl`` wrapper, ``default_grouping`` and writer
live there). ``job.py`` collects the metrics and emits ONE
``quality_control.json`` per plane.

The extracted-ROIs payload is the core
``aind_data_schema.core.quality_control.CurationMetric`` (one reference
entry per ROI).
"""

from datetime import datetime as dt
from pathlib import Path
from typing import List

import pytz
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

EVALUATION = "Extracted ROIs"
METRIC_TYPE = "Informational"
CURATION_TYPE = "Detected ROI validity"
DESCRIPTION = "ROIs detected in this plane, each shown on the plane's summary images."


def build_detected_rois_metric(
    unique_id: str,
    num_rois: int,
    reference: str,
) -> CurationMetric:
    """Construct the extracted-ROIs curation metric.

    Seeded PASS by ``"Automated"`` (extraction ran and produced ROIs). The
    ``value`` is a one-element list holding a mapping from each ROI id (the
    ROI index as a string) to ``{"reference": reference}``, the QC portal
    curation shape. A plane without ROIs gets ``[{}]``.

    Parameters
    ----------
    unique_id : str
        Plane/FOV identifier used in the metric name.
    num_rois : int
        Number of ROIs detected; one mapping entry is created per ROI.
    reference : str
        Path to the ROI summary image with IDs, relative to the asset root;
        the metric and every ROI entry point to it.

    Returns
    -------
    CurationMetric
        The extracted-ROIs curation metric.
    """
    roi_references = {str(roi_id): {"reference": reference} for roi_id in range(num_rois)}
    return CurationMetric(
        name=f"{unique_id} Detected ROIs",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        type=CURATION_TYPE,
        value=[roi_references],
        description=DESCRIPTION,
        reference=reference,
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        status_history=[
            QCStatus(
                evaluator="Automated",
                status=Status.PASS,
                timestamp=dt.now(_SEATTLE_TZ).isoformat(),
            )
        ],
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
