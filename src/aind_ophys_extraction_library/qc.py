"""quality_control.json generation for ophys extraction.

Capsule-specific metric content over the shared ``aind_pophys_metadata``
QC builders. The ``QualityControl`` wrapper with the project-standard
``default_grouping`` and the writer both live in the shared library. This
module owns the extraction-specific ``CheckboxMetric`` for detected ROIs.
``job.py`` collects the metrics and emits ONE full ``quality_control.json``
per plane (never per-metric files).
"""

from datetime import datetime as dt
from pathlib import Path
from typing import List

import pytz
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc
from aind_qcportal_schema.metric_value import CheckboxMetric

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")


def build_detected_rois_metric(
    unique_id: str,
    num_rois: int,
    reference: str,
) -> QCMetric:
    """Construct the detected-ROIs CheckboxMetric.

    The metric status history is seeded as PASS by "Automated" (extraction ran
    successfully). Reviewers mark individual checkboxes (``"Missing ROIs"``,
    ``"ROI {i} invalid"``) to flag failures; the overall status then becomes FAIL.

    Parameters
    ----------
    unique_id : str
        Plane/FOV identifier used in the metric name.
    num_rois : int
        Number of ROIs detected; one checkbox option is created per ROI.
    reference : str
        Path to the ROI summary image with IDs, relative to the asset root.

    Returns
    -------
    QCMetric
        The detected-ROIs checkbox metric.
    """
    options = ["Missing ROIs"] + [f"ROI {i} invalid" for i in range(num_rois)]
    statuses = [Status.FAIL] * len(options)

    return QCMetric(
        name=f"{unique_id} Detected ROIs",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": "Detected ROIs", "type": "Operational QC"},
        description="",
        reference=reference,
        status_history=[
            QCStatus(
                evaluator="Automated",
                status=Status.PASS,
                timestamp=dt.now(_SEATTLE_TZ).isoformat(),
            )
        ],
        value=CheckboxMetric(value=[], options=options, status=statuses),
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
