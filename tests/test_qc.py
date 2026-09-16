"""Tests for the quality_control.json builder."""

import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import QualityControl, Status

from aind_ophys_extraction_library import qc


class TestDetectedROIsMetric(unittest.TestCase):
    """build_detected_rois_metric carries the expected tags and values."""

    def test_metric_name_contains_unique_id(self) -> None:
        """Metric name contains the plane unique_id."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertIn("plane_0", m.name)

    def test_metric_evaluation_tag(self) -> None:
        """evaluation tag is 'Detected ROIs'."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertEqual(m.tags["evaluation"], "Detected ROIs")

    def test_metric_reference_set(self) -> None:
        """reference path is stored on the metric."""
        m = qc.build_detected_rois_metric("plane_0", 2, "some/path.png")
        self.assertEqual(m.reference, "some/path.png")

    def test_initial_status_is_pass(self) -> None:
        """Initial status_history entry is PASS (extraction ran ok)."""
        m = qc.build_detected_rois_metric("plane_0", 2, "ref.png")
        self.assertEqual(m.status_history[0].status, Status.PASS)
        self.assertEqual(m.status_history[0].evaluator, "Automated")

    def test_curation_type(self) -> None:
        """type carries the curation vocabulary label."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertEqual(m.type, qc.CURATION_TYPE)

    def test_value_is_per_roi_ids(self) -> None:
        """value is one id per detected ROI, in ROI order."""
        m = qc.build_detected_rois_metric("plane_0", 5, "ref.png")
        self.assertEqual(m.value, ["0", "1", "2", "3", "4"])

    def test_zero_rois(self) -> None:
        """Zero ROIs produces a curation metric with an empty value list."""
        m = qc.build_detected_rois_metric("plane_0", 0, "ref.png")
        self.assertEqual(m.value, [])


class TestQualityControlCollector(unittest.TestCase):
    """build_quality_control + write round-trip."""

    def test_build_and_write(self) -> None:
        """Metrics collect into a QualityControl that re-validates."""
        d = Path(tempfile.mkdtemp())
        metrics = [qc.build_detected_rois_metric("plane_0", 4, "roi.png")]
        qc_obj = qc.build_quality_control(metrics)
        self.assertEqual(qc_obj.default_grouping, ["evaluation"])
        self.assertEqual(len(qc_obj.metrics), 1)
        out = qc.write_quality_control_json(qc_obj, d)
        self.assertEqual(out.name, "quality_control.json")
        QualityControl.model_validate_json(out.read_text())


if __name__ == "__main__":
    unittest.main()
