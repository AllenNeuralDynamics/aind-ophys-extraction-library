"""Tests for the quality_control.json builder."""

import json
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
        """evaluation tag is 'Extracted ROIs'."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertEqual(m.tags["evaluation"], "Extracted ROIs")

    def test_metric_type_tag(self) -> None:
        """type tag is 'Informational'."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertEqual(m.tags["type"], "Informational")

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

    def test_value_maps_each_roi_to_reference(self) -> None:
        """value is one mapping of ROI id to its figure reference."""
        m = qc.build_detected_rois_metric("plane_0", 5, "p/ref.png")
        self.assertEqual(
            m.value,
            [{str(i): {"reference": "p/ref.png"} for i in range(5)}],
        )
        self.assertEqual(list(m.value[0]), ["0", "1", "2", "3", "4"])

    def test_value_is_one_plain_mapping(self) -> None:
        """value is a one-element list of a dict, not a JSON string."""
        m = qc.build_detected_rois_metric("plane_0", 3, "ref.png")
        self.assertEqual(len(m.value), 1)
        self.assertIsInstance(m.value[-1], dict)
        for record in m.value[-1].values():
            self.assertEqual(set(record), {"reference"})

    def test_zero_rois(self) -> None:
        """Zero ROIs produces a curation metric with an empty mapping."""
        m = qc.build_detected_rois_metric("plane_0", 0, "ref.png")
        self.assertEqual(m.value, [{}])


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

    def test_written_document_keeps_curation_shape(self) -> None:
        """The written JSON holds the mapping value and the group tags."""
        d = Path(tempfile.mkdtemp())
        metrics = [qc.build_detected_rois_metric("plane_0", 3, "roi.png")]
        out = qc.write_quality_control_json(qc.build_quality_control(metrics), d)
        metric = json.loads(out.read_text())["metrics"][0]
        self.assertEqual(metric["object_type"], "Curation metric")
        self.assertEqual(
            metric["tags"],
            {"evaluation": "Extracted ROIs", "type": "Informational"},
        )
        self.assertEqual(
            metric["value"],
            [{str(i): {"reference": "roi.png"} for i in range(3)}],
        )


if __name__ == "__main__":
    unittest.main()
