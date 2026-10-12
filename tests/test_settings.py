"""Tests for ExtractionSettings validators."""

import sys
import unittest
from typing import Any
from unittest.mock import patch

from aind_ophys_extraction_library.settings import ExtractionSettings


def _make_settings(**kwargs: Any) -> ExtractionSettings:
    """Construct ExtractionSettings with a clean ``sys.argv``.

    ``cli_parse_args`` would otherwise consume the test runner's arguments.
    """
    with patch.object(sys, "argv", ["extraction.py"]):
        return ExtractionSettings(**kwargs)


class TestExtractionSettings(unittest.TestCase):
    """Field validators and consistency checks."""

    def test_init_lowercased(self) -> None:
        """init field is lowercased on construction."""
        s = _make_settings(init="MEAN")
        self.assertEqual(s.init, "mean")

    def test_neuropil_lowercased(self) -> None:
        """neuropil field is lowercased on construction."""
        s = _make_settings(neuropil="MutualInfo")
        self.assertEqual(s.neuropil, "mutualinfo")

    def test_rf_zero_maps_to_none(self) -> None:
        """rf=0 is coerced to None (full FOV mode)."""
        s = _make_settings(rf=0)
        self.assertIsNone(s.rf)

    def test_rf_nonzero_preserved(self) -> None:
        """rf > 0 is preserved unchanged."""
        s = _make_settings(rf=32)
        self.assertEqual(s.rf, 32)

    def test_backward_compat_init_numeric(self) -> None:
        """Numeric init strings ('1'–'4') map to named modes."""
        mapping = {
            "1": "max/mean",
            "2": "mean",
            "3": "enhanced_mean",
            "4": "max",
        }
        for num, name in mapping.items():
            with self.subTest(num=num):
                s = _make_settings(init=num)
                self.assertEqual(s.init, name)

    def test_consistency_cnmf_corr_pnr_sets_ssub(self) -> None:
        """neuropil=cnmf + init=corr_pnr forces ssub=1."""
        s = _make_settings(neuropil="cnmf", init="corr_pnr", ssub=4)
        self.assertEqual(s.ssub, 1)

    def test_consistency_cnmfe_greedy_roi_raises(self) -> None:
        """neuropil=cnmf-e + init=greedy_roi raises ValueError."""
        with self.assertRaises((ValueError, SystemExit)):
            _make_settings(neuropil="cnmf-e", init="greedy_roi")

    def test_consistency_suite2p_neuropil_with_caiman_init_raises(self) -> None:
        """Suite2p neuropil with CaImAn init raises ValueError."""
        with self.assertRaises((ValueError, SystemExit)):
            _make_settings(neuropil="mutualinfo", init="greedy_roi")

    def test_defaults(self) -> None:
        """Default field values match expected pipeline defaults."""
        s = _make_settings()
        self.assertEqual(s.diameter, 0)
        self.assertEqual(s.init, "mean")
        self.assertEqual(s.neuropil, "mutualinfo")
        self.assertFalse(s.contour_video)
        self.assertFalse(s.verbose)


if __name__ == "__main__":
    unittest.main()
