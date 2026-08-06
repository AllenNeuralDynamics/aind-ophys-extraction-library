"""Tests for the v1/v2 metadata read layer and the extraction Code block.

The v1/v2 field getters, path helpers and ``Processing`` builders live in the
shared ``aind_pophys_metadata`` library (tested there); these tests cover the
extraction surface: ``load_extraction_metadata`` dispatch, the dual-version
Bergamo epoch-stem resolution, the ``Code`` identity, and the
``output_parameters`` summary.

The two v2 fixtures under ``tests/fixtures`` are real ``acquisition.json``
documents produced by running ``aind-metadata-upgrader`` over real v1
``session.json`` records - a multiplane MESO session (no stimulus epochs) and
a Bergamo photostimulation session (five). They are generated, never
hand-written, so the v2 branches are exercised against the published
migration rules rather than a guess at them.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import Processing
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata.runner import stage_guard

from aind_ophys_extraction_library.utils import metadata_utils as mu

_FIXTURES = Path(__file__).parent / "fixtures"

_PIPELINE_ENV = {
    shared_processing.PIPELINE_NAME_ENV: "aind-pophys-pipeline",
    shared_processing.PIPELINE_VERSION_ENV: "2.0.0",
    shared_processing.PIPELINE_URL_ENV: "https://example.com/pipeline",
}

#: Stems the Bergamo fixture must yield: five epochs, the photostimulation
#: one excluded.
_BERGAMO_STEMS = ("spont2", "spont", "neuron3", "neuron3_2")


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write a JSON payload into directory/name."""
    (directory / name).write_text(json.dumps(payload))


def _fixture(name: str) -> dict:
    """Load a checked-in v2 acquisition fixture."""
    return json.loads((_FIXTURES / f"{name}.json").read_text())


def _copy_fixture(directory: Path, name: str) -> None:
    """Copy a fixture into ``directory`` as ``acquisition.json``."""
    _write(directory, "acquisition.json", _fixture(name))


def _imaging_config_type() -> str:
    """Return the v2 ImagingConfig discriminator string."""
    from aind_data_schema.components.configs import ImagingConfig
    from aind_pophys_metadata import io

    return io.object_type_value(ImagingConfig)


class TestLoadMetadata(unittest.TestCase):
    """File-presence dispatch and independent id reads."""

    def test_v2_dispatch(self):
        """acquisition.json present -> v2 path with instrument_id."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "acquisition.json",
            {
                "instrument_id": "MESO.1",
                "data_streams": [
                    {
                        "configurations": [
                            {
                                "object_type": _imaging_config_type(),
                                "sampling_strategy": {"frame_rate": 9.6},
                            }
                        ]
                    }
                ],
            },
        )
        _write(d, "data_description.json", {"name": "mouse_1_2024"})
        _write(d, "subject.json", {"subject_id": "781"})
        md = mu.load_extraction_metadata(d)
        self.assertEqual(md.frame_rate, 9.6)
        self.assertEqual(md.instrument_id, "MESO.1")
        self.assertEqual(md.subject_id, "781")
        self.assertEqual(md.dataset_name, "mouse_1_2024")
        self.assertEqual(md.valid_epoch_stems, ())

    def test_v1_dispatch(self):
        """session.json present -> v1 path with rig_id as instrument."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {
                "rig_id": "MESO.1",
                "data_streams": [{"ophys_fovs": [{"frame_rate": 11.0}]}],
            },
        )
        md = mu.load_extraction_metadata(d)
        self.assertEqual(md.frame_rate, 11.0)
        self.assertEqual(md.instrument_id, "MESO.1")
        self.assertIsNone(md.subject_id)

    def test_subject_id_canonical_from_core(self):
        """subject_id is read from the core file when present."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {
                "rig_id": "MESO.2",
                "subject_id": "555",
                "data_streams": [{"ophys_fovs": [{"frame_rate": 5.0}]}],
            },
        )
        _write(d, "subject.json", {"subject_id": "999"})
        md = mu.load_extraction_metadata(d)
        self.assertEqual(md.subject_id, "555")

    def test_platform_fallback_used(self):
        """frame_rate falls back to platform.json when core lacks it."""
        d = Path(tempfile.mkdtemp())
        _write(
            d,
            "session.json",
            {"rig_id": "MESO.2", "data_streams": [{"ophys_fovs": [{}]}]},
        )
        _write(
            d,
            "platform.json",
            {"imaging_plane_groups": [{"acquisition_framerate_Hz": 30.9}]},
        )
        md = mu.load_extraction_metadata(d)
        self.assertEqual(md.frame_rate, 30.9)

    def test_cli_override_used_when_metadata_has_none(self):
        """The operator-supplied rate is the last resort, not the first."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        md = mu.load_extraction_metadata(d, frame_rate_override=7.5)
        self.assertEqual(md.frame_rate, 7.5)

    def test_missing_frame_rate_is_fatal_at_load(self):
        """A rate-less asset fails in the loader, naming the core file."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        with self.assertRaises(ValueError) as ctx:
            mu.load_extraction_metadata(d)
        self.assertIn("session.json", str(ctx.exception))

    def test_raises_without_core_file(self):
        """Neither core file present -> FileNotFoundError."""
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            mu.load_extraction_metadata(d)

    def test_real_v2_multiplane_acquisition_loads(self):
        """A real upgraded multiplane acquisition reaches the compute path."""
        d = Path(tempfile.mkdtemp())
        _copy_fixture(d, "acquisition_multiplane_v2")
        md = mu.load_extraction_metadata(d, frame_rate_override=9.48)
        self.assertEqual(md.instrument_id, "MESO.2")
        self.assertFalse(mu.is_bergamo(md.instrument_id))
        self.assertEqual(md.valid_epoch_stems, ())

    def test_real_v2_bergamo_acquisition_yields_stems(self):
        """A real upgraded Bergamo acquisition resolves its epoch stems."""
        d = Path(tempfile.mkdtemp())
        _copy_fixture(d, "acquisition_bergamo_v2")
        md = mu.load_extraction_metadata(d, frame_rate_override=10.7)
        self.assertTrue(mu.is_bergamo(md.instrument_id))
        self.assertEqual(md.valid_epoch_stems, _BERGAMO_STEMS)


class TestIsBergamo(unittest.TestCase):
    """Instrument routing by substring, never by dataset name."""

    def test_bergamo_detected(self):
        """A Bergamo rig id matches."""
        self.assertTrue(mu.is_bergamo("442_Bergamo_2p_photostim"))

    def test_other_instrument_not_bergamo(self):
        """A mesoscope rig id does not."""
        self.assertFalse(mu.is_bergamo("MESO.2"))

    def test_missing_instrument_not_bergamo(self):
        """An absent instrument id does not."""
        self.assertFalse(mu.is_bergamo(None))


class TestValidEpochStems(unittest.TestCase):
    """Dual-version TIFF-stem resolution for the Bergamo compute path."""

    def test_v1_stems_exclude_photostim(self):
        """v1 reads output_parameters.tiff_stem in epoch order."""
        session = {
            "stimulus_epochs": [
                {
                    "stimulus_name": "spontaneous activity",
                    "output_parameters": {"tiff_stem": "spont"},
                },
                {
                    "stimulus_name": mu.PHOTOSTIM_STIMULUS_NAME,
                    "output_parameters": {"tiff_stem": "spont_slm"},
                },
                {
                    "stimulus_name": "single neuron BCI conditioning",
                    "output_parameters": {"tiff_stem": "neuron3"},
                },
            ]
        }
        self.assertEqual(
            mu.get_valid_epoch_stems("v1", session, Path("session.json")),
            ("spont", "neuron3"),
        )

    def test_v1_missing_stem_raises(self):
        """A v1 epoch with no tiff_stem names itself and the file."""
        session = {
            "stimulus_epochs": [{"stimulus_name": "spontaneous activity"}]
        }
        with self.assertRaises(ValueError) as ctx:
            mu.get_valid_epoch_stems("v1", session, Path("session.json"))
        self.assertIn("session.json", str(ctx.exception))
        self.assertIn("tiff_stem", str(ctx.exception))

    def test_v2_stems_from_real_fixture(self):
        """v2 recovers the stems the upgrade moved onto the data streams."""
        acquisition = _fixture("acquisition_bergamo_v2")
        self.assertEqual(
            mu.get_valid_epoch_stems(
                "v2", acquisition, Path("acquisition.json")
            ),
            _BERGAMO_STEMS,
        )

    def test_v2_empty_stimulus_epochs_is_not_an_error(self):
        """A multiplane v2 acquisition has no epochs and no stems."""
        acquisition = _fixture("acquisition_multiplane_v2")
        self.assertEqual(
            mu.get_valid_epoch_stems(
                "v2", acquisition, Path("acquisition.json")
            ),
            (),
        )

    def test_v2_unmatched_epoch_raises(self):
        """An epoch with no correspondingly-timed stream note fails loudly."""
        acquisition = {
            "data_streams": [
                {"stream_start_time": "t0", "notes": "tiff_stem:spont"},
                {"stream_start_time": "t2", "notes": None},
            ],
            "stimulus_epochs": [
                {"stimulus_name": "spontaneous", "stimulus_start_time": "t1"}
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            mu.get_valid_epoch_stems(
                "v2", acquisition, Path("acquisition.json")
            )
        self.assertIn("acquisition.json", str(ctx.exception))

    def test_v2_photostim_epoch_needs_no_stem(self):
        """A photostimulation epoch is dropped before any stem lookup."""
        acquisition = {
            "data_streams": [],
            "stimulus_epochs": [
                {
                    "stimulus_name": mu.PHOTOSTIM_STIMULUS_NAME,
                    "stimulus_start_time": "t0",
                }
            ],
        }
        self.assertEqual(
            mu.get_valid_epoch_stems(
                "v2", acquisition, Path("acquisition.json")
            ),
            (),
        )


class TestCodeIdentity(unittest.TestCase):
    """Code.version/url must identify one real, resolvable artifact."""

    @patch.dict(os.environ, {}, clear=True)
    def test_version_comes_from_the_installed_library(self):
        """Code.version is the backing library's own version, never empty."""
        code = mu.build_extraction_code({"a": 1})
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))
        self.assertIn(mu.LIBRARY_NAME, code.url)
        self.assertNotIn("/tree/", code.url)

    @patch.dict(os.environ, {"VERSION": "1.4.0"}, clear=True)
    def test_wrapper_version_is_ignored(self):
        """The backing library is authoritative, not the capsule wrapper."""
        code = mu.build_extraction_code({"a": 1})
        self.assertNotEqual(code.version, "1.4.0")
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))

    def test_input_data_optional(self):
        """input_data=None yields a Code without input assets."""
        self.assertIsNone(mu.build_extraction_code({}).input_data)

    def test_input_data_recorded(self):
        """A provided input path is recorded as a data asset."""
        code = mu.build_extraction_code({}, input_data=["plane_0/mc.h5"])
        self.assertEqual(code.input_data[0].name, "plane_0/mc.h5")


class TestProcessName(unittest.TestCase):
    """The emitted DataProcess name, needed before the process exists."""

    def test_name_is_plane_qualified(self):
        """The name matches what the shared builder composes."""
        self.assertEqual(
            mu.process_name("VISp_0"),
            f"VISp_0: {mu.PROCESS_TYPE.value}",
        )


class TestOutputParameters(unittest.TestCase):
    """Scalar summary statistics describing this step's outputs."""

    def _result(self, **overrides) -> dict:
        """Return a minimal detection result, overridable per test."""
        result = {
            "traces_corrected": [[0.0], [1.0], [2.0]],
            "iscell": [[1, 0.9], [0, 0.1], [1, 0.8]],
            "nframes": 100,
            "dims": (512, 480),
        }
        result.update(overrides)
        return result

    def test_summary_fields(self):
        """Counts, frame rate and movie shape are all recorded."""
        summary = mu.build_output_parameters(self._result(), 3, 10.5)
        self.assertEqual(summary["n_rois"], 3)
        self.assertEqual(summary["n_traces"], 3)
        self.assertEqual(summary["n_frames"], 100)
        self.assertEqual(summary["frame_rate_hz"], 10.5)
        self.assertEqual(summary["n_rois_classified_as_cells"], 2)
        self.assertEqual(summary["movie_height_px"], 512)
        self.assertEqual(summary["movie_width_px"], 480)

    def test_scalar_iscell_supported(self):
        """A flat classification sequence counts the same way."""
        summary = mu.build_output_parameters(
            self._result(iscell=[1, 0, 1, 1]), 4, 10.5
        )
        self.assertEqual(summary["n_rois_classified_as_cells"], 3)

    def test_unrecognised_iscell_omits_the_key(self):
        """An unreadable classification is omitted, never guessed."""
        summary = mu.build_output_parameters(
            self._result(iscell=3), 1, 10.5
        )
        self.assertNotIn("n_rois_classified_as_cells", summary)

    def test_empty_result_yields_zero_counts(self):
        """A run that detected nothing still reports its shape honestly."""
        summary = mu.build_output_parameters(
            {"traces_corrected": [], "iscell": [], "nframes": None, "dims": ()},
            0,
            10.5,
        )
        self.assertEqual(summary["n_rois"], 0)
        self.assertEqual(summary["n_frames"], 0)
        self.assertNotIn("movie_height_px", summary)


class TestStageDocument(unittest.TestCase):
    """The document the shared stage guard writes for this capsule."""

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_processing_json_round_trips_with_dependency_graph(self):
        """One plane's document validates and names its upstreams."""
        out = Path(tempfile.mkdtemp())
        upstream = out / "upstream"
        upstream.mkdir()
        code = mu.build_extraction_code({"init": "mean"})
        with stage_guard(
            out,
            mu.PROCESS_TYPE,
            code,
            plane_id="VISp_0",
            output_path="VISp_0/extraction",
        ) as ctx:
            ctx.upstream_names = ["VISp_0: Video motion correction"]
            ctx.output_parameters = {"n_rois": 12}
        doc = Processing.model_validate_json(
            (out / "processing.json").read_text()
        )
        name = mu.process_name("VISp_0")
        self.assertEqual(doc.data_processes[0].name, name)
        self.assertEqual(
            doc.dependency_graph[name],
            ["VISp_0: Video motion correction"],
        )
        self.assertTrue(doc.data_processes[0].code.version)
        self.assertEqual(
            doc.data_processes[0].output_parameters.model_dump()["n_rois"], 12
        )

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_failed_stage_still_writes_a_valid_document(self):
        """A raising stage leaves a document with the error in notes."""
        out = Path(tempfile.mkdtemp())
        code = mu.build_extraction_code({})
        with self.assertRaises(RuntimeError):
            with stage_guard(out, mu.PROCESS_TYPE, code, plane_id="VISp_0"):
                raise RuntimeError("suite2p exploded")
        doc = Processing.model_validate_json(
            (out / "processing.json").read_text()
        )
        self.assertIn("suite2p exploded", doc.data_processes[0].notes)


class TestUpstreamCollection(unittest.TestCase):
    """Sibling upstream discovery, excluding this capsule's own document."""

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_own_document_excluded(self):
        """An extraction document in the input dir is not its own upstream."""
        d = Path(tempfile.mkdtemp())
        own = d / "VISp_0" / "extraction"
        own.mkdir(parents=True)
        code = mu.build_extraction_code({})
        with stage_guard(own, mu.PROCESS_TYPE, code, plane_id="VISp_0"):
            pass
        names = shared_processing.collect_upstream_process_names(
            d, exclude=mu.process_name("VISp_0")
        )
        self.assertEqual(names, [])


class TestStageTimestamps(unittest.TestCase):
    """Timestamps written by the guard must be timezone-aware."""

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_start_time_is_aware(self):
        """The guard stamps an aware start time before the body runs."""
        out = Path(tempfile.mkdtemp())
        before = datetime.now(timezone.utc)
        with stage_guard(out, mu.PROCESS_TYPE, mu.build_extraction_code({})):
            pass
        doc = Processing.model_validate_json(
            (out / "processing.json").read_text()
        )
        start = doc.data_processes[0].start_date_time
        self.assertIsNotNone(start.tzinfo)
        self.assertGreaterEqual(start, before.replace(microsecond=0))


if __name__ == "__main__":
    unittest.main()
