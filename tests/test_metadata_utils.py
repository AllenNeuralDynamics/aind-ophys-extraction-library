"""Tests for the metadata read layer and the extraction Code block.

Schema-version handling, the field getters, path helpers and ``Processing``
builders live in the shared ``aind_pophys_metadata`` library (tested there);
these tests cover the extraction surface: ``load_extraction_metadata``, the
Bergamo epoch-stem filter, the ``Code`` identity, and the
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
from typing import Any
from unittest.mock import patch

import numpy as np
from aind_data_schema.core.processing import Processing
from aind_pophys_metadata import CoreMetadata
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

    def test_v2_dispatch(self) -> None:
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

    def test_v1_dispatch(self) -> None:
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

    def test_subject_id_canonical_from_core(self) -> None:
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

    def test_platform_fallback_used(self) -> None:
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

    def test_cli_override_used_when_metadata_has_none(self) -> None:
        """The operator-supplied rate is the last resort, not the first."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        md = mu.load_extraction_metadata(d, frame_rate_override=7.5)
        self.assertEqual(md.frame_rate, 7.5)

    def test_missing_frame_rate_is_fatal_at_load(self) -> None:
        """A rate-less asset fails in the loader, naming the core file."""
        d = Path(tempfile.mkdtemp())
        _write(d, "session.json", {"rig_id": "MESO.2"})
        with self.assertRaises(ValueError) as ctx:
            mu.load_extraction_metadata(d)
        self.assertIn("session.json", str(ctx.exception))

    def test_raises_without_core_file(self) -> None:
        """Neither core file present -> FileNotFoundError."""
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            mu.load_extraction_metadata(d)

    def test_real_v2_multiplane_acquisition_loads(self) -> None:
        """A real upgraded multiplane acquisition reaches the compute path."""
        d = Path(tempfile.mkdtemp())
        _copy_fixture(d, "acquisition_multiplane_v2")
        md = mu.load_extraction_metadata(d, frame_rate_override=9.48)
        self.assertEqual(md.instrument_id, "MESO.2")
        self.assertFalse(mu.is_bergamo(md.instrument_id))
        self.assertEqual(md.valid_epoch_stems, ())

    def test_real_v2_bergamo_acquisition_yields_stems(self) -> None:
        """A real upgraded Bergamo acquisition resolves its epoch stems."""
        d = Path(tempfile.mkdtemp())
        _copy_fixture(d, "acquisition_bergamo_v2")
        md = mu.load_extraction_metadata(d, frame_rate_override=10.7)
        self.assertTrue(mu.is_bergamo(md.instrument_id))
        self.assertEqual(md.valid_epoch_stems, _BERGAMO_STEMS)


class TestIsBergamo(unittest.TestCase):
    """Instrument routing by substring, never by dataset name."""

    def test_bergamo_detected(self) -> None:
        """A Bergamo rig id matches."""
        self.assertTrue(mu.is_bergamo("442_Bergamo_2p_photostim"))

    def test_other_instrument_not_bergamo(self) -> None:
        """A mesoscope rig id does not."""
        self.assertFalse(mu.is_bergamo("MESO.2"))

    def test_missing_instrument_not_bergamo(self) -> None:
        """An absent instrument id does not."""
        self.assertFalse(mu.is_bergamo(None))


class TestValidEpochStems(unittest.TestCase):
    """TIFF-stem resolution for the Bergamo compute path.

    The filter is version-blind: the shared reader resolves a stem from
    either the data-stream note or ``output_parameters.tiff_stem``, and both
    routes are present in real assets of both schema versions.
    """

    def _core(self, core_name: str, payload: dict) -> CoreMetadata:
        """Write a core document and return its located metadata."""
        d = Path(tempfile.mkdtemp())
        _write(d, core_name, payload)
        return CoreMetadata.load(d)

    def test_stems_from_output_parameters(self) -> None:
        """A document carrying only output_parameters.tiff_stem resolves."""
        core = self._core(
            "session.json",
            {
                "stimulus_epochs": [
                    {
                        "stimulus_name": "spontaneous activity",
                        "output_parameters": {"tiff_stem": "spont"},
                    },
                    {
                        "stimulus_name": "single neuron BCI conditioning",
                        "output_parameters": {"tiff_stem": "neuron3"},
                    },
                ]
            },
        )
        self.assertEqual(
            mu.valid_epoch_stems(core), ("spont", "neuron3")
        )

    def test_stems_from_data_stream_notes(self) -> None:
        """A document carrying only the stream note resolves identically."""
        core = self._core(
            "acquisition.json",
            {
                "data_streams": [
                    {"stream_start_time": "t0", "notes": "tiff_stem:spont"},
                    {"stream_start_time": "t1", "notes": "tiff_stem:neuron3"},
                ],
                "stimulus_epochs": [
                    {
                        "stimulus_name": "spontaneous activity",
                        "stimulus_start_time": "t0",
                    },
                    {
                        "stimulus_name": "single neuron BCI conditioning",
                        "stimulus_start_time": "t1",
                    },
                ],
            },
        )
        self.assertEqual(
            mu.valid_epoch_stems(core), ("spont", "neuron3")
        )

    def test_photostim_epoch_excluded_and_needs_no_stem(self) -> None:
        """Photostimulation is dropped, stem or not, in recorded order."""
        core = self._core(
            "acquisition.json",
            {
                "data_streams": [],
                "stimulus_epochs": [
                    {
                        "stimulus_name": "spontaneous activity",
                        "output_parameters": {"tiff_stem": "spont"},
                    },
                    {
                        "stimulus_name": mu.PHOTOSTIM_STIMULUS_NAME,
                        "stimulus_modalities": ["Optogenetics"],
                    },
                    {
                        "stimulus_name": "single neuron BCI conditioning",
                        "output_parameters": {"tiff_stem": "neuron3"},
                    },
                ],
            },
        )
        self.assertEqual(
            mu.valid_epoch_stems(core), ("spont", "neuron3")
        )

    def test_order_is_the_recorded_order_not_sorted(self) -> None:
        """Stems keep document order; each pairs positionally with frames."""
        core = self._core(
            "acquisition.json",
            {
                "stimulus_epochs": [
                    {
                        "stimulus_name": "b",
                        "output_parameters": {"tiff_stem": "spont2"},
                    },
                    {
                        "stimulus_name": "a",
                        "output_parameters": {"tiff_stem": "spont"},
                    },
                ]
            },
        )
        self.assertEqual(mu.valid_epoch_stems(core), ("spont2", "spont"))

    def test_stems_from_real_fixture(self) -> None:
        """The real upgraded Bergamo acquisition yields its five stems."""
        core = self._core(
            "acquisition.json", _fixture("acquisition_bergamo_v2")
        )
        self.assertEqual(mu.valid_epoch_stems(core), _BERGAMO_STEMS)

    def test_empty_stimulus_epochs_is_not_an_error(self) -> None:
        """A multiplane acquisition has no epochs and no stems."""
        core = self._core(
            "acquisition.json", _fixture("acquisition_multiplane_v2")
        )
        self.assertEqual(mu.valid_epoch_stems(core), ())

    def test_missing_stem_raises(self) -> None:
        """A kept epoch with neither stem route names itself and the file."""
        core = self._core(
            "acquisition.json",
            {
                "data_streams": [
                    {"stream_start_time": "t0", "notes": "tiff_stem:spont"},
                    {"stream_start_time": "t2", "notes": None},
                ],
                "stimulus_epochs": [
                    {
                        "stimulus_name": "spontaneous",
                        "stimulus_start_time": "t1",
                    }
                ],
            },
        )
        with self.assertRaises(ValueError) as ctx:
            mu.valid_epoch_stems(core)
        self.assertIn("acquisition.json", str(ctx.exception))
        self.assertIn("spontaneous", str(ctx.exception))


class TestCodeIdentity(unittest.TestCase):
    """Code.version/url must identify one real, resolvable artifact."""

    @patch.dict(os.environ, {}, clear=True)
    def test_version_comes_from_the_installed_library(self) -> None:
        """Code.version is the backing library's own version, never empty."""
        code = mu.build_extraction_code({"a": 1})
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))
        self.assertIn(mu.LIBRARY_NAME, code.url)
        self.assertNotIn("/tree/", code.url)

    @patch.dict(os.environ, {"VERSION": "1.4.0"}, clear=True)
    def test_wrapper_version_is_ignored(self) -> None:
        """The backing library is authoritative, not the runtime wrapper."""
        code = mu.build_extraction_code({"a": 1})
        self.assertNotEqual(code.version, "1.4.0")
        self.assertEqual(code.version, version(mu.LIBRARY_NAME))

    def test_input_data_optional(self) -> None:
        """input_data=None yields a Code without input assets."""
        self.assertIsNone(mu.build_extraction_code({}).input_data)

    def test_input_data_recorded(self) -> None:
        """A provided input path is recorded as a data asset."""
        code = mu.build_extraction_code({}, input_data=["plane_0/mc.h5"])
        self.assertEqual(code.input_data[0].name, "plane_0/mc.h5")


class TestProcessName(unittest.TestCase):
    """The emitted DataProcess name, needed before the process exists."""

    def test_name_is_plane_qualified(self) -> None:
        """The name matches what the shared builder composes."""
        self.assertEqual(
            mu.process_name("VISp_0"),
            f"VISp_0: {mu.PROCESS_TYPE.value}",
        )


class TestOutputParameters(unittest.TestCase):
    """Scalar summary statistics describing this step's outputs."""

    def _result(self, **overrides: Any) -> dict:
        """Return a minimal detection result, overridable per test."""
        result = {
            "traces_corrected": [[0.0], [1.0], [2.0]],
            "iscell": [[1, 0.9], [0, 0.1], [1, 0.8]],
            "nframes": 100,
            "dims": (512, 480),
        }
        result.update(overrides)
        return result

    def test_numpy_result_is_supported(self) -> None:
        """Suite2p hands back numpy arrays, not lists.

        Regression for a production failure: the builder used
        ``result.get(key) or default``, which calls ``__bool__`` on an array
        and raises ``ValueError: The truth value of an array with more than
        one element is ambiguous``. The list-based fixtures below never
        exercised that, so the whole test suite passed while a real run died
        after extraction had already finished.
        """
        summary = mu.build_output_parameters(
            {
                "traces_corrected": np.zeros((279, 46307)),
                "iscell": np.column_stack(
                    (np.ones(279), np.full(279, 0.9))
                ),
                "nframes": 46307,
                "dims": np.array([512, 512]),
            },
            279,
            10.71,
        )
        self.assertEqual(summary["n_rois"], 279)
        self.assertEqual(summary["n_traces"], 279)
        self.assertEqual(summary["n_frames"], 46307)
        self.assertEqual(summary["n_rois_classified_as_cells"], 279)
        self.assertEqual(summary["movie_height_px"], 512)
        self.assertEqual(summary["movie_width_px"], 512)

    def test_empty_numpy_arrays_are_supported(self) -> None:
        """A zero-ROI run still reports honestly rather than raising."""
        summary = mu.build_output_parameters(
            {
                "traces_corrected": np.zeros((0, 0)),
                "iscell": np.zeros((0, 2)),
                "nframes": 0,
                "dims": np.array([512, 512]),
            },
            0,
            10.71,
        )
        self.assertEqual(summary["n_traces"], 0)
        self.assertEqual(summary["n_rois_classified_as_cells"], 0)

    def test_summary_fields(self) -> None:
        """Counts, frame rate and movie shape are all recorded."""
        summary = mu.build_output_parameters(self._result(), 3, 10.5)
        self.assertEqual(summary["n_rois"], 3)
        self.assertEqual(summary["n_traces"], 3)
        self.assertEqual(summary["n_frames"], 100)
        self.assertEqual(summary["frame_rate_hz"], 10.5)
        self.assertEqual(summary["n_rois_classified_as_cells"], 2)
        self.assertEqual(summary["movie_height_px"], 512)
        self.assertEqual(summary["movie_width_px"], 480)

    def test_scalar_iscell_supported(self) -> None:
        """A flat classification sequence counts the same way."""
        summary = mu.build_output_parameters(
            self._result(iscell=[1, 0, 1, 1]), 4, 10.5
        )
        self.assertEqual(summary["n_rois_classified_as_cells"], 3)

    def test_unrecognised_iscell_omits_the_key(self) -> None:
        """An unreadable classification is omitted, never guessed."""
        summary = mu.build_output_parameters(
            self._result(iscell=3), 1, 10.5
        )
        self.assertNotIn("n_rois_classified_as_cells", summary)

    def test_empty_result_yields_zero_counts(self) -> None:
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
    """The document the shared stage guard writes for this step."""

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_processing_json_round_trips_with_dependency_graph(self) -> None:
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
    def test_failed_stage_still_writes_a_valid_document(self) -> None:
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
    """Sibling upstream discovery, excluding this step's own document."""

    @patch.dict(os.environ, _PIPELINE_ENV, clear=True)
    def test_own_document_excluded(self) -> None:
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
    def test_start_time_is_aware(self) -> None:
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


def _write_model_files(directory: Path, *names: str) -> None:
    """Write stand-in cellpose model files, real ``.npy`` where named so."""
    for name in names:
        path = directory / name
        if name.endswith(".npy"):
            np.save(
                path,
                np.array(
                    {
                        "diam_mean": np.float32(30.0),
                        "A": np.zeros((256, 1), dtype=np.float32),
                    },
                    dtype=object,
                ),
                allow_pickle=True,
            )
        else:
            path.write_bytes(np.arange(1024, dtype=np.uint8).tobytes())


class TestCellposeModelFiles(unittest.TestCase):
    """Basenames must mirror cellpose's own model-path construction."""

    def test_torch_named_family(self) -> None:
        """cyto/cyto2/nuclei carry the torch infix and an index."""
        self.assertEqual(
            mu.cellpose_model_files("cyto"),
            ("cytotorch_0", "size_cytotorch_0.npy"),
        )
        self.assertEqual(
            mu.cellpose_model_files("nuclei"),
            ("nucleitorch_0", "size_nucleitorch_0.npy"),
        )

    def test_verbatim_family(self) -> None:
        """cyto3 and friends are used verbatim."""
        self.assertEqual(
            mu.cellpose_model_files("cyto3"), ("cyto3", "size_cyto3.npy")
        )


class TestCellposeModelsDir(unittest.TestCase):
    """The model directory comes from the environment, never a literal."""

    @patch.dict(os.environ, {}, clear=True)
    def test_falls_back_to_cellpose_cache(self) -> None:
        """With the variable unset, cellpose's own cache directory is used."""
        self.assertEqual(
            mu.cellpose_models_dir(), mu.CELLPOSE_DEFAULT_MODELS_DIR
        )

    def test_environment_wins(self) -> None:
        """The deployment's directory is honoured verbatim."""
        with patch.dict(
            os.environ,
            {mu.CELLPOSE_MODELS_PATH_ENV: "/opt/models/cellpose"},
            clear=True,
        ):
            self.assertEqual(
                mu.cellpose_models_dir(), Path("/opt/models/cellpose")
            )


class TestCellposeModelPlan(unittest.TestCase):
    """Which model family a given settings combination actually loads."""

    def setUp(self) -> None:
        """Point cellpose's model directory at an empty temp directory."""
        self.models = Path(tempfile.mkdtemp())
        patcher = patch.dict(
            os.environ,
            {mu.CELLPOSE_MODELS_PATH_ENV: str(self.models)},
            clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_anatomical_modes_use_pretrained_model(self) -> None:
        """Every anatomical init mode resolves the cyto family by default."""
        for init in mu.ANATOMICAL_INIT_MODES:
            with self.subTest(init=init):
                plan = mu.plan_cellpose_models(init, 0, "cyto")
                self.assertEqual(plan.model_type, "cyto")
                self.assertEqual(
                    plan.filenames,
                    ("cytotorch_0", "size_cytotorch_0.npy"),
                )

    def test_functional_modes_with_auto_diameter_use_cyto3(self) -> None:
        """A zero diameter is estimated with cellpose 3.x's default model."""
        for init in mu.AUTO_DIAMETER_INIT_MODES:
            with self.subTest(init=init):
                plan = mu.plan_cellpose_models(init, 0, "cyto")
                self.assertEqual(plan.model_type, mu.AUTO_DIAMETER_MODEL)
                self.assertEqual(
                    plan.filenames, ("cyto3", "size_cyto3.npy")
                )

    def test_functional_modes_with_given_diameter_load_nothing(self) -> None:
        """An operator-supplied diameter means cellpose is never called."""
        for init in mu.AUTO_DIAMETER_INIT_MODES:
            with self.subTest(init=init):
                self.assertIsNone(mu.plan_cellpose_models(init, 12, "cyto"))

    def test_sparsery_loads_nothing(self) -> None:
        """sparsery is purely functional at any diameter."""
        self.assertIsNone(mu.plan_cellpose_models("sparsery", 0, "cyto"))

    def test_custom_model_path_is_omitted(self) -> None:
        """A custom model file is not named rather than mislabelled."""
        custom = self.models / "my_model"
        custom.write_bytes(b"weights")
        self.assertIsNone(
            mu.plan_cellpose_models("mean", 0, str(custom))
        )

    def test_source_name_is_the_directory_basename(self) -> None:
        """The label is stable and logical, not the resolved path."""
        plan = mu.plan_cellpose_models("mean", 0, "cyto")
        self.assertEqual(plan.source_name, self.models.name)
        self.assertNotIn("/", plan.source_name)

    def test_source_name_falls_back_to_the_model_type(self) -> None:
        """A nameless directory still yields a non-empty label."""
        with patch.dict(
            os.environ, {mu.CELLPOSE_MODELS_PATH_ENV: "/"}, clear=True
        ):
            plan = mu.plan_cellpose_models("mean", 0, "cyto")
        self.assertEqual(plan.source_name, "cyto")


class TestCellposeProvenance(unittest.TestCase):
    """Records describe the bytes loaded, and label their origin honestly."""

    def setUp(self) -> None:
        """Point cellpose's model directory at an empty temp directory."""
        self.models = Path(tempfile.mkdtemp())
        patcher = patch.dict(
            os.environ,
            {mu.CELLPOSE_MODELS_PATH_ENV: str(self.models)},
            clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_locally_present_files_are_labelled_as_the_asset(self) -> None:
        """Both cyto files present up front are asset-sourced."""
        _write_model_files(
            self.models, "cytotorch_0", "size_cytotorch_0.npy"
        )
        plan = mu.plan_cellpose_models("mean", 0, "cyto")
        records = mu.build_cellpose_provenance(plan)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["source"], mu.MODEL_SOURCE_ASSET)
            self.assertEqual(record["source_name"], self.models.name)
            self.assertEqual(len(record["sha256"]), 64)
        self.assertEqual(
            {r["filename"] for r in records},
            {"cytotorch_0", "size_cytotorch_0.npy"},
        )

    def test_downloaded_file_is_labelled_as_a_network_download(self) -> None:
        """A file absent before the run and present after was downloaded."""
        _write_model_files(self.models, "cyto3")
        plan = mu.plan_cellpose_models("sourcery", 0, "cyto")
        _write_model_files(self.models, "size_cyto3.npy")
        records = mu.build_cellpose_provenance(plan)
        by_name = {r["filename"]: r for r in records}
        self.assertEqual(
            by_name["cyto3"]["source"], mu.MODEL_SOURCE_ASSET
        )
        self.assertEqual(
            by_name["size_cyto3.npy"]["source"], mu.MODEL_SOURCE_NETWORK
        )

    def test_missing_file_is_omitted(self) -> None:
        """A file absent throughout is left out, not asserted."""
        _write_model_files(self.models, "cytotorch_0")
        plan = mu.plan_cellpose_models("max", 0, "cyto")
        records = mu.build_cellpose_provenance(plan)
        self.assertEqual([r["filename"] for r in records], ["cytotorch_0"])

    @patch.dict(os.environ, _PIPELINE_ENV)
    def test_records_land_in_code_parameters(self) -> None:
        """Provenance recorded mid-run reaches the written document."""
        os.environ[mu.CELLPOSE_MODELS_PATH_ENV] = str(self.models)
        _write_model_files(
            self.models, "cytotorch_0", "size_cytotorch_0.npy"
        )
        out = Path(tempfile.mkdtemp())
        plan = mu.plan_cellpose_models("mean", 0, "cyto")
        code = mu.build_extraction_code({"init": "mean"})
        with stage_guard(
            out, mu.PROCESS_TYPE, code, plane_id="VISp_0"
        ) as ctx:
            ctx.parameters[mu.CELLPOSE_MODELS_PARAMETER] = (
                mu.build_cellpose_provenance(plan)
            )
        doc = Processing.model_validate_json(
            (out / "processing.json").read_text()
        )
        parameters = doc.data_processes[0].code.parameters.model_dump()
        recorded = parameters[mu.CELLPOSE_MODELS_PARAMETER]
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0]["source"], mu.MODEL_SOURCE_ASSET)


if __name__ == "__main__":
    unittest.main()
