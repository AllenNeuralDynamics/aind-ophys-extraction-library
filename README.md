# aind-ophys-extraction-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)

ROI detection and trace extraction for AIND planar/population ophys movies. This
is the backing library for the `aind-ophys-extraction` capsule: all logic lives
here, and the capsule is a thin wrapper that calls `job.run()`.

Detection runs through Cellpose, Suite2p and CaImAn. One invocation processes one
plane: it reads the asset metadata, locates the input movie, detects ROIs,
extracts and neuropil-corrects the traces, and writes the output HDF5 plus
`processing.json` and `quality_control.json` for that plane.

## Usage

```python
from aind_ophys_extraction_library.job import run

run()  # settings parsed from CLI/environment
```

Settings can also be constructed explicitly:

```python
from aind_ophys_extraction_library.job import run
from aind_ophys_extraction_library.settings import ExtractionSettings

run(ExtractionSettings(init="mean", neuropil="mutualinfo"))
```

### Outputs

Written under `<output_dir>/<plane_id>/extraction/`:

- `<plane_id>_extraction.h5` - traces (`traces/corrected`, `traces/neuropil`,
  `traces/roi`), sparse ROI masks, `iscell`, summary images, and the Cellpose
  output when Suite2p produced one.
- `<plane_id>_detected_ROIs.png` and `<plane_id>_detected_ROIs_withIDs.png`.
- `<plane_id>_ROI_contours_overlay.webm` when `contour_video` is set.
- `processing.json` and `quality_control.json`.

## Modules

### `settings.py`

`ExtractionSettings`, a `pydantic-settings` model parsed from CLI arguments and
`EXTRACTION_`-prefixed environment variables. Import-light on purpose (pydantic
plus stdlib only), so the Code Ocean app panel generator can read it without the
segmentation stack. Validators lowercase `init`/`neuropil`, map `rf=0` to `None`
(full-FOV CaImAn), and reject inconsistent `init`/`neuropil` combinations.

`init` selects the detection mode:

| `init` | Detection |
|---|---|
| `max/mean`, `mean`, `enhanced_mean`, `max` | Cellpose on the named summary image, via Suite2p's anatomical detection |
| `sourcery`, `sparsery` | Suite2p functional detection (with `sparse_mode` for `sparsery`) |
| `greedy_roi`, `corr_pnr` | CaImAn functional initialization |

`neuropil` selects trace correction: `mutualinfo` (r minimizing mutual
information), `suite2p` (fixed r), or `cnmf`/`cnmf-e` (CaImAn demixing, required
for the CaImAn `init` modes).

Suite2p defaults can be overridden with `suite2p_ops` (an `ops.npy` path) and
`suite2p_params` (a JSON string, applied second so it wins). Pipeline-controlled
keys are stripped from both, and dropped keys are logged.

### `job.py`

`run()`, the entry point. Reads metadata, configures logging, skips the plane if
a prior run's output is mounted next to the input, then runs detection and writes
every output inside the shared `stage_guard`, so a failed run still leaves a
`processing.json` with the error in `notes`.

### `extraction.py`

The computation. `run_suite2p_extraction()` and `run_caiman_extraction()` are the
two detection paths; `bergamo_segmentation()` builds the virtual dataset that
drops photostimulation frames; `get_FC_from_r()` does the mutual-information
neuropil correction; `save_summary_images_with_rois()` and `contour_video()`
produce the diagnostics; `write_extraction_h5()` writes the output file. Imports
the heavy dependencies at module level and is omitted from coverage, since
exercising it needs the full CO environment.

### `qc.py`

`build_detected_rois_metric()` builds the informational extracted-ROIs
curation metric (aind-data-schema's core `CurationMetric`, one figure
reference per detected ROI); `build_quality_control()` and
`write_quality_control_json()` delegate to `aind-pophys-metadata` so the
document shape stays fleet-wide.

### `utils/metadata_utils.py`

The metadata layer over `aind-pophys-metadata`.
`load_extraction_metadata()` dispatches on which core file is present
(`acquisition.json` for schema v2, `session.json` for v1) and returns the
`ExtractionMetadata` view this step consumes. `get_valid_epoch_stems()` resolves
the Bergamo TIFF stems from either schema version, and `is_bergamo()` routes on
`instrument_id`. `build_extraction_code()` builds the `Code` block,
`build_output_parameters()` the scalar output summary, and
`plan_cellpose_models()` / `build_cellpose_provenance()` record which Cellpose
model files a run loaded and whether they were already local. The model
directory is read from `CELLPOSE_LOCAL_MODELS_PATH`, which is Cellpose's own
variable - the deployment, not this library, decides where the weights live.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

## Tests and linting

Python 3.10 only (`requires-python = ">=3.10,<3.11"`), matching the capsule
image. The CI gate is

```bash
flake8 .
interrogate --verbose .
coverage run -m unittest discover
coverage report
```

Tests are `unittest`. Coverage and interrogate are both gated at 100%.
