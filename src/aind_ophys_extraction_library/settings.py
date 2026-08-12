"""CLI/environment settings for the ophys extraction pipeline.

Kept import-light (pydantic + stdlib only) so it stays cheap to import
and free of caiman / suite2p / aind-data-schema side effects.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionSettings(BaseSettings, cli_parse_args=True):
    """Settings for extraction pipeline using pydantic-settings."""

    # Basic parameters
    input_dir: Path = Field(
        default=Path("../data/"),
        description="Directory containing the input data files.",
    )
    output_dir: Path = Field(
        default=Path("../results/"),
        description="Directory where output files will be saved.",
    )
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description="Directory for temporary files created during processing.",
    )
    frame_rate: Optional[float] = Field(
        default=None,
        description=(
            "Frame rate [Hz] fallback used only when it cannot be "
            "read from acquisition/session/platform metadata."
        ),
    )

    # Cell detection parameters
    diameter: int = Field(
        default=0,
        description=(
            "Expected diameter of cells in pixels. "
            "If set to 0, CellPose will estimate the diameter from the data."
        ),
    )
    init: str = Field(
        default="mean",
        description=(
            "Initialization method for finding masks. Options: "
            "max/mean: Cellpose on max projection image divided by mean image; "
            "mean: Cellpose on mean image; "
            "enhanced_mean: Cellpose on enhanced mean image; "
            "max: Cellpose on maximum projection image; "
            "sourcery: Suite2p's functional mode without 'sparse_mode'; "
            "sparsery: Suite2p's functional mode with 'sparse_mode'; "
            "greedy_roi: CaImAn's functional 'greedy_roi' mode; "
            "corr_pnr: CaImAn's functional 'corr_pnr' mode."
        ),
    )
    functional_chan: int = Field(
        default=1,
        description="this channel is used to extract functional ROIs (1-based)",
    )
    threshold_scaling: int = Field(
        default=1,
        description="adjust the automatically determined threshold by this scalar multiplier",
    )
    max_overlap: float = Field(
        default=0.75,
        description="cells with more overlap than this get removed during triage, before refinement",  # noqa: E501
    )
    soma_crop: bool = Field(
        default=False,
        description="crop dendrites for cell classification stats like compactness",
    )
    allow_overlap: bool = Field(
        default=False,
        description="pixels that are overlapping are thrown out (False) or added to both ROIs (True)",  # noqa: E501
    )
    denoise: bool = Field(
        default=False,
        description=(
            "If True, applies denoising to the binned movie before cell detection."
        ),
    )
    cellprob_threshold: float = Field(
        default=0.0,
        description=(
            "Probability threshold for CellPose cell detection. "
            "Decrease this threshold if CellPose is not returning enough ROIs."
        ),
    )
    flow_threshold: float = Field(
        default=1.5,
        description=(
            "Flow threshold used by CellPose during cell detection. "
            "Increase this threshold if CellPose is not returning enough ROIs."
        ),
    )
    spatial_hp_cp: int = Field(
        default=0,
        description=(
            "Window size for spatial high-pass filtering of the image before CellPose "
            "detection. Set to 0 to disable filtering."
        ),
    )
    pretrained_model: str = Field(
        default="cyto",
        description=(
            "CellPose pretrained model to use. Common options: 'cyto' (standard model), "
            "'cyto2' (improved model), or path to a custom model file."
        ),
    )
    # Neuropil parameters
    neuropil: str = Field(
        default="mutualinfo",
        description=(
            "Method to estimate and subtract neuropil contamination, and whether to "
            "perform demixing. cnmf(-e) demix traces of overlapping ROIs via NMF, "
            "suite2p & mutualinfo do not. Options: "
            "'cnmf' (CaImAn standard: low-rank background of CNMF), "
            "'cnmf-e' (CaImAn for endoscopic data: ring model of CNMF-E ), "
            "'suite2p' (fixed r=0.7), "
            "'mutualinfo' (optimize r by minimizing mutual information)."
        ),
    )
    # CNMF parameters
    k_components: int = Field(
        default=5,
        description=(
            "Maximum number of components to be found per patch when using greedy_roi "
            "or corr_pnr initialization in CaImAn (CaImAn's 'K'). Named with more than "
            "one character so pydantic-settings renders a long CLI flag: a "
            "single-character field becomes a short option (-k), which a Code Ocean "
            "app panel cannot emit."
        ),
    )
    nb: int = Field(
        default=2,
        description=(
            "Number of background components if using CaImAn with neuropil=cnmf."
        ),
    )
    rf: Optional[int] = Field(
        default=40,
        description=(
            "Half-size of patches in pixels for CaImAn processing. If 0, the entire FOV "
            "is processed as a single patch. Larger patches are more memory-intensive."
        ),
    )
    stride: int = Field(
        default=18,
        description=(
            "Overlap between neighboring patches in pixels for CaImAn processing. "
            "Should be smaller than rf."
        ),
    )
    tsub: int = Field(
        default=2,
        description=(
            "Temporal downsampling factor during initialization phase in CaImAn. "
            "Higher values speed up processing but may miss transients."
        ),
    )
    ssub: int = Field(
        default=2,
        description=(
            "Spatial downsampling factor during initialization phase in CaImAn. "
            "Higher values speed up processing but may miss smaller cells."
        ),
    )
    ssub_B: int = Field(
        default=2,
        description=(
            "Additional spatial downsampling factor for background "
            "during CNMF-E processing."
        ),
    )
    merge_thr: float = Field(
        default=0.8,
        description=(
            "Trace correlation threshold for merging components in CaImAn. "
            "Components with correlation above this value will be merged."
        ),
    )

    # CORR_PNR parameters
    min_corr: float = Field(
        default=0.6,
        description=(
            "Minimum local correlation for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )
    min_pnr: float = Field(
        default=4,
        description=(
            "Minimum peak-to-noise ratio for a component to be considered in corr_pnr "
            "initialization. Higher values result in fewer, more reliable components."
        ),
    )

    # Component evaluation parameters
    snr_thr: float = Field(
        default=1.5,
        description=(
            "Signal-to-noise ratio threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    rval_thr: float = Field(
        default=0.6,
        description=(
            "Spatial correlation threshold for component acceptance in CaImAn "
            "evaluation. Components below this value will be rejected."
        ),
    )
    cnn_thr: float = Field(
        default=0.9,
        description=(
            "CNN classifier threshold for component acceptance in CaImAn evaluation. "
            "Components below this value will be rejected. Set to 0 to disable "
            "CNN-based classification."
        ),
    )

    # Output options
    contour_video: bool = Field(
        default=False,
        description=(
            "If True, creates a video overlaying raw data, ROI activity, and residual "
            "with contours for visualization and quality assessment."
        ),
    )

    verbose: bool = Field(
        default=False, description="Enable verbose logging and debug information."
    )

    verify: bool = Field(
        default=False,
        description=(
            "Round-trip the emitted processing.json / quality_control.json "
            "through the core v2 schema objects after writing, so bad output "
            "fails in-capsule rather than downstream."
        ),
    )

    # Suite2p parameter overrides
    suite2p_params: str = Field(
        default="",
        description=(
            "Serialized JSON string of Suite2p parameters to override defaults. "
            "Only applied when running a Suite2p-based init mode "
            "(max/mean, mean, enhanced_mean, max, sourcery, sparsery). "
            "Pipeline-controlled keys (input/output paths, fs, nbinned, "
            "do_registration, roidetect, spikedetect, neuropil_extract, "
            "bin_duration, and any key already exposed as a CLI flag) are "
            "stripped before merging; dropped keys are logged. "
            "Applied AFTER --suite2p-ops, so it wins on key collisions."
        ),
    )
    suite2p_ops: Optional[Path] = Field(
        default=None,
        description=(
            "Path to a Suite2p ops.npy file to load parameters from. "
            "Only applied when running a Suite2p-based init mode. "
            "Pipeline-controlled keys (same set stripped from --suite2p-params) "
            "are filtered before merging; dropped keys are logged. "
            "Merged BEFORE --suite2p-params so JSON overrides win."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="EXTRACTION_", case_sensitive=False, extra="ignore"
    )

    @field_validator("init", "neuropil")
    @classmethod
    def lowercase_str_fields(cls, v: str) -> str:
        """Convert string fields to lowercase."""
        return v.lower()

    @field_validator("rf")
    @classmethod
    def validate_rf(cls, v: int) -> Optional[int]:
        """Map rf=0 to None (full FOV mode)."""
        if v == 0:
            return None
        return v

    def validate_consistency(self) -> Optional[str]:
        """Validate command line arguments for consistency."""
        if self.neuropil == "cnmf" and self.init == "corr_pnr":
            self.ssub = 1
            return (
                "'corr_pnr' initialization with neuropil model 'cnmf' does "
                "not support spatial downsampling. Setting ssub to 1"
            )

        if self.neuropil == "cnmf-e" and self.init == "greedy_roi":
            raise ValueError(
                "Can't use neuropil model 'cnmf-e' with 'greedy_roi' initialization"
            )

        if self.init in ("greedy_roi", "corr_pnr") and self.neuropil[:4] != "cnmf":
            raise ValueError(
                "Can't use Suite2p neuropil model with 'greedy_roi' or 'corr_pnr' initialization"
            )

        # For backward compatibility
        if self.init in ("1", "2", "3", "4"):
            self.init = ("max/mean", "mean", "enhanced_mean", "max")[int(self.init) - 1]

        return None

    def model_post_init(self, _: Any) -> None:
        """Run validation after model initialization."""
        warning = self.validate_consistency()
        if warning:
            logging.warning(warning)
