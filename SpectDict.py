"""
esi_config.py
=============
Synthetic Data Generation Configuration for Electron Spectroscopy Imaging (ESI).

Defines ESI_CONFIG — a typed metadata registry for XPS, AES, EDS, and EELS
techniques. Intended as the single source of truth for a training-data generator
pipeline feeding a deep-learning spectral analysis model.

Author : Computational Materials Science & ML Engineering
License: MIT
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
NoiseProfile  = Dict[str, float]
SpecConfig    = Dict[str, Any]
PlotStyle     = Dict[str, Any]
WatermarkCfg  = Dict[str, Any]
LowResCfg     = Dict[str, Any]


# ---------------------------------------------------------------------------
# Primary Configuration Registry
# ---------------------------------------------------------------------------
ESI_CONFIG: Dict[str, SpecConfig] = {

    # -----------------------------------------------------------------------
    # XPS — X-ray Photoelectron Spectroscopy
    # Source: Al Kα (1486.6 eV) or Mg Kα (1253.6 eV) excitation.
    # Binding energy referenced to Fermi level (vacuum-referred for insulators).
    # Typical lab range : 0–1400 eV (Al Kα source).
    # Background       : Shirley (iterative) or Tougaard (physically rigorous).
    # -----------------------------------------------------------------------
    "XPS": {
        "x_axis"          : "Binding Energy",
        "y_axis"          : "Intensity",
        "x_units"         : "eV",
        "y_units"         : "CPS",           # Counts Per Second
        "x_range"         : (0.0, 1400.0),   # eV; Al Kα full survey range
        "noise_profile"   : {
            "gaussian_sigma"  : 0.45,         # eV; instrument broadening (FWHM ~1 eV)
            "poisson_lambda"  : 2500.0,        # mean photon-count rate (CPS)
        },
        "background_type" : "Shirley",        # iterative integral background
        "peak_shape"      : "Voigt",          # Gaussian-Lorentzian convolution
        "energy_reference": "Fermi Level",
        "charge_correction_eV": 0.0,          # set >0 for insulating samples
        "pass_energy_eV"  : 20.0,             # analyser pass energy (high-res mode)
    },

    # -----------------------------------------------------------------------
    # AES — Auger Electron Spectroscopy
    # Primary beam: 3–10 keV electrons; Auger electrons 20–2500 eV KE.
    # Spectra commonly plotted as dN(E)/dE (derivative) to enhance contrast.
    # Background       : Power Law (Shirley also valid in integral mode).
    # -----------------------------------------------------------------------
    "AES": {
        "x_axis"          : "Kinetic Energy",
        "y_axis"          : "dN(E)/dE",       # derivative mode (standard in AES)
        "x_units"         : "eV",
        "y_units"         : "a.u.",            # arbitrary units (derivative signal)
        "x_range"         : (20.0, 2500.0),    # eV; covers KLL, LMM, MNN transitions
        "noise_profile"   : {
            "gaussian_sigma"  : 0.60,           # eV; broader due to beam spread
            "poisson_lambda"  : 800.0,           # lower count rate than XPS
        },
        "background_type" : "Power Law",        # E^-n form; n typically 2.6–3.4
        "peak_shape"      : "Lorentzian",       # lifetime-broadened Auger lineshapes
        "primary_beam_keV": 5.0,                # incident electron energy
        "modulation_eV"   : 2.0,                # lock-in amplifier modulation voltage
        "derivative_order": 1,                  # 1 = dN/dE, 2 = d²N/dE² (rare)
    },

    # -----------------------------------------------------------------------
    # EDS — Energy Dispersive X-ray Spectroscopy
    # Detector: Si(Li) or SDD; characteristic X-ray lines 0.1–20 keV.
    # Background       : Bremsstrahlung continuum (Kramers' law / SNIP).
    # Resolution: 130–140 eV FWHM at Mn Kα (5.895 keV) for modern SDDs.
    # -----------------------------------------------------------------------
    "EDS": {
        "x_axis"          : "X-ray Energy",
        "y_axis"          : "Counts",
        "x_units"         : "keV",
        "y_units"         : "Counts",          # raw integer detector counts
        "x_range"         : (0.1, 20.0),       # keV; practical SDD range
        "noise_profile"   : {
            "gaussian_sigma"  : 0.065,          # keV; ~130 eV FWHM at Mn Kα
            "poisson_lambda"  : 15000.0,         # high count-rate typical for EDS
        },
        "background_type" : "Bremsstrahlung",   # Kramers / SNIP subtraction
        "peak_shape"      : "Gaussian",         # EDS peaks well-approximated by Gaussian
        "accelerating_kV" : 15.0,              # SEM/TEM accelerating voltage (kV)
        "detector_type"   : "SDD",             # Silicon Drift Detector
        "takeoff_angle_deg": 35.0,             # specimen-to-detector geometry
        "dead_time_pct"   : 20.0,              # target dead-time for live-time correction
    },

    # -----------------------------------------------------------------------
    # EELS — Electron Energy Loss Spectroscopy
    # TEM-based; probes low-loss (plasmons 5–50 eV) & core-loss (50–2000 eV).
    # Zero-loss peak (ZLP) at 0 eV is the dominant feature; must be stripped.
    # Background       : Power Law (Egerton model: AE^-r fit before each edge).
    # -----------------------------------------------------------------------
    "EELS": {
        "x_axis"          : "Energy Loss",
        "y_axis"          : "Intensity",
        "x_units"         : "eV",
        "y_units"         : "a.u.",            # relative units after ZLP normalisation
        "x_range"         : (-5.0, 2000.0),    # eV; −5 eV for ZLP tail, up to M-edges
        "noise_profile"   : {
            "gaussian_sigma"  : 0.10,           # eV; cold-FEG / monochromated resolution
            "poisson_lambda"  : 5000.0,          # counts per channel (core-loss region)
        },
        "background_type" : "Power Law",        # Egerton AE^-r; fit pre-edge window
        "peak_shape"      : "Voigt",            # core-loss edges are asymmetric
        "zlp_fwhm_eV"     : 0.3,               # zero-loss peak width (monochromator off ~1 eV)
        "collection_angle_mrad": 5.0,           # β collection semi-angle
        "convergence_angle_mrad": 10.0,         # α convergence semi-angle
        "accelerating_kV" : 200.0,             # TEM accelerating voltage (kV)
        "thickness_lambda": 0.5,               # specimen thickness in units of MFP
        "regions": {                            # named spectral regions for segmentation
            "zero_loss"  : (-5.0,   5.0),      # eV
            "low_loss"   : (  5.0,  50.0),     # plasmons, inter-band transitions
            "core_loss"  : ( 50.0, 2000.0),    # ionisation edges (element ID)
        },
    },
}


# ---------------------------------------------------------------------------
# Plot Style, Resolution & Watermark Configuration
# ---------------------------------------------------------------------------
# Consumed by matplotlib / plotly rendering pipelines and by the synthetic-
# image augmentation pass that stamps training figures before export.
#
# Three orthogonal concerns are kept in separate sub-dicts so they can be
# applied independently:
#   1. visual_style  — colours, markers, grid
#   2. low_res       — downsampling / degradation for robustness training
#   3. watermark     — text / logo overlay for provenance simulation
# ---------------------------------------------------------------------------
PLOT_STYLE_CONFIG: Dict[str, PlotStyle] = {

    # -----------------------------------------------------------------------
    # XPS — clean, publication-style; dark blue line on white, no markers
    # (survey spectra have thousands of points — markers would be unreadable)
    # -----------------------------------------------------------------------
    "XPS": {
        "visual_style": {
            "line_color"        : "#1A3A6B",        # deep navy — journal-standard
            "line_width"        : 1.5,              # pt
            "line_style"        : "solid",          # "-"
            "marker"            : None,             # suppressed for high-density data
            "marker_size"       : 0,
            "marker_edge_color" : None,
            "fill_under_curve"  : True,             # shaded area under envelope
            "fill_alpha"        : 0.12,
            "fill_color"        : "#1A3A6B",
            "grid_visible"      : True,
            "grid_axis"         : "both",           # "x", "y", or "both"
            "grid_linestyle"    : "--",
            "grid_alpha"        : 0.35,
            "grid_color"        : "#AAAAAA",
            "background_color"  : "#FFFFFF",        # axes face colour
            "figure_facecolor"  : "#F7F7F7",        # outer figure background
        },
        "low_res": {
            "enabled"           : True,
            "downsample_factor" : 8,                # keep every Nth point
            "blur_sigma_px"     : 1.2,              # Gaussian blur on rendered image
            "jpeg_quality"      : 45,               # JPEG compression artefacts (0-95)
            "bit_depth"         : 8,                # reduce from 16-bit to 8-bit
            "add_scan_lines"    : False,
            "target_dpi"        : 72,               # screen-res export
        },
        "watermark": {
            "enabled"           : True,
            "text"              : "XPS — SYNTHETIC / NOT FOR PUBLICATION",
            "font_size"         : 10,               # pt
            "font_color"        : "#CC0000",
            "font_alpha"        : 0.28,
            "position"          : "center",         # "center", "top-left", "bottom-right"
            "rotation_deg"      : 30,
            "logo_path"         : None,             # path to PNG logo (None = text only)
            "logo_alpha"        : 0.15,
            "logo_position"     : "bottom-right",
        },
    },

    # -----------------------------------------------------------------------
    # AES — derivative spectrum; green tones, subtle markers at zero-crossings
    # -----------------------------------------------------------------------
    "AES": {
        "visual_style": {
            "line_color"        : "#1E7A3A",        # forest green
            "line_width"        : 1.2,
            "line_style"        : "solid",
            "marker"            : "x",              # mark zero-crossings (peak positions)
            "marker_size"       : 5,
            "marker_edge_color" : "#0D4A22",
            "fill_under_curve"  : False,
            "fill_alpha"        : 0.0,
            "fill_color"        : None,
            "grid_visible"      : True,
            "grid_axis"         : "x",              # vertical lines only — tracks peak KE
            "grid_linestyle"    : ":",
            "grid_alpha"        : 0.40,
            "grid_color"        : "#888888",
            "background_color"  : "#FAFAFA",
            "figure_facecolor"  : "#F0F0F0",
        },
        "low_res": {
            "enabled"           : True,
            "downsample_factor" : 4,
            "blur_sigma_px"     : 0.8,
            "jpeg_quality"      : 55,
            "bit_depth"         : 8,
            "add_scan_lines"    : True,             # simulate old CRT / lock-in output
            "scan_line_spacing" : 6,               # px between horizontal scan lines
            "scan_line_alpha"   : 0.10,
            "target_dpi"        : 96,
        },
        "watermark": {
            "enabled"           : True,
            "text"              : "AES — SYNTHETIC / NOT FOR PUBLICATION",
            "font_size"         : 9,
            "font_color"        : "#005500",
            "font_alpha"        : 0.25,
            "position"          : "bottom-right",
            "rotation_deg"      : 0,
            "logo_path"         : None,
            "logo_alpha"        : 0.12,
            "logo_position"     : "top-left",
        },
    },

    # -----------------------------------------------------------------------
    # EDS — bar/histogram aesthetic; warm orange, discrete channel markers
    # -----------------------------------------------------------------------
    "EDS": {
        "visual_style": {
            "line_color"        : "#B84B00",        # burnt orange — echoes X-ray glow
            "line_width"        : 1.0,
            "line_style"        : "solid",
            "marker"            : "|",              # vertical tick per channel (histogram)
            "marker_size"       : 3,
            "marker_edge_color" : "#7A3000",
            "fill_under_curve"  : True,             # step-filled histogram look
            "fill_alpha"        : 0.20,
            "fill_color"        : "#E07030",
            "grid_visible"      : True,
            "grid_axis"         : "y",              # horizontal count-level guides
            "grid_linestyle"    : "-",
            "grid_alpha"        : 0.25,
            "grid_color"        : "#BBBBBB",
            "background_color"  : "#FFFDF8",
            "figure_facecolor"  : "#F5F0E8",
        },
        "low_res": {
            "enabled"           : True,
            "downsample_factor" : 2,                # EDS channels already coarse
            "blur_sigma_px"     : 0.5,
            "jpeg_quality"      : 60,
            "bit_depth"         : 8,
            "add_scan_lines"    : False,
            "target_dpi"        : 96,
            "quantise_y_levels" : 256,              # posterise counts to 8-bit depth
        },
        "watermark": {
            "enabled"           : True,
            "text"              : "EDS — SYNTHETIC / NOT FOR PUBLICATION",
            "font_size"         : 10,
            "font_color"        : "#8B2500",
            "font_alpha"        : 0.22,
            "position"          : "top-left",
            "rotation_deg"      : 0,
            "logo_path"         : None,
            "logo_alpha"        : 0.13,
            "logo_position"     : "bottom-right",
        },
    },

    # -----------------------------------------------------------------------
    # EELS — dark theme; vivid cyan on near-black (mirrors TEM software look),
    # circle markers on identified edge onsets, major grid only
    # -----------------------------------------------------------------------
    "EELS": {
        "visual_style": {
            "line_color"        : "#00C8E0",        # bright cyan — Gatan DigitalMicrograph
            "line_width"        : 1.3,
            "line_style"        : "solid",
            "marker"            : "o",              # circle at edge-onset positions
            "marker_size"       : 4,
            "marker_edge_color" : "#008FA0",
            "fill_under_curve"  : False,
            "fill_alpha"        : 0.0,
            "fill_color"        : None,
            "grid_visible"      : True,
            "grid_axis"         : "both",
            "grid_linestyle"    : "--",
            "grid_alpha"        : 0.20,
            "grid_color"        : "#404040",
            "background_color"  : "#111418",        # dark axes (TEM software convention)
            "figure_facecolor"  : "#0A0C0F",
        },
        "low_res": {
            "enabled"           : True,
            "downsample_factor" : 16,               # simulate low-dose / fast-acquire
            "blur_sigma_px"     : 2.0,              # stronger blur — monochromator off
            "jpeg_quality"      : 35,               # heavy compression for robustness
            "bit_depth"         : 6,                # extreme depth reduction
            "add_scan_lines"    : True,             # simulate CCD row noise
            "scan_line_spacing" : 4,
            "scan_line_alpha"   : 0.08,
            "target_dpi"        : 72,
            "add_detector_glow" : True,             # vignette / bloom at ZLP region
            "glow_sigma_px"     : 12,
            "glow_alpha"        : 0.30,
        },
        "watermark": {
            "enabled"           : True,
            "text"              : "EELS — SYNTHETIC / NOT FOR PUBLICATION",
            "font_size"         : 9,
            "font_color"        : "#00FFFF",
            "font_alpha"        : 0.20,
            "position"          : "center",
            "rotation_deg"      : 45,
            "logo_path"         : None,
            "logo_alpha"        : 0.10,
            "logo_position"     : "bottom-right",
        },
    },
}


# ---------------------------------------------------------------------------
# Helper: typed accessor
# ---------------------------------------------------------------------------
def get_config(technique: str) -> SpecConfig:
    """Return the configuration block for *technique*, with a clear KeyError."""
    if technique not in ESI_CONFIG:
        available = ", ".join(ESI_CONFIG.keys())
        raise KeyError(
            f"Unknown ESI technique '{technique}'. "
            f"Available options: {available}"
        )
    return ESI_CONFIG[technique]
def get_plot_style(technique: str) -> PlotStyle:
    """Return the full plot-style block for *technique*."""
    if technique not in PLOT_STYLE_CONFIG:
        available = ", ".join(PLOT_STYLE_CONFIG.keys())
        raise KeyError(
            f"No plot style registered for '{technique}'. "
            f"Available: {available}"
        )
    return PLOT_STYLE_CONFIG[technique]



def generate_synthetic_spectrum(
    technique : str,
    config    : SpecConfig,
    n_points  : int = 2048,
    n_spectra : int = 1,
) -> None:
    """
    Stub: initialise a synthetic spectrum generator for *technique*.

    In production, replace the body with:
      - numpy linspace over config["x_range"]
      - Voigt / Gaussian / Lorentzian peak synthesis
      - Shirley / Tougaard / Power-Law background addition
      - Poisson shot-noise + Gaussian detector noise overlay
      - yield (x, y) arrays to feed directly into a torch.Dataset

    Parameters
    ----------
    technique : str
        Key into ESI_CONFIG (e.g., "XPS").
    config    : SpecConfig
        Unpacked configuration sub-dictionary.
    n_points  : int
        Number of discrete energy-axis points (spectrum resolution).
    n_spectra : int
        Number of independent synthetic spectra to produce per call.
    """
    x_lo, x_hi = config["x_range"]
    print(
        f"[{technique:>4s}]  Initialising generator │ "
        f"x: {x_lo:>7.1f} – {x_hi:>7.1f} {config['x_units']:<4s} │ "
        f"y: {config['y_axis']:<12s} [{config['y_units']}] │ "
        f"BG: {config['background_type']:<14s} │ "
        f"n_pts: {n_points:,}  n_spectra: {n_spectra:,}"
    )


# ---------------------------------------------------------------------------
# Iteration example — entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 90)
    print("  ESI Synthetic Data Generator — Configuration Pass")
    print("=" * 90)

    SPECTRA_PER_TECHNIQUE: int = 5_000   # training samples per class
    POINTS_PER_SPECTRUM  : int = 2_048   # spectral resolution (power-of-2 for FFT)

    for technique, config in ESI_CONFIG.items():

        style    = PLOT_STYLE_CONFIG[technique]
        vs       = style["visual_style"]
        lr       = style["low_res"]
        wm       = style["watermark"]

        generate_synthetic_spectrum(
            technique  = technique,
            config     = config,
            n_points   = POINTS_PER_SPECTRUM,
            n_spectra  = SPECTRA_PER_TECHNIQUE,
        )

        # -- Style summary --------------------------------------------------
        grid_tag = f"grid={'ON' if vs['grid_visible'] else 'OFF'}({vs['grid_axis']})"
        lr_tag   = (
            f"low-res=ON  [↓{lr['downsample_factor']}x  "
            f"blur={lr['blur_sigma_px']}px  JPEG={lr['jpeg_quality']}  "
            f"{lr['target_dpi']}dpi]"
            if lr["enabled"] else "low-res=OFF"
        )
        wm_tag   = (
            f"watermark=ON  [alpha={wm['font_alpha']}  rot={wm['rotation_deg']}°]"
            if wm["enabled"] else "watermark=OFF"
        )
        print(
            f"         └─ color={vs['line_color']}  "
            f"marker={str(vs['marker']):<4s}  "
            f"{grid_tag:<20s}  {lr_tag}  {wm_tag}"
        )

    print("=" * 90)
    print(f"  Total techniques registered : {len(ESI_CONFIG)}")
    print(f"  Total spectra to generate   : {len(ESI_CONFIG) * SPECTRA_PER_TECHNIQUE:,}")
    print("=" * 90)