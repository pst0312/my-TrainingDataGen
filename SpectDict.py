"""
SpectDict.py

Centralized spectroscopy configuration and peak library for the synthetic
spectrum generator.

Exports:
- ESI_CONFIG
- PLOT_STYLE_CONFIG
- PEAK_LIBRARY

The PEAK_LIBRARY contains the five materials requested with per-technique
peak/edge definitions. Values are intentionally realistic but simplified for
synthetic generation; they can be tuned later.
"""
from typing import Dict, List

# Techniques supported by the generator
TECHNIQUES = [
    "XPS",
    "AES",
    "EDS",
    "EELS",
    "IR",
    "Raman",
]

# ESI_CONFIG: per-technique physical and generation defaults
ESI_CONFIG: Dict[str, Dict] = {
    "XPS": {
        "x_axis": "Binding Energy",
        "y_axis": "Intensity",
        "x_units": "eV",
        "y_units": "CPS",
        "x_range": (0.0, 1400.0),
        "axis_reversed": True,
        "noise_profile": {"gaussian_sigma": 0.45, "poisson_lambda": 2500.0},
        "background_type": "Shirley",
        "peak_shape": "Voigt",
        "particle_technique": True,
        "base_data_complexity": 4,
    },
    "AES": {
        "x_axis": "Kinetic Energy",
        "y_axis": "dN(E)/dE",
        "x_units": "eV",
        "y_units": "a.u.",
        "x_range": (20.0, 2500.0),
        "axis_reversed": False,
        "noise_profile": {"gaussian_sigma": 0.60, "poisson_lambda": 800.0},
        "background_type": "Power Law",
        "peak_shape": "Lorentzian",
        "particle_technique": True,
        "base_data_complexity": 5,
    },
    "EDS": {
        "x_axis": "X-ray Energy",
        "y_axis": "Counts",
        "x_units": "keV",
        "y_units": "Counts",
        "x_range": (0.1, 20.0),
        "axis_reversed": False,
        "noise_profile": {"gaussian_sigma": 0.065, "poisson_lambda": 15000.0},
        "background_type": "Bremsstrahlung",
        "peak_shape": "Gaussian",
        "particle_technique": True,
        "accelerating_voltage_keV": 15.0,
        "base_data_complexity": 2,
    },
    "EELS": {
        "x_axis": "Energy Loss",
        "y_axis": "Intensity",
        "x_units": "eV",
        "y_units": "a.u.",
        "x_range": (-5.0, 2000.0),
        "axis_reversed": False,
        "noise_profile": {"gaussian_sigma": 0.10, "poisson_lambda": 5000.0},
        "background_type": "Power Law",
        "peak_shape": "Edge",
        "particle_technique": True,
        "zlp_fwhm_eV": 0.3,
        "base_data_complexity": 7,
    },
    "IR": {
        "x_axis": "Wavenumber",
        "y_axis": "Transmittance",
        "x_units": "cm-1",
        "y_units": "%T",
        "x_range": (400.0, 4000.0),
        "axis_reversed": False,
        "noise_profile": {"gaussian_sigma": 2.0, "poisson_lambda": 1200.0},
        "background_type": "Polynomial",
        "peak_shape": "Lorentzian",
        "particle_technique": False,
        "baseline_level": 1.0,
        "fwhm_range": (2.0, 8.0),
        "base_data_complexity": 6,
    },
    "Raman": {
        "x_axis": "Raman Shift",
        "y_axis": "Intensity",
        "x_units": "cm-1",
        "y_units": "a.u.",
        "x_range": (0.0, 3500.0),
        "axis_reversed": False,
        "noise_profile": {"gaussian_sigma": 1.5, "poisson_lambda": 3000.0},
        "background_type": "Polynomial + Fluorescence",
        "peak_shape": "Voigt",
        "particle_technique": False,
        "base_data_complexity": 8,
    },
}

# PLOT_STYLE_CONFIG: visual aesthetics and low-res post-processing defaults
base_visual_style = {
    "figsize": (12, 7),
    "dpi": 100,
    "line_width": 1.5,
    "line_color": "#1A3A6B",
    "grid_visible": True,
    "fill_under_curve": False,
    "fill_alpha": 0.12,
    "background_color": "#FFFFFF",
    "figure_facecolor": "#F7F7F7",
}

base_low_res = {
    "enabled": True,
    "blur_sigma_px": 1.0,
    "downsample_factor": 3,
    "jpeg_quality": 70,
    "add_scan_lines": True,
    "scan_line_spacing": 4,
    "scan_line_alpha": 0.08,
    "paper_grain": True,
    "paper_grain_sigma": 6.0,
}

default_watermark = {"enabled": True, "text": "SYNTHETIC", "font_size": 10, "font_color": "#CC0000", "font_alpha": 0.28, "rotation_deg": 30}

# Create per-technique style entries reusing base definitions
PLOT_STYLE_CONFIG: Dict[str, Dict] = {}
for tech in TECHNIQUES:
    PLOT_STYLE_CONFIG[tech] = {
        "visual_style": dict(base_visual_style),
        "low_res": dict(base_low_res),
        "watermark": dict(default_watermark),
    }

# PEAK_LIBRARY: minimal, five materials with per-technique peak/edge lists
# Each technique entry contains a list of definitions. Keys per-entry:
#  - position: center or threshold (units per ESI_CONFIG)
#  - intensity: relative amplitude
#  - fwhm: full-width-half-maximum (same units as position)
#  - shape: 'gaussian','lorentzian','voigt','edge','zlp','derivative'
#  - notes: optional human-readable hints
PEAK_LIBRARY: Dict[str, Dict] = {
    "Sodium Azide": {
        "description": "Inorganic azide: Na and N features",
        "XPS": {
            "peaks": [
                {"position": 1072.0, "intensity": 0.6, "fwhm": 1.2, "shape": "voigt", "notes": "Na 1s (approx)"},
                {"position": 400.0, "intensity": 0.4, "fwhm": 1.5, "shape": "voigt", "notes": "N 1s (approx)"},
            ]
        },
        "EDS": {"peaks": [{"position": 1.04, "intensity": 0.7, "fwhm": 0.08, "shape": "gaussian", "notes": "Na K"}]},
        "AES": {"peaks": [{"position": 30.0, "intensity": 0.5, "fwhm": 2.0, "shape": "derivative"}]},
        "EELS": {"peaks": [{"position": 401.0, "intensity": 0.3, "fwhm": 3.0, "shape": "edge", "edge_threshold": 401.0}]},
        "IR": {"peaks": [{"position": 2100.0, "intensity": 0.6, "fwhm": 30.0, "shape": "lorentzian"}]},
        "Raman": {"peaks": [{"position": 990.0, "intensity": 0.2, "fwhm": 10.0, "shape": "voigt"}]},
    },
    "Gold": {
        "description": "Metallic gold (Au) reference",
        "XPS": {"peaks": [{"position": 84.0, "intensity": 1.0, "fwhm": 0.6, "shape": "voigt", "notes": "Au 4f7/2"}]},
        "EDS": {"peaks": [
            {"position": 2.12, "intensity": 1.0, "fwhm": 0.03, "shape": "gaussian", "notes": "Au L alpha (approx)"}
        ]},
        "AES": {"peaks": [{"position": 115.0, "intensity": 0.8, "fwhm": 1.5, "shape": "derivative"}]},
        "EELS": {"peaks": [{"position": 220.0, "intensity": 0.2, "fwhm": 4.0, "shape": "edge", "edge_threshold": 220.0}]},
        "IR": {"peaks": []},
        "Raman": {"peaks": []},
    },
    "Graphite": {
        "description": "Carbon allotrope - strong C 1s and Raman G/D bands",
        "XPS": {"peaks": [{"position": 284.5, "intensity": 1.0, "fwhm": 0.7, "shape": "voigt", "notes": "C 1s"}]},
        "EDS": {"peaks": [{"position": 0.277, "intensity": 0.6, "fwhm": 0.05, "shape": "gaussian", "notes": "C K"}]},
        "AES": {"peaks": [{"position": 272.0, "intensity": 0.6, "fwhm": 2.0, "shape": "derivative"}]},
        "EELS": {"peaks": [{"position": 285.0, "intensity": 0.5, "fwhm": 3.0, "shape": "edge", "edge_threshold": 285.0}]},
        "IR": {"peaks": []},
        "Raman": {"peaks": [
            {"position": 1580.0, "intensity": 1.0, "fwhm": 20.0, "shape": "voigt", "notes": "G band"},
            {"position": 1350.0, "intensity": 0.4, "fwhm": 30.0, "shape": "voigt", "notes": "D band"}
        ]},
    },
    "Iron Oxide": {
        "description": "Fe2O3 / iron oxide reference",
        "XPS": {"peaks": [{"position": 710.5, "intensity": 1.0, "fwhm": 1.2, "shape": "voigt", "notes": "Fe 2p3/2"}]},
        "EDS": {"peaks": [{"position": 6.40, "intensity": 0.9, "fwhm": 0.04, "shape": "gaussian", "notes": "Fe K alpha"}]},
        "AES": {"peaks": [{"position": 648.0, "intensity": 0.6, "fwhm": 3.0, "shape": "derivative"}]},
        "EELS": {"peaks": [{"position": 708.0, "intensity": 0.6, "fwhm": 5.0, "shape": "edge", "edge_threshold": 708.0}]},
        "IR": {"peaks": [{"position": 560.0, "intensity": 0.5, "fwhm": 25.0, "shape": "lorentzian"}]},
        "Raman": {"peaks": [{"position": 660.0, "intensity": 0.4, "fwhm": 20.0, "shape": "voigt"}]},
    },
    "Silicon": {
        "description": "Crystalline silicon",
        "XPS": {"peaks": [{"position": 99.4, "intensity": 1.0, "fwhm": 0.8, "shape": "voigt", "notes": "Si 2p"}]},
        "EDS": {"peaks": [{"position": 1.74, "intensity": 0.9, "fwhm": 0.03, "shape": "gaussian", "notes": "Si K alpha"}]},
        "AES": {"peaks": [{"position": 92.0, "intensity": 0.7, "fwhm": 1.8, "shape": "derivative"}]},
        "EELS": {"peaks": [{"position": 99.0, "intensity": 0.4, "fwhm": 4.0, "shape": "edge", "edge_threshold": 99.0}]},
        "IR": {"peaks": []},
        "Raman": {"peaks": [{"position": 520.0, "intensity": 1.0, "fwhm": 8.0, "shape": "voigt", "notes": "Si phonon"}]},
    },
}

# Exported names
__all__ = ["ESI_CONFIG", "PLOT_STYLE_CONFIG", "PEAK_LIBRARY", "TECHNIQUES"]
