"""
spectrum_generator.py
=====================
Generate realistic synthetic spectra using physics-based peak models and
material libraries. Applies advanced line shapes (Lorentzian, Voigt),
realistic backgrounds (Shirley, Bremsstrahlung), visual degradation, and
trailing lines for cross-polarization simulation.

This script:
  1. Accepts complexity ranges for both data and visual axes
  2. Selects a technique matching target data complexity
  3. Generates trailing lines for high-complexity data
  4. Applies physics-based peak synthesis
  5. Renders plot with visual degradation scaled to visual complexity
  6. Stores the data in a pandas DataFrame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special, ndimage
from scipy.signal import find_peaks, peak_widths, savgol_filter
from PIL import Image
import io
import random
import json
import argparse
from typing import Optional
from SpectDict import ESI_CONFIG, PLOT_STYLE_CONFIG, PEAK_LIBRARY



# ============================================================================
# LINE SHAPE FUNCTIONS (Physics-Based Peak Models)
# ============================================================================

def gaussian(x, center, fwhm, intensity=1.0):
    """
    Gaussian line shape: exp(-4*ln(2) * (x-center)² / fwhm²)
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return intensity * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def lorentzian(x, center, fwhm, intensity=1.0):
    """
    Lorentzian (Cauchy) line shape: intensity / (1 + (2(x-center)/fwhm)²)
    """
    gamma = fwhm / 2
    return intensity * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)


def voigt(x, center, fwhm_g, fwhm_l, intensity=1.0):
    """
    Voigt profile: convolution of Gaussian and Lorentzian.
    Uses scipy.special.wofz for efficient computation.
    
    Parameters
    ----------
    fwhm_g : float
        FWHM of Gaussian component (homogeneous broadening)
    fwhm_l : float
        FWHM of Lorentzian component (lifetime/pressure broadening)
    """
    sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_l / 2
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return intensity * np.real(special.wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def aes_derivative_profile(x, center, fwhm, intensity=1.0, asymmetry=0.2):
    """Generate an AES-like derivative peak: rapid dip followed by sharp spike.

    This implementation uses the analytical derivative of a Gaussian and
    applies a mild asymmetry factor to skew the response so the dip and
    spike are not perfectly symmetric.
    """
    # base Gaussian
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gauss = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    # analytical derivative d/dx of Gaussian ~ (x-center)*gauss
    deriv = -(x - center) * gauss / (sigma ** 2)
    # apply asymmetry by tilting the x-axis weighting
    skew = 1.0 + asymmetry * (x - center) / (fwhm + 1e-12)
    profile = intensity * deriv * skew
    # normalize to requested amplitude (peak-to-peak ~ intensity*2)
    if np.max(np.abs(profile)) > 0:
        profile = profile / np.max(np.abs(profile)) * intensity
    return profile


def zlp_profile(x, center=0.0, amplitude=1.0, fwhm=0.5):
    """Zero-loss peak for EELS: very narrow, very intense near 0 eV."""
    return gaussian(x, center, fwhm, amplitude)


def edge_profile(x, threshold, amplitude=1.0, decay_scale=50.0):
    """Simple edge: zero below threshold, sharp rise then gradual decay."""
    # smooth step at threshold followed by exponential/power-law decay
    shifted = x - threshold
    edge = np.zeros_like(x)
    mask = shifted >= 0
    edge[mask] = amplitude * (1.0 - np.exp(-shifted[mask] / (decay_scale + 1e-12))) * np.exp(-shifted[mask] / (decay_scale * 5.0 + 1e-12))
    return edge


# ============================================================================
# BACKGROUND FUNCTIONS (Physics-Based Models)
# ============================================================================

def shirley_background(x, y, n_iter=10, axis_reversed=False):
    """
    Iterative Shirley background subtraction for XPS/ESCA data.
    Approximates the accumulated intensity of inelastic scattering.
    
    Parameters
    ----------
    x : ndarray
        Energy axis (ascending order)
    y : ndarray
        Spectrum intensity
    n_iter : int
        Number of iterations (convergence)
    
    Returns
    -------
    ndarray
        Background estimate (same shape as y)
    """
    # If axis_reversed (e.g., XPS binding energy plotted high→low),
    # compute background on the reversed axis so the Shirley step behaves
    # correctly (it should step up when moving right-to-left for XPS).
    if axis_reversed:
        x_work = x[::-1]
        y_work = y[::-1]
    else:
        x_work = x
        y_work = y

    bg = np.copy(y_work)
    y_min = np.min(y_work)
    x_min, x_max = x_work[0], x_work[-1]

    for _ in range(n_iter):
        integral = np.cumsum(bg)
        integral = (integral - integral[0]) / (integral[-1] - integral[0] + 1e-12)
        bg_new = y_min + (np.max(y_work) - y_min) * integral
        bg = 0.5 * bg + 0.5 * bg_new

    # If we computed on reversed axis, flip back
    if axis_reversed:
        return bg[::-1]
    return bg


def bremsstrahlung_background(x, x_min, intensity=1000.0, k=1.7, E0=20.0):
    """
    Bremsstrahlung (Kramers' law) background for EDS.
    Approximates continuum X-ray generation: I(E) ∝ intensity / E^k
    
    Parameters
    ----------
    x : ndarray
        Energy axis (keV)
    x_min : float
        Minimum energy (typically threshold of first X-ray line)
    intensity : float
        Background intensity scaling
    k : float
        Power law exponent (typically 1.5–2.0)
    
    Returns
    -------
    ndarray
        Bremsstrahlung background
    """
    # Convert to numpy array and ensure energies are positive
    x_keV = np.maximum(x, 1e-6)
    # Kramers-like shape: (E0 - E)/E * (1/E^k) with a soft peak and cutoff
    prefactor = (np.maximum(E0 - x_keV, 0.0) / (E0 + 1e-12))
    # Suppress the unphysical low-energy divergence and produce a peak around 1-2 keV
    low_energy_rise = x_keV ** 2.65 / (x_keV + 0.5)
    raw = prefactor * low_energy_rise / (x_keV ** k)
    # normalize raw shape to have max == 1 then scale
    if np.max(raw) > 0:
        raw = raw / np.max(raw)
    return intensity * raw


def polynomial_baseline(x, degree=1, coeffs=None):
    """
    Polynomial baseline (common in IR and Raman).
    
    Parameters
    ----------
    x : ndarray
        Axis values
    degree : int
        Polynomial degree (1=linear, 2=quadratic, etc.)
    coeffs : ndarray, optional
        Polynomial coefficients (if None, generated randomly)
    
    Returns
    -------
    ndarray
        Polynomial background
    """
    if coeffs is None:
        # Random coefficients for natural variation
        coeffs = np.random.uniform(-0.5, 0.5, degree + 1)
    return np.polyval(coeffs, x)


def power_law_background(x, scale=1.0, exponent=-1.5):
    """Power-law baseline used for AES/EELS backgrounds."""
    x_safe = np.maximum(x, 1e-6)
    return scale * np.power(x_safe, exponent)


def transmittance_baseline(x, baseline_level=1.0, slope=0.01):
    """
    Transmittance baseline for absorbance/transmittance spectroscopy (FTIR, UV-Vis).
    Starts at a high level (typically 0.95-1.0 or 95-100%) with optional drift.
    
    Parameters
    ----------
    x : ndarray
        Axis values (wavenumber or wavelength)
    baseline_level : float
        High baseline value (1.0 for 100% transmittance, 0.95 for 95%)
    slope : float
        Linear drift per unit x (simulates instrument baseline drift)
    
    Returns
    -------
    ndarray
        Transmittance baseline (starts high, absorption dips downward)
    """
    # Normalize x to 0-1 range for slope calculation
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) > np.min(x) else 0
    drift = slope * x_norm * baseline_level * 0.05  # small drift
    baseline = baseline_level - drift + np.random.normal(0, baseline_level * 0.01, len(x))
    return np.maximum(baseline, baseline_level * 0.85)  # keep above 85% of baseline


# ============================================================================
# MAIN SPECTRUM GENERATION
# ============================================================================

def generate_synthetic_data(
    technique: str,
    config: dict,
    material: str = None,
    n_points: int = 2048,
    n_lines: int = 1,
    data_complexity: int = 5,
    seed: int = None,
) -> dict:
    """
    Generate realistic synthetic spectra with physics-based line shapes
    and material-specific peak positions from PEAK_LIBRARY. Optionally includes
    trailing lines for cross-polarization simulation (high complexity data).
    
    Parameters
    ----------
    technique : str
        Spectroscopy technique ("XPS", "AES", "EDS", "EELS", "IR", "Raman")
    config : dict
        Configuration dictionary for the technique
    material : str, optional
        Material name from PEAK_LIBRARY. If None, randomly selects from available.
    n_points : int
        Number of data points per spectrum
    n_lines : int
        Number of independent lines to generate (1-5)
    data_complexity : int
        Complexity score (1-10) controlling trailing line probability and n_lines.
        1 = simple single line, 10 = complex multi-line with trailing lines
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary mapping line_id to (x, y) tuples. Includes trailing lines
        (identified with _trailing suffix) when data_complexity is high.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_lines = max(1, min(n_lines, 5))
    
    # Generate x-axis
    x_lo, x_hi = config["x_range"]
    x = np.linspace(x_lo, x_hi, n_points)
    if technique == "EELS" and x_lo < 0.0 < x_hi and not np.any(np.isclose(x, 0.0)):
        x = np.sort(np.concatenate([x, [0.0]]))
        n_points = len(x)
    axis_reversed = config.get("axis_reversed", False)
    
    # Get noise parameters (kept conservative to preserve peak visibility)
    noise = config.get("noise_profile", {})
    gaussian_sigma = noise.get("gaussian_sigma", 0.1)
    poisson_lambda = noise.get("poisson_lambda", 1000)
    
    # Select or validate material
    if material is None:
        # Find available materials for this technique in PEAK_LIBRARY
        available_materials = [
            m for m in PEAK_LIBRARY
            if technique in PEAK_LIBRARY[m]
        ]
        if available_materials:
            material = random.choice(available_materials)
        else:
            material = None
    
    spectra = {}
    
    # Check if this is transmittance mode (FTIR, UV-Vis)
    is_transmittance = config.get("background_type") == "Transmittance"
    baseline_level = config.get("baseline_level", 1.0)
    
    for line_id in range(1, n_lines + 1):
        # For AES we will build a raw signal and then take the first derivative
        if technique == "AES":
            y_raw = np.zeros_like(x)
        if is_transmittance:
            # For transmittance mode, start at high baseline
            y = np.full_like(x, baseline_level)
        else:
            # For emission/absorption modes, start from zero
            y = np.zeros_like(x)
        
        # ============================================================
        # PEAK GENERATION (Physics-based or from library)
        # ============================================================
        
        if material and material in PEAK_LIBRARY:
            # Use peaks from material library
            peak_data = PEAK_LIBRARY[material].get(technique, {}).get("peaks", [])
            
            # Add slight variation in intensity and width (sample variation)
            for peak_info in peak_data:
                pos = peak_info["position"]
                fwhm = peak_info["fwhm"] * np.random.uniform(0.9, 1.1)
                
                # For transmittance mode, invert peak logic
                if is_transmittance:
                    # Convert library intensity to transmittance dip depth (20-60% absorption)
                    # Library intensity is relative; for FTIR we want strong dips
                    intensity = (peak_info["intensity"] / 100.0) * np.random.uniform(0.85, 1.15) * baseline_level
                    # Scale to reasonable dip depth (not too shallow)
                    intensity = np.clip(intensity, baseline_level * 0.15, baseline_level * 0.5)
                else:
                    # Normal mode: peaks are upward
                    intensity = peak_info["intensity"] * np.random.uniform(0.85, 1.15)
                
                # Choose shape based on technique and peak metadata
                line_shape = peak_info.get("shape", config.get("peak_shape", "Gaussian")).lower()

                if technique == "AES" or line_shape in ("derivative", "asymmetric_derivative"):
                    # For AES and derivative-specified peaks, accumulate into raw signal
                    peak = aes_derivative_profile(x, pos, fwhm, intensity)
                    y_raw += peak
                elif technique == "EELS" and line_shape == "zlp":
                    # Add ZLP explicitly (very narrow, intense)
                    peak = zlp_profile(x, center=pos, amplitude=intensity * 10.0, fwhm=max(0.3, fwhm * 0.2))
                    y += peak
                elif technique == "EELS" and line_shape == "edge":
                    # edge_profile expects threshold
                    peak = edge_profile(x, threshold=peak_info.get("edge_threshold", pos), amplitude=intensity, decay_scale=max(10.0, fwhm))
                    y += peak
                else:
                    if line_shape == "gaussian":
                        peak = gaussian(x, pos, fwhm, intensity)
                    elif line_shape == "lorentzian":
                        peak = lorentzian(x, pos, fwhm, intensity)
                    elif line_shape == "voigt":
                        fwhm_g = fwhm * 0.7
                        fwhm_l = fwhm * 0.3
                        peak = voigt(x, pos, fwhm_g, fwhm_l, intensity)
                    elif line_shape == "edge":
                        peak = edge_profile(x, threshold=peak_info.get("edge_threshold", pos), amplitude=intensity, decay_scale=max(10.0, fwhm))
                    else:
                        peak = gaussian(x, pos, fwhm, intensity)

                    if is_transmittance:
                        y -= peak
                    else:
                        y += peak
        else:
            # Fallback: random peak generation (for techniques without library)
            num_peaks = np.random.randint(2, 6)
            peak_positions = np.random.uniform(x_lo, x_hi, num_peaks)
            
            if is_transmittance:
                # For transmittance, generate deeper dips
                # Narrower FWHM for transmittance (sharp molecular transitions)
                fwhm_range = config.get("fwhm_range", (2.0, 8.0))
                peak_widths = np.random.uniform(fwhm_range[0], fwhm_range[1], num_peaks)
                # Dip depth: 20-60% of baseline (strong absorption)
                peak_depths = np.random.uniform(baseline_level * 0.2, baseline_level * 0.6, num_peaks)
            else:
                peak_widths = np.random.uniform(5, 50, num_peaks)
                peak_depths = np.random.uniform(10, 100, num_peaks)
            
            line_shape = config.get("peak_shape", "Gaussian")
            
            for pos, height, width in zip(peak_positions, peak_widths, peak_depths):
                if line_shape.lower() == "lorentzian":
                    peak = lorentzian(x, pos, width, height)
                elif line_shape.lower() == "voigt":
                    peak = voigt(x, pos, width * 0.7, width * 0.3, height)
                else:
                    peak = gaussian(x, pos, width, height)

                if is_transmittance:
                    y -= peak
                else:
                    y += peak
        
        # ============================================================
        # Add EELS zero-loss peak in every EELS spectrum
        # ============================================================
        if technique == "EELS":
            zlp_fwhm = config.get("zlp_fwhm_eV", 0.3)
            # ZLP should be prominent but not completely drown core-loss peaks
            zlp_amp = max(5.0, np.max(y) * 3.0 if np.max(y) > 0 else 50.0)
            y += zlp_profile(x, center=0.0, amplitude=zlp_amp, fwhm=zlp_fwhm)

        # ============================================================
        # BACKGROUND GENERATION (Physics-based)
        # ============================================================
        
        bg_type = config.get("background_type", "Linear")
        
        if bg_type == "Transmittance":
            # FTIR / UV-Vis transmittance baseline
            baseline = transmittance_baseline(x, baseline_level=baseline_level, slope=0.01)
            # Replace the high baseline we started with
            y = baseline + (y - baseline_level)
        elif bg_type == "Shirley":
            # Create temporary y with peaks for Shirley estimation
            temp_y = np.copy(y) + np.random.normal(0, 0.5, n_points)
            bg = shirley_background(x, temp_y, n_iter=5, axis_reversed=axis_reversed)
            y += bg
        elif bg_type == "Bremsstrahlung":
            # EDS-style background
            x_min = x_lo
            E0 = config.get("accelerating_voltage_keV", 20.0)
            bg = bremsstrahlung_background(x, x_min, intensity=np.max(y) * 0.3, k=1.7, E0=E0)
            y += bg
        elif bg_type == "Polynomial" or "Polynomial" in bg_type:
            # IR / Raman polynomial baseline
            bg = polynomial_baseline(x, degree=np.random.randint(1, 3))
            bg = np.maximum(bg, 0)  # ensure non-negative
            bg_scaled = bg * np.max(y) * 0.2
            y += bg_scaled
        elif "Power" in bg_type:
            # Power-law background for AES/EELS using a shifted x-axis to keep values finite.
            bg_scale = np.max(y) * 0.15 + 2.0
            x_positive = x - x_lo + 1.0
            y += power_law_background(x_positive, scale=bg_scale, exponent=-1.6)
        else:
            # Linear baseline (default) scaled relative to peak amplitude so it
            # doesn't dwarf peaks when peaks are small
            if not is_transmittance:
                base_min = max(0.5, np.max(y) * 0.005)
                base_max = max(1.0, np.max(y) * 0.02)
                y += np.linspace(base_min, base_max, n_points)
        
        # ============================================================
        # NOISE ADDITION (Realistic detector & photon noise)
        # ============================================================
        
        if is_transmittance:
            # For transmittance, noise is relative to baseline (1.0)
            # Keep noise small so dips remain clear for training
            y += np.random.normal(0, gaussian_sigma * baseline_level * 0.0025, n_points)
            y = np.clip(y, 0, baseline_level * 1.05)  # keep in reasonable range
        else:
            # Normal noise handling (reduced amplitude to preserve peaks)
            if technique == "AES":
                # add noise to raw signal (smaller factor)
                y_raw += np.random.normal(0, gaussian_sigma * np.max(y_raw + 1e-6) * 0.005, n_points)
            else:
                y += np.random.normal(0, gaussian_sigma * max(np.max(y), 1.0) * 0.005, n_points)
            # Poisson shot noise (reduced)
            if technique == "AES":
                y_raw = np.maximum(y_raw, 0)
                y_raw += np.random.poisson(max(1, int(poisson_lambda / 50000)), n_points) / 200
            else:
                y = np.maximum(y, 0)
                y += np.random.poisson(max(1, int(poisson_lambda / 50000)), n_points) / 200
        
        # If AES, convert raw signal to first derivative (dN/dE)
        if technique == "AES":
            y = np.gradient(y_raw, x)
            # normalize derivative scale to be visually comparable to other techniques
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y)) * (np.max(np.abs(y_raw)) + 1e-12)

        # Add slight vertical offset between lines (for multi-line display)
        if not is_transmittance:
            y += (line_id - 1) * np.random.uniform(5, 20)

        spectra[line_id] = (x[::-1], y[::-1]) if axis_reversed else (x, y)
        
        # ============================================================
        # TRAILING LINE GENERATION (cross-polarization, high complexity)
        # ============================================================
        # At high data_complexity, generate trailing lines for some primary lines.
        # A trailing line shares the exact same x-axis and peak positions, but with
        # reduced overall intensity (0.3-0.7x), simulating cross-polarized measurements
        # or secondary optical correlations.
        
        trailing_probability = min(0.7, data_complexity / 10.0)  # up to 70% chance
        # Do not generate cross-polarization trailing lines for particle techniques
        allow_trailing = not config.get("particle_technique", True)
        if data_complexity >= 6 and allow_trailing and np.random.random() < trailing_probability:
            # Create trailing line with same peaks but reduced intensity
            if is_transmittance:
                y_trailing = np.full_like(x, baseline_level)
            else:
                y_trailing = np.zeros_like(x)
            
            trailing_intensity_scale = np.random.uniform(0.3, 0.7)
            
            if material and material in PEAK_LIBRARY:
                peak_data = PEAK_LIBRARY[material].get(technique, {}).get("peaks", [])
                for peak_info in peak_data:
                    pos = peak_info["position"]
                    fwhm = peak_info["fwhm"] * np.random.uniform(0.9, 1.1)
                    
                    if is_transmittance:
                        intensity = peak_info["intensity"] * trailing_intensity_scale * np.random.uniform(0.85, 1.15) * 0.5
                        intensity = np.minimum(intensity, baseline_level * 0.4)
                    else:
                        intensity = peak_info["intensity"] * trailing_intensity_scale * np.random.uniform(0.85, 1.15)
                    
                    line_shape = config.get("peak_shape", "Gaussian")
                    if line_shape == "Gaussian":
                        peak = gaussian(x, pos, fwhm, intensity)
                    elif line_shape == "Lorentzian":
                        peak = lorentzian(x, pos, fwhm, intensity)
                    elif line_shape == "Voigt":
                        peak = voigt(x, pos, fwhm * 0.7, fwhm * 0.3, intensity)
                    else:
                        peak = gaussian(x, pos, fwhm, intensity)
                    
                    if is_transmittance:
                        y_trailing -= peak
                    else:
                        y_trailing += peak
            else:
                # Fallback: scale primary line peaks
                num_peaks = np.random.randint(2, 6)
                peak_positions = np.random.uniform(x_lo, x_hi, num_peaks)
                
                if is_transmittance:
                    fwhm_range = config.get("fwhm_range", (2.0, 8.0))
                    peak_widths = np.random.uniform(fwhm_range[0], fwhm_range[1], num_peaks)
                    peak_depths = np.random.uniform(baseline_level * 0.05, baseline_level * 0.25, num_peaks)
                else:
                    peak_widths = np.random.uniform(5, 50, num_peaks)
                    peak_depths = np.random.uniform(10, 100, num_peaks) * trailing_intensity_scale
                
                line_shape = config.get("peak_shape", "Gaussian")
                for pos, width, height in zip(peak_positions, peak_widths, peak_depths):
                    if line_shape == "Lorentzian":
                        peak = lorentzian(x, pos, width, height)
                    elif line_shape == "Voigt":
                        peak = voigt(x, pos, width * 0.7, width * 0.3, height)
                    else:
                        peak = gaussian(x, pos, width, height)
                    
                    if is_transmittance:
                        y_trailing -= peak
                    else:
                        y_trailing += peak
            
            # Add same background and noise
            bg_type = config.get("background_type", "Linear")
            if bg_type == "Transmittance":
                baseline = transmittance_baseline(x, baseline_level=baseline_level * 0.9, slope=0.005)
                y_trailing = baseline + (y_trailing - baseline_level)
            elif bg_type == "Shirley":
                temp_y_trailing = np.copy(y_trailing) + np.random.normal(0, 0.5, n_points)
                bg = shirley_background(x, temp_y_trailing, n_iter=5)
                y_trailing += bg * 0.5  # scale background for trailing line
            elif bg_type == "Bremsstrahlung":
                x_min = x_lo
                bg = bremsstrahlung_background(x, x_min, intensity=np.max(y_trailing) * 0.3, k=1.7)
                y_trailing += bg * 0.5
            elif bg_type == "Polynomial" or "Polynomial" in bg_type:
                bg = polynomial_baseline(x, degree=np.random.randint(1, 3))
                bg = np.maximum(bg, 0)
                bg_scaled = bg * np.max(y_trailing) * 0.1
                y_trailing += bg_scaled
            elif "Power" in bg_type:
                y_trailing += 50 * np.exp(-0.001 * (x - x_lo)) * 0.5
            else:
                if not is_transmittance:
                    y_trailing += np.linspace(5, 20, n_points) * 0.5
            
            # Add noise
            if is_transmittance:
                y_trailing += np.random.normal(0, gaussian_sigma * baseline_level * 0.002, n_points)
                y_trailing = np.clip(y_trailing, 0, baseline_level * 1.05)
            else:
                y_trailing += np.random.normal(0, gaussian_sigma * max(np.max(y_trailing),1.0) * 0.005, n_points)
                y_trailing = np.maximum(y_trailing, 0)
                y_trailing += np.random.poisson(max(1, int(poisson_lambda / 100000)), n_points) / 200
            
            # Store trailing line with special key
            spectra["{0}_trailing".format(line_id)] = (x, y_trailing)
    
    return spectra


def create_dataframe(spectra: dict, technique: str, material: str = "Unknown") -> pd.DataFrame:
    """
    Create a pandas DataFrame from multiple spectrum lines.
    
    Parameters
    ----------
    spectra : dict
        Dictionary mapping line_id to (x, y) tuples
    technique : str
        Spectroscopy technique name
    material : str
        Material name from library (default: "Unknown")
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: energy, intensity, line_id, technique, material
    """
    records = []

    # Detect peaks for metadata export (top 6 per line)
    peaks_map = get_peaks_for_spectra(spectra, technique, ESI_CONFIG.get(technique, {}), top_n=6)

    for line_id, (x, y) in spectra.items():
        # Prepare flattened peak columns for this line
        peaks = peaks_map.get(line_id, [])
        # Build JSON-serializable metadata list
        peak_meta = []
        for p in peaks:
            peak_meta.append({
                "position": p.get("position"),
                "amplitude": p.get("amplitude"),
                "fwhm": p.get("fwhm"),
            })

        # Flatten first 6 peaks into dedicated columns (peak_1_position, etc.)
        flat = {}
        for idx in range(6):
            if idx < len(peaks):
                flat[f"peak_{idx+1}_position"] = peaks[idx].get("position")
                flat[f"peak_{idx+1}_amplitude"] = peaks[idx].get("amplitude")
                flat[f"peak_{idx+1}_fwhm"] = peaks[idx].get("fwhm")
            else:
                flat[f"peak_{idx+1}_position"] = None
                flat[f"peak_{idx+1}_amplitude"] = None
                flat[f"peak_{idx+1}_fwhm"] = None

        for energy, intensity in zip(x, y):
            rec = {
                "energy": energy,
                "intensity": intensity,
                "line_id": line_id,
                "technique": technique,
                "material": material,
                "peak_metadata": json.dumps(peak_meta),
            }
            rec.update(flat)
            records.append(rec)

    df = pd.DataFrame(records)
    return df


def detect_peaks(x: np.ndarray, y: np.ndarray, prominence: float = None, width: float = None):
    """Detect peaks and estimate positions, amplitudes, and FWHM.

    Returns a list of dicts: {'position','amplitude','fwhm','left_ips','right_ips'}
    """
    if prominence is None:
        # set heuristic prominence based on RMS noise
        prominence = max( np.std(y) * 5.0, (np.max(y) - np.min(y)) * 0.02 )
    # find peaks (works for positive peaks); for derivative-like signals, caller should pass abs(y)
    peaks_idx, props = find_peaks(y, prominence=prominence)
    results = []
    if len(peaks_idx) == 0:
        return results

    # estimate widths at half prominence using scipy.signal.peak_widths
    widths_results = peak_widths(y, peaks_idx, rel_height=0.5)
    # widths_results: (widths, h_eval, left_ips, right_ips)

    for i, pk in enumerate(peaks_idx):
        pos = float(x[pk])
        amp = float(y[pk])
        w = float(widths_results[0][i]) if widths_results[0].size > i else 0.0
        left_ip = float(widths_results[2][i]) if widths_results[2].size > i else np.nan
        right_ip = float(widths_results[3][i]) if widths_results[3].size > i else np.nan
        # convert width in samples to FWHM in x-units
        fwhm = w * (x[1] - x[0]) if w > 0 else 0.0
        results.append({
            "position": pos,
            "amplitude": amp,
            "fwhm": fwhm,
            "left_ips": left_ip,
            "right_ips": right_ip,
        })

    # Sort by amplitude descending
    results = sorted(results, key=lambda r: abs(r['amplitude']), reverse=True)
    return results


def baseline_subtract(x: np.ndarray, y: np.ndarray, method: str = 'savgol', **kwargs):
    """Compute a baseline to subtract for display purposes.

    Methods supported: 'savgol' (savitzky-golay smoothing), 'poly' (polynomial fit),
    'median' (median filter).
    Returns baseline array and y_corrected = y - baseline.
    """
    method = method.lower()
    if method == 'savgol':
        # window length must be odd and <= len(y)
        window = int(kwargs.get('window', min(101, len(y) // 5 * 2 + 1)))
        if window >= len(y):
            window = len(y) - 1 if len(y) % 2 == 0 else len(y)
        if window % 2 == 0:
            window = max(3, window - 1)
        polyorder = min(3, max(1, int(kwargs.get('polyorder', 2))))
        try:
            baseline = savgol_filter(y, window_length=window, polyorder=polyorder)
        except Exception:
            baseline = np.full_like(y, np.median(y))
    elif method == 'poly':
        deg = int(kwargs.get('deg', 2))
        coeffs = np.polyfit(x, y, deg)
        baseline = np.polyval(coeffs, x)
    else:
        # median filter fallback
        from scipy.ndimage import median_filter
        size = int(kwargs.get('size', max(3, len(y) // 50)))
        baseline = median_filter(y, size=size)

    y_corrected = y - baseline
    return baseline, y_corrected


def get_peaks_for_spectra(spectra: dict, technique: str, config: dict, top_n: int = 5):
    """Detect peaks for each line in spectra and return a dict mapping line_id -> peaks list."""
    peaks_by_line = {}
    for line_id, (x, y) in spectra.items():
        x = np.array(x)
        y = np.array(y)
        # Choose baseline subtraction and search strategy per technique
        tech = technique.upper()
        if tech == 'AES':
            # AES is derivative-like: detect on absolute value after smoothing
            baseline, y_corr = baseline_subtract(x, y, method='savgol', window=max(11, len(y)//200))
            search_y = ndimage.gaussian_filter(np.abs(y_corr), sigma=2.0)
            prominence = max(np.std(search_y) * 3.0, (np.max(search_y)-np.min(search_y)) * 0.01)
        elif tech == 'XPS':
            # XPS often needs smooth baseline removed; use savgol then detect
            baseline, y_corr = baseline_subtract(x, y, method='savgol', window=max(31, len(y)//100))
            search_y = ndimage.gaussian_filter(y_corr, sigma=1.5)
            prominence = max(np.std(search_y) * 2.0, (np.max(search_y)-np.min(search_y)) * 0.005)
        elif tech == 'EDS':
            # EDS continuum dominates; subtract a robust median baseline
            baseline, y_corr = baseline_subtract(x, y, method='median', size=max(3, len(y)//200))
            search_y = ndimage.gaussian_filter(y_corr, sigma=1.0)
            prominence = max(np.std(search_y) * 2.5, (np.max(search_y)-np.min(search_y)) * 0.01)
        elif tech == 'EELS':
            # EELS has ZLP at zero; subtract a low-degree polynomial pre-edge
            baseline, y_corr = baseline_subtract(x, y, method='poly', deg=2)
            search_y = ndimage.gaussian_filter(y_corr, sigma=1.2)
            prominence = max(np.std(search_y) * 2.0, (np.max(search_y)-np.min(search_y)) * 0.01)
        else:
            baseline, y_corr = baseline_subtract(x, y, method='savgol')
            search_y = ndimage.gaussian_filter(y_corr, sigma=1.0)
            prominence = None

        peaks = detect_peaks(x, search_y, prominence=prominence)
        peaks_by_line[line_id] = peaks[:top_n]
    return peaks_by_line


def plot_focus_regions(spectra: dict, technique: str, config: dict, style: dict, n_peaks: int = 3, window_factor: float = 3.0, out_dir: str = None):
    """Create focused zoom plots around top peaks for each line.

    Saves per-peak PNGs in `out_dir` or current folder. Returns list of saved filenames.
    """
    import os
    saved = []
    peaks_map = get_peaks_for_spectra(spectra, technique, config, top_n=n_peaks)
    vs = style['visual_style']
    lr = style['low_res']

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    for line_id, (x, y) in list(spectra.items()):
        x = np.array(x)
        y = np.array(y)
        peaks = peaks_map.get(line_id, [])
        for i, pk in enumerate(peaks):
            center = pk['position']
            fwhm = pk['fwhm'] if pk['fwhm'] > 0 else ( (x[-1]-x[0]) / 100.0 )
            half_width = max(abs(fwhm * window_factor), (x[-1] - x[0]) * 0.01)
            x_min = center - half_width
            x_max = center + half_width

            # create plot similar to plot_spectrum but zoomed
            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            ax.plot(x, y, color=vs.get('line_color', '#1A3A6B'), linewidth=vs.get('line_width', 1.2))
            ax.set_xlim(x_min, x_max)

            # set y-limits to show peak clearly
            local_mask = (x >= x_min) & (x <= x_max)
            if np.any(local_mask):
                y_local = y[local_mask]
                y_min = np.min(y_local)
                y_max = np.max(y_local)
                yrange = max(1e-6, y_max - y_min)
                ax.set_ylim(y_min - 0.1 * yrange, y_max + 0.2 * yrange)

            ax.set_title(f"{technique} — Line {line_id} Peak {i+1} @ {center:.3g}")
            ax.set_xlabel(f"{config.get('x_axis','Energy')} ({config.get('x_units','a.u.')})")
            ax.set_ylabel(f"{config.get('y_axis','Intensity')} ({config.get('y_units','a.u.')})")
            plt.tight_layout()

            fname = os.path.join(out_dir, f"{technique.lower()}_line{line_id}_peak{i+1}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            saved.append(fname)

    return saved


def plot_before_after_comparison(spectra: dict, technique: str, config: dict, style: dict, n_peaks: int = 3, window_factor: float = 3.0, out_dir: str = None):
    """Create side-by-side before/after comparison images for top peaks.

    Left: raw zoomed region. Right: baseline-subtracted, annotated with peak position and FWHM.
    """
    import os
    saved = []
    peaks_map = get_peaks_for_spectra(spectra, technique, config, top_n=n_peaks)
    vs = style.get('visual_style', {}) if isinstance(style, dict) else {}

    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    for line_id, (x, y) in list(spectra.items()):
        x = np.array(x)
        y = np.array(y)
        peaks = peaks_map.get(line_id, [])
        for i, pk in enumerate(peaks):
            center = pk['position']
            fwhm = pk['fwhm'] if pk['fwhm'] > 0 else ((x[-1]-x[0]) / 100.0)
            half_width = max(abs(fwhm * window_factor), (x[-1] - x[0]) * 0.01)
            x_min = center - half_width
            x_max = center + half_width

            # compute baseline-subtracted data for 'after' panel
            baseline, y_corr = baseline_subtract(x, y, method='savgol')

            fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

            # Before (raw)
            axs[0].plot(x, y, color=vs.get('line_color', '#333333'), linewidth=vs.get('line_width', 1.0))
            axs[0].set_xlim(x_min, x_max)
            axs[0].set_title(f"Before — Line {line_id} Peak {i+1} @ {center:.3g}")
            axs[0].set_xlabel(f"{config.get('x_axis','Energy')} ({config.get('x_units','a.u.')})")
            axs[0].set_ylabel(f"{config.get('y_axis','Intensity')} ({config.get('y_units','a.u.')})")

            # After (baseline-subtracted & annotated)
            axs[1].plot(x, y_corr, color=vs.get('highlight_color', '#B22222'), linewidth=vs.get('line_width', 1.2))
            # overlay baseline for reference (shifted to zero line)
            axs[1].plot(x, baseline, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
            axs[1].set_xlim(x_min, x_max)
            # annotate peak and FWHM
            axs[1].axvline(center, color='k', linestyle=':', linewidth=1.0)
            axs[1].text(center, np.max(y_corr[(x>=x_min)&(x<=x_max)])*0.9, f"{center:.3g}", ha='center', va='top')
            axs[1].set_title(f"After — Baseline-subtracted")
            axs[1].set_xlabel(f"{config.get('x_axis','Energy')} ({config.get('x_units','a.u.')})")

            plt.tight_layout()
            fname = os.path.join(out_dir, f"{technique.lower()}_line{line_id}_peak{i+1}_comparison.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            saved.append(fname)

    return saved


def apply_visual_degradation(fig_path: str, low_res_config: dict, visual_complexity: int = 5, dpi: int = 100, seed: Optional[int] = None, blur: bool = True):
    """
    Apply visual degradation effects to rendered spectrum image.
    Severity of degradation is scaled by visual_complexity (1-10).
    
    Parameters
    ----------
    fig_path : str
        Path to PNG image file
    low_res_config : dict
        Base configuration with keys: blur_sigma_px, downsample_factor, jpeg_quality, etc.
    visual_complexity : int
        Complexity score (1-10) that scales degradation severity
        1 = pristine, no degradation
        10 = maximum degradation
    dpi : int
        Original image DPI (used for scan line calculations)
    seed : int, optional
        Random seed for deterministic degradation (especially for paper grain)
    blur : bool
        Whether blur-based degradation should be applied.
        If False, the image is preserved with pristine visibility.
    """
    if not low_res_config.get("enabled", True) or not blur:
        return
    
    # Clamp visual_complexity to valid range
    visual_complexity = max(1, min(visual_complexity, 10))
    
    # Set seed for any random operations in degradation
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Scale degradation parameters based on complexity
    # complexity 1 → scale 0.0 (no degradation)
    # complexity 5 → scale 0.5 (medium degradation)
    # complexity 10 → scale 1.0 (full degradation)
    degradation_scale = (visual_complexity - 1) / 9.0
    
    # Load image
    img = Image.open(fig_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # ====== BLUR (scaled, conservative) ======
    base_blur_sigma = low_res_config.get("blur_sigma_px", 0.0)
    # reduce blur impact to preserve peak sharpness
    blur_sigma = base_blur_sigma * (degradation_scale * 0.6)
    if blur_sigma > 0.05:  # Apply only if noticeable
        for ch in range(3):
            img_array[:, :, ch] = ndimage.gaussian_filter(img_array[:, :, ch], sigma=blur_sigma)
    
    # ====== DOWNSAMPLING (scaled) ======
    base_downsample = low_res_config.get("downsample_factor", 1)
    # Scale downsampling: at complexity 1, use 1 (no downsampling)
    # at complexity 10, use full base_downsample
    downsample_factor = int(1 + (base_downsample - 1) * degradation_scale)
    # Cap downsample to avoid obliterating peaks
    downsample_factor = min(downsample_factor, max(1, int(base_downsample)))
    
    if downsample_factor > 1:
        h, w = img_array.shape[:2]
        # Simple mean pooling
        h_small = h // downsample_factor
        w_small = w // downsample_factor
        img_small = np.zeros((h_small, w_small, 3), dtype=np.float32)
        for i in range(h_small):
            for j in range(w_small):
                i_start, i_end = i * downsample_factor, (i + 1) * downsample_factor
                j_start, j_end = j * downsample_factor, (j + 1) * downsample_factor
                img_small[i, j] = np.mean(
                    img_array[i_start:i_end, j_start:j_end], axis=(0, 1)
                )
        # Upscale back (nearest neighbor for pixelated effect)
        img_array = np.repeat(np.repeat(img_small, downsample_factor, axis=0), downsample_factor, axis=1)
        # Crop to original size if needed
        img_array = img_array[:h, :w]
    
    # ====== SCAN LINES (scaled) ======
    if degradation_scale > 0.1 and low_res_config.get("add_scan_lines", False):
        spacing = low_res_config.get("scan_line_spacing", 4)
        base_alpha = low_res_config.get("scan_line_alpha", 0.1)
        alpha = base_alpha * degradation_scale  # Scale alpha with complexity
        h = img_array.shape[0]
        for i in range(0, h, spacing):
            img_array[i, :] = img_array[i, :] * (1 - alpha)  # darken scan line
    
    # ====== PAPER GRAIN (IR, scaled) ======
    if degradation_scale > 0.1 and low_res_config.get("paper_grain", False):
        # Reduce grain strength so features remain recognizable
        grain_sigma = max(0.5, low_res_config.get("paper_grain_sigma", 1.5) * (degradation_scale * 0.4))
        grain = np.random.normal(0, grain_sigma / 255.0, img_array.shape)
        img_array = np.clip(img_array + grain, 0, 1)
    
    
    # Convert back to uint8 and save as JPEG (if quality specified, scaled by complexity)
    img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img_out = Image.fromarray(img_array)
    
    base_jpeg_quality = low_res_config.get("jpeg_quality", None)
    if base_jpeg_quality is not None and base_jpeg_quality < 90:
        # Scale JPEG quality conservatively: do not degrade aggressively
        jpeg_quality = 95 - (95 - base_jpeg_quality) * degradation_scale * 0.6
        buffer = io.BytesIO()
        img_out.save(buffer, format="JPEG", quality=int(jpeg_quality))
        buffer.seek(0)
        img_out = Image.open(buffer).convert("RGB")
        img_out.save(fig_path, "PNG")
    else:
        img_out.save(fig_path, "PNG")


def plot_spectrum(df: pd.DataFrame, spectra: dict, technique: str, config: dict, style: dict) -> None:
    """
    Plot multiple synthetic spectrum lines with full style and visual degradation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spectrum data
    spectra : dict
        Dictionary mapping line_id to (x, y) tuples
    technique : str
        Spectroscopy technique name
    config : dict
        Configuration for the technique
    style : dict
        Plot style configuration (visual_style, low_res, watermark)
    """
    vs = style["visual_style"]  # visual style
    lr = style["low_res"]       # low-res degradation
    wm = style["watermark"]     # watermark config
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Color and line style palettes
    color_palette = [
        vs.get("line_color", "#1A3A6B"),
        "#C41E3A",
        "#2E8B57",
        "#FF8C00",
        "#663399",
    ]
    linestyle_palette = ["-", "--", "-.", ":", "-"]
    
    # Plot each line (including trailing lines if present)
    plotted_lines = set()
    for idx, (line_id, (x, y)) in enumerate(spectra.items()):
        is_trailing = isinstance(line_id, str) and "_trailing" in line_id
        
        if is_trailing:
            # Trailing line: use trailing_line_style from config, same color as parent
            parent_id = int(line_id.split("_")[0])
            color = color_palette[(parent_id - 1) % len(color_palette)]
            linestyle = vs.get("trailing_line_style", "--")
            label = f"Line {parent_id} (cross-polarized)"
            alpha = 0.6
        else:
            # Primary line
            color = color_palette[idx % len(color_palette)]
            linestyle = linestyle_palette[idx % len(linestyle_palette)]
            label = f"Line {line_id}"
            alpha = 0.85
            plotted_lines.add(line_id)
        
        ax.plot(
            x,
            y,
            color=color,
            linewidth=vs.get("line_width", 1.5),
            linestyle=linestyle,
            label=label,
            alpha=alpha,
        )
    
    # Fill under curve if enabled
    if vs.get("fill_under_curve", False):
        for idx, (line_id, (x, y)) in enumerate(spectra.items()):
            fill_color = vs.get("fill_color", color_palette[idx % len(color_palette)])
            ax.fill_between(x, y, alpha=vs.get("fill_alpha", 0.12), color=fill_color)
    
    # Configure axes
    x_units = config.get("x_units", "eV")
    y_label = config.get("y_axis", "Intensity")
    y_units = config.get("y_units", "a.u.")
    
    # Special handling for transmittance mode (FTIR, UV-Vis)
    is_transmittance = config.get("background_type") == "Transmittance"
    if is_transmittance:
        y_label = config.get("y_axis", "Transmittance")
        y_units = config.get("y_units", "%T")
    
    ax.set_xlabel(f"{config.get('x_axis', 'Energy')} ({x_units})", fontsize=12)
    ax.set_ylabel(f"{y_label} ({y_units})", fontsize=12)
    ax.set_title(
        f"Synthetic {technique} Spectrum — {len(spectra)} Lines",
        fontsize=14,
        fontweight="bold",
    )
    
    # Set Y-axis limits for transmittance mode
    if is_transmittance:
        baseline_level = config.get("baseline_level", 1.0)
        # For transmittance, set limits to show dips nicely (0 to 110% of baseline)
        if y_units == "%T":
            ax.set_ylim(0, baseline_level * 100 * 1.1)
        else:
            ax.set_ylim(0, baseline_level * 1.1)
    
    # Configure grid
    if vs.get("grid_visible", True):
        grid_axis = vs.get("grid_axis", "both")
        ax.grid(
            True,
            which="major",
            axis=grid_axis,
            linestyle=vs.get("grid_linestyle", "--"),
            alpha=vs.get("grid_alpha", 0.35),
            color=vs.get("grid_color", "#AAAAAA"),
        )
    
    # Set background colors
    ax.set_facecolor(vs.get("background_color", "#FFFFFF"))
    fig.patch.set_facecolor(vs.get("figure_facecolor", "#F7F7F7"))
    
    # Add watermark
    if wm.get("enabled", False):
        ax.text(
            0.5, 0.95,
            wm.get("text", f"{technique} — SYNTHETIC"),
            transform=ax.transAxes,
            fontsize=wm.get("font_size", 10),
            color=wm.get("font_color", "#CC0000"),
            alpha=wm.get("font_alpha", 0.28),
            ha="center",
            va="top",
            rotation=wm.get("rotation_deg", 30),
        )
    
    # Legend
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    
    # Return the plotting function for filename control
    return (fig, ax)
    

def save_spectrum_plot(fig, technique: str, index: int = None, low_res_config: dict = None, visual_complexity: int = 5, blur: bool = True) -> str:
    """
    Save spectrum plot with optional visual degradation scaled by visual_complexity.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    technique : str
        Spectroscopy technique name
    index : int, optional
        Index for filename
    low_res_config : dict, optional
        Low-resolution degradation config (base settings)
    visual_complexity : int
        Visual complexity score (1-10) that scales degradation severity
    blur : bool
        Whether blur-based image degradation is enabled.
        If False, the image remains pristine.
    
    Returns
    -------
    str
        Path to saved file
    """
    if index is not None:
        png_filename = f"spectrum_{technique.lower()}_multiline_{index}.png"
    else:
        png_filename = f"spectrum_{technique.lower()}_multiline.png"
    
    # Before saving, remove non-essential text annotations so batch images
    # contain only plot lines, axes, and grids (no titles, legends, watermarks)
    for ax in fig.get_axes():
        # remove title
        try:
            ax.set_title("")
        except Exception:
            pass
        # remove legend if present
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        # remove free text (watermarks, annotations), keep axis labels
        for txt in list(ax.texts):
            try:
                txt.remove()
            except Exception:
                pass

    fig.savefig(png_filename, dpi=100, bbox_inches="tight")
    print(f"✓ Plot saved (initial, clean): {png_filename}")
    
    # Apply visual degradation scaled by visual_complexity
    if low_res_config and low_res_config.get("enabled", False):
        apply_visual_degradation(
            png_filename,
            low_res_config,
            visual_complexity=visual_complexity,
            dpi=100,
            blur=blur,
        )
        if blur:
            print(f"✓ Visual degradation applied (complexity {visual_complexity}/10): {png_filename}")
        else:
            print(f"✓ Blur disabled, pristine image saved: {png_filename}")
    
    return png_filename


if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic spectroscopy data with controllable data and visual complexity."
    )
    parser.add_argument("index", nargs="?", type=int, default=None, help="Optional index for output files")
    parser.add_argument("--min-vis", type=int, default=1, help="Minimum visual complexity (1-10)")
    parser.add_argument("--max-vis", type=int, default=10, help="Maximum visual complexity (1-10)")
    parser.add_argument("--min-data", type=int, default=1, help="Minimum data complexity (1-10)")
    parser.add_argument("--max-data", type=int, default=10, help="Maximum data complexity (1-10)")
    parser.add_argument("--blur", dest="blur", action="store_true", default=True, help="Enable blur-based visual degradation")
    parser.add_argument("--no-blur", dest="blur", action="store_false", help="Disable all blur surface degradation and keep plot pristine")
    
    args = parser.parse_args()
    
    # Validate and clamp complexity ranges
    min_vis = max(1, min(args.min_vis, 10))
    max_vis = max(1, min(args.max_vis, 10))
    min_data = max(1, min(args.min_data, 10))
    max_data = max(1, min(args.max_data, 10))
    
    # Ensure min <= max
    if min_vis > max_vis:
        min_vis, max_vis = max_vis, min_vis
    if min_data > max_data:
        min_data, max_data = max_data, min_data
    
    # Select random complexity scores within provided ranges
    target_visual_complexity = np.random.randint(min_vis, max_vis + 1)
    target_data_complexity = np.random.randint(min_data, max_data + 1)
    
    print("=" * 90)
    print("  Synthetic Spectrum Generator — Two-Axis Complexity Control")
    print("=" * 90)
    print(f"\nComplexity Targets:")
    print(f"  Data Complexity: {target_data_complexity}/10 (range: {min_data}-{max_data})")
    print(f"  Visual Complexity: {target_visual_complexity}/10 (range: {min_vis}-{max_vis})")
    
    # Filter techniques by base_data_complexity (within ±2 of target)
    suitable_techniques = []
    for tech, config in ESI_CONFIG.items():
        base_complexity = config.get("base_data_complexity", 5)
        if abs(base_complexity - target_data_complexity) <= 2:
            suitable_techniques.append(tech)
    
    # If no suitable techniques, use all
    if not suitable_techniques:
        suitable_techniques = list(ESI_CONFIG.keys())
    
    technique = random.choice(suitable_techniques)
    config = ESI_CONFIG[technique]
    style = PLOT_STYLE_CONFIG[technique]
    base_data_complexity = config.get("base_data_complexity", 5)
    
    # Select material randomly from available options
    available_materials = [
        m for m in PEAK_LIBRARY
        if technique in PEAK_LIBRARY[m]
    ]
    material = random.choice(available_materials) if available_materials else None
    
    # Scale n_lines based on data complexity (1-5 range)
    # Low complexity (1-2) → 1 line
    # Medium complexity (4-6) → 2-3 lines
    # High complexity (9-10) → 4-5 lines
    if target_data_complexity <= 3:
        num_lines = 1
    elif target_data_complexity <= 5:
        num_lines = np.random.randint(1, 3)
    elif target_data_complexity <= 7:
        num_lines = np.random.randint(2, 4)
    else:
        num_lines = np.random.randint(3, 6)
    
    print(f"\nSelected technique: {technique}")
    print(f"  Base data complexity: {base_data_complexity}/10")
    if material:
        print(f"  Selected material: {material}")
    print(f"  Number of lines to generate: {num_lines}")
    print(f"  Trailing lines likely: {'Yes' if target_data_complexity >= 6 else 'No'}")
    
    # Generate synthetic spectra with both complexity axes
    print(f"\nGenerating synthetic spectra...")
    spectra = generate_synthetic_data(
        technique=technique,
        config=config,
        material=material,
        n_points=2048,
        n_lines=num_lines,
        data_complexity=target_data_complexity,
        seed=42,  # For reproducibility
    )
    
    # Create DataFrame
    material_name = material if material else "Unknown"
    df = create_dataframe(spectra, technique, material=material_name)
    print(f"\nDataFrame created:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique lines: {df['line_id'].unique().tolist()}")
    print(f"\nFirst 15 rows:")
    print(df.head(15))
    print(f"\nDataFrame statistics (by line):")
    print(df.groupby("line_id")[["energy", "intensity"]].describe())
    
    # Save to CSV
    if args.index is not None:
        csv_filename = f"spectrum_data_{technique.lower()}_multiline_{args.index}.csv"
    else:
        csv_filename = f"spectrum_data_{technique.lower()}_multiline.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Data saved: {csv_filename}")

    # ALSO: write dataset schema markdown companion for vision models and downstream parsers
    schema_path = csv_filename.replace('.csv', '_dataset_schema.md')
    schema_lines = [
        f"# Dataset Schema for {csv_filename}",
        "",
        "This file documents the CSV layout produced by the synthetic spectrum generator.",
        "Use this to map visual PNGs to the numeric columns programmatically.",
        "",
        "## Column Definitions",
        "- **energy**: float — X-axis value (units in `x_units` column or in the generator config).",
        "- **intensity**: float — Measured intensity/counts at the corresponding energy/wavenumber.",
        "- **line_id**: int|string — Identifier for the spectrum line (multiple lines may be present per file).",
        "- **technique**: string — Spectroscopy technique (e.g., XPS, AES, EDS, EELS, IR, Raman).",
        "- **material**: string — Material selection from the `PEAK_LIBRARY` used to inject peaks (or 'Unknown').",
        "- **peak_metadata**: JSON string — Array of peak objects injected/detected for this `line_id`. Each object: `{position, amplitude, fwhm}`.",
        "",
        "## Flattened Peak Columns (first 6 peaks)",
        "- **peak_1_position**, **peak_1_amplitude**, **peak_1_fwhm**: floats — First (largest) peak for the line. Subsequent `peak_n_*` follow the same pattern up to 6.",
        "",
        "## Notes",
        "- Each row corresponds to a single point on a spectrum; to get per-spectrum metadata, group rows by `line_id`.",
        "- `peak_metadata` is provided as a JSON string per-row but is identical for all rows sharing the same `line_id` (it describes the line, not the point).",
        "- PNG visualizations produced by the generator contain only plot lines, axes, and grids (no textual annotations).",
        "",
        "## Example JSON `peak_metadata`",
        "```json",
        "[{\"position\": 284.5, \"amplitude\": 1.0, \"fwhm\": 0.7}]",
        "```",
    ]
    try:
        with open(schema_path, 'w') as sf:
            sf.write('\n'.join(schema_lines))
        print(f"\n✓ Dataset schema written: {schema_path}")
    except Exception as e:
        print(f"Warning: could not write dataset schema: {e}")
    
    # Display line summary (handle both int and string line IDs)
    print(f"\nLine Summary:")
    # Sort line IDs with custom logic (ints first, then strings)
    line_ids = df['line_id'].unique()
    int_ids = sorted([lid for lid in line_ids if isinstance(lid, int)])
    str_ids = sorted([lid for lid in line_ids if isinstance(lid, str)])
    sorted_line_ids = int_ids + str_ids
    
    for line_id in sorted_line_ids:
        line_data = df[df['line_id'] == line_id]
        print(f"  Line {line_id}: {len(line_data)} points | "
              f"Intensity range: {line_data['intensity'].min():.2f} – {line_data['intensity'].max():.2f}")
    
    # Create visualization with physics-based rendering
    print(f"\nGenerating multi-line plot with visual degradation...")
    fig, ax = plot_spectrum(df, spectra, technique, config, style)
    
    # Save with visual degradation scaled to visual_complexity
    png_filename = save_spectrum_plot(
        fig,
        technique,
        index=args.index,
        low_res_config=style.get("low_res", {}),
        visual_complexity=target_visual_complexity,
        blur=args.blur,
    )
    
    plt.close(fig)
    
    print("\n" + "=" * 90)
    print("  Generation complete!")
    print("  Data complexity is decoupled from visual complexity")
    print("  Complex data may appear pristine; simple data may appear degraded")
    print("=" * 90)
