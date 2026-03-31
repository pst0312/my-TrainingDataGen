"""
spectrum_generator.py
=====================
Generate realistic synthetic spectra using physics-based peak models and
material libraries. Applies advanced line shapes (Lorentzian, Voigt),
realistic backgrounds (Shirley, Bremsstrahlung), and visual degradation.

This script:
  1. Selects a random spectroscopy technique and material
  2. Pulls characteristic peaks from PEAK_LIBRARY
  3. Generates synthetic spectrum with physical line shapes
  4. Applies realistic backgrounds and noise
  5. Renders plot with visual degradation (blur, downsampling, JPEG artifacts)
  6. Stores the data in a pandas DataFrame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special, ndimage
from PIL import Image
import io
import random
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


# ============================================================================
# BACKGROUND FUNCTIONS (Physics-Based Models)
# ============================================================================

def shirley_background(x, y, n_iter=10):
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
    bg = np.copy(y)
    y_min = np.min(y)
    x_min, x_max = x[0], x[-1]
    
    for _ in range(n_iter):
        integral = np.cumsum(bg)
        # Normalize integral to span [0, 1] across energy range
        integral = (integral - integral[0]) / (integral[-1] - integral[0] + 1e-12)
        bg_new = y_min + (np.max(y) - y_min) * integral
        # Smooth update to avoid oscillation
        bg = 0.5 * bg + 0.5 * bg_new
    
    return bg


def bremsstrahlung_background(x, x_min, intensity=1000.0, k=1.5):
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
    x_clipped = np.maximum(x, x_min + 0.01)  # avoid singularity
    return intensity / (x_clipped ** k)


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


# ============================================================================
# MAIN SPECTRUM GENERATION
# ============================================================================

def generate_synthetic_data(
    technique: str,
    config: dict,
    material: str = None,
    n_points: int = 2048,
    n_lines: int = 1,
    seed: int = None,
) -> dict:
    """
    Generate realistic synthetic spectra with physics-based line shapes
    and material-specific peak positions from PEAK_LIBRARY.
    
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
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary mapping line_id to (x, y) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_lines = max(1, min(n_lines, 5))
    
    # Generate x-axis
    x_lo, x_hi = config["x_range"]
    x = np.linspace(x_lo, x_hi, n_points)
    
    # Get noise parameters
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
    
    for line_id in range(1, n_lines + 1):
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
                intensity = peak_info["intensity"] * np.random.uniform(0.85, 1.15)
                fwhm = peak_info["fwhm"] * np.random.uniform(0.9, 1.1)
                
                # Use line shape from config
                line_shape = config.get("peak_shape", "Gaussian")
                
                if line_shape == "Gaussian":
                    y += gaussian(x, pos, fwhm, intensity)
                elif line_shape == "Lorentzian":
                    y += lorentzian(x, pos, fwhm, intensity)
                elif line_shape == "Voigt":
                    # Voigt: blend Gaussian (50%) and Lorentzian (50%)
                    fwhm_g = fwhm * 0.7
                    fwhm_l = fwhm * 0.3
                    y += voigt(x, pos, fwhm_g, fwhm_l, intensity)
                else:
                    y += gaussian(x, pos, fwhm, intensity)  # default
        else:
            # Fallback: random peak generation (for techniques without library)
            num_peaks = np.random.randint(2, 6)
            peak_positions = np.random.uniform(x_lo, x_hi, num_peaks)
            peak_heights = np.random.uniform(10, 100, num_peaks)
            peak_widths = np.random.uniform(5, 50, num_peaks)
            
            line_shape = config.get("peak_shape", "Gaussian")
            
            for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
                if line_shape == "Lorentzian":
                    y += lorentzian(x, pos, width, height)
                elif line_shape == "Voigt":
                    y += voigt(x, pos, width * 0.7, width * 0.3, height)
                else:
                    y += gaussian(x, pos, width, height)
        
        # ============================================================
        # BACKGROUND GENERATION (Physics-based)
        # ============================================================
        
        bg_type = config.get("background_type", "Linear")
        
        if bg_type == "Shirley":
            # Create temporary y with peaks for Shirley estimation
            temp_y = np.copy(y) + np.random.normal(0, 0.5, n_points)
            bg = shirley_background(x, temp_y, n_iter=5)
            y += bg
        elif bg_type == "Bremsstrahlung":
            # EDS-style background
            x_min = x_lo
            bg = bremsstrahlung_background(x, x_min, intensity=np.max(y) * 0.3, k=1.7)
            y += bg
        elif bg_type == "Polynomial" or "Polynomial" in bg_type:
            # IR / Raman polynomial baseline
            bg = polynomial_baseline(x, degree=np.random.randint(1, 3))
            bg = np.maximum(bg, 0)  # ensure non-negative
            bg_scaled = bg * np.max(y) * 0.2
            y += bg_scaled
        elif "Power" in bg_type:
            # Power-law background (AES, EELS)
            y += 50 * np.exp(-0.001 * (x - x_lo))
        else:
            # Linear baseline (default)
            y += np.linspace(5, 20, n_points)
        
        # ============================================================
        # NOISE ADDITION (Realistic detector & photon noise)
        # ============================================================
        
        # Gaussian detector noise (readout noise, amplifier noise)
        y += np.random.normal(0, gaussian_sigma * np.max(y) * 0.01, n_points)
        
        # Poisson shot noise (photon counting statistics)
        # Scale Poisson contribution based on technique sensitivity
        y = np.maximum(y, 0)  # ensure non-negative for Poisson
        y += np.random.poisson(poisson_lambda / 10000, n_points) / 100
        
        # Add slight vertical offset between lines (for multi-line display)
        y += (line_id - 1) * np.random.uniform(5, 20)
        
        spectra[line_id] = (x, y)
    
    return spectra


def create_dataframe(spectra: dict, technique: str) -> pd.DataFrame:
    """
    Create a pandas DataFrame from multiple spectrum lines.
    
    Parameters
    ----------
    spectra : dict
        Dictionary mapping line_id to (x, y) tuples
    technique : str
        Spectroscopy technique name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: energy, intensity, line_id, technique
    """
    records = []
    
    for line_id, (x, y) in spectra.items():
        for energy, intensity in zip(x, y):
            records.append({
                "energy": energy,
                "intensity": intensity,
                "line_id": line_id,
                "technique": technique,
            })
    
    df = pd.DataFrame(records)
    return df


def apply_visual_degradation(fig_path: str, low_res_config: dict, dpi: int = 100):
    """
    Apply visual degradation effects to rendered spectrum image:
    - Gaussian blur (instrument response simulation)
    - Downsampling (pixel density reduction)
    - JPEG compression (digital artifact simulation)
    - Scan lines (CRT/scanner row artifacts)
    
    Parameters
    ----------
    fig_path : str
        Path to PNG image file
    low_res_config : dict
        Configuration with keys: blur_sigma_px, downsample_factor, jpeg_quality, etc.
    dpi : int
        Original image DPI (used for scan line calculations)
    """
    if not low_res_config.get("enabled", True):
        return
    
    # Load image
    img = Image.open(fig_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # ====== BLUR ======
    blur_sigma = low_res_config.get("blur_sigma_px", 0.0)
    if blur_sigma > 0:
        for ch in range(3):
            img_array[:, :, ch] = ndimage.gaussian_filter(img_array[:, :, ch], sigma=blur_sigma)
    
    # ====== DOWNSAMPLING ======
    downsample_factor = low_res_config.get("downsample_factor", 1)
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
    
    # ====== SCAN LINES ======
    if low_res_config.get("add_scan_lines", False):
        spacing = low_res_config.get("scan_line_spacing", 4)
        alpha = low_res_config.get("scan_line_alpha", 0.1)
        h = img_array.shape[0]
        for i in range(0, h, spacing):
            img_array[i, :] = img_array[i, :] * (1 - alpha)  # darken scan line
    
    # ====== PAPER GRAIN (IR) ======
    if low_res_config.get("paper_grain", False):
        grain_sigma = low_res_config.get("paper_grain_sigma", 1.5)
        grain = np.random.normal(0, grain_sigma / 255.0, img_array.shape)
        img_array = np.clip(img_array + grain, 0, 1)
    
    # Convert back to uint8 and save as JPEG (if quality specified)
    img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img_out = Image.fromarray(img_array)
    
    jpeg_quality = low_res_config.get("jpeg_quality", None)
    if jpeg_quality is not None and jpeg_quality < 95:
        # Compress with JPEG to introduce artifacts
        img_out.save(fig_path, "JPEG", quality=int(jpeg_quality))
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
    
    # Plot each line
    for idx, (line_id, (x, y)) in enumerate(spectra.items()):
        color = color_palette[idx % len(color_palette)]
        linestyle = linestyle_palette[idx % len(linestyle_palette)]
        
        ax.plot(
            x,
            y,
            color=color,
            linewidth=vs.get("line_width", 1.5),
            linestyle=linestyle,
            label=f"Line {line_id}",
            alpha=0.85,
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
    ax.set_xlabel(f"{config.get('x_axis', 'Energy')} ({x_units})", fontsize=12)
    ax.set_ylabel(f"{y_label} ({y_units})", fontsize=12)
    ax.set_title(
        f"Synthetic {technique} Spectrum — {len(spectra)} Lines",
        fontsize=14,
        fontweight="bold",
    )
    
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
    

def save_spectrum_plot(fig, technique: str, index: int = None, low_res_config: dict = None) -> str:
    """
    Save spectrum plot with optional visual degradation.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    technique : str
        Spectroscopy technique name
    index : int, optional
        Index for filename
    low_res_config : dict, optional
        Low-resolution degradation config
    
    Returns
    -------
    str
        Path to saved file
    """
    if index is not None:
        png_filename = f"spectrum_{technique.lower()}_multiline_{index}.png"
    else:
        png_filename = f"spectrum_{technique.lower()}_multiline.png"
    
    fig.savefig(png_filename, dpi=100, bbox_inches="tight")
    print(f"✓ Plot saved (initial): {png_filename}")
    
    # Apply visual degradation
    if low_res_config and low_res_config.get("enabled", False):
        apply_visual_degradation(png_filename, low_res_config, dpi=100)
        print(f"✓ Visual degradation applied: {png_filename}")
    
    return png_filename


if __name__ == "__main__":
    import sys
    
    # Get optional index argument
    index = None
    if len(sys.argv) > 1:
        try:
            index = int(sys.argv[1])
        except Exception:
            index = None
    
    print("=" * 90)
    print("  Synthetic Spectrum Generator — Physics-Based Multi-Line Instance")
    print("=" * 90)
    
    # Select a random technique
    technique = random.choice(list(ESI_CONFIG.keys()))
    config = ESI_CONFIG[technique]
    style = PLOT_STYLE_CONFIG[technique]
    
    # Randomly select material from available options for this technique
    available_materials = [
        m for m in PEAK_LIBRARY
        if technique in PEAK_LIBRARY[m]
    ]
    material = random.choice(available_materials) if available_materials else None
    
    # Randomly select 1-5 lines to generate
    num_lines = np.random.randint(1, 6)
    
    print(f"\nSelected technique: {technique}")
    if material:
        print(f"Selected material: {material}")
    print(f"Number of lines to generate: {num_lines}")
    
    # Generate synthetic spectra (multiple lines, physics-based)
    print(f"Generating synthetic spectra...")
    spectra = generate_synthetic_data(
        technique=technique,
        config=config,
        material=material,
        n_points=2048,
        n_lines=num_lines,
        seed=42,  # For reproducibility
    )
    
    # Create DataFrame
    df = create_dataframe(spectra, technique)
    print(f"\nDataFrame created:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique lines: {df['line_id'].unique().tolist()}")
    print(f"\nFirst 15 rows:")
    print(df.head(15))
    print(f"\nDataFrame statistics (by line):")
    print(df.groupby("line_id")[["energy", "intensity"]].describe())
    
    # Save to CSV
    if index is not None:
        csv_filename = f"spectrum_data_{technique.lower()}_multiline_{index}.csv"
    else:
        csv_filename = f"spectrum_data_{technique.lower()}_multiline.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Data saved: {csv_filename}")
    
    # Display line summary
    print(f"\nLine Summary:")
    for line_id in sorted(df['line_id'].unique()):
        line_data = df[df['line_id'] == line_id]
        print(f"  Line {line_id}: {len(line_data)} points | "
              f"Intensity range: {line_data['intensity'].min():.2f} – {line_data['intensity'].max():.2f}")
    
    # Create visualization with physics-based rendering
    print(f"\nGenerating multi-line plot with visual degradation...")
    fig, ax = plot_spectrum(df, spectra, technique, config, style)
    
    # Save with visual degradation
    png_filename = save_spectrum_plot(
        fig,
        technique,
        index=index,
        low_res_config=style.get("low_res", {})
    )
    
    plt.close(fig)
    
    print("\n" + "=" * 90)
    print("  Generation complete! (Physics-based, material library, visual degradation)")
    print("=" * 90)
