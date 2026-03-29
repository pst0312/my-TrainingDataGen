"""
spectrum_generator.py
=====================
Generate a single synthetic spectrum instance and store data in a pandas DataFrame.

This script:
  1. Selects a random spectroscopy technique
  2. Generates synthetic spectrum data (x, y points)
  3. Stores the data in a pandas DataFrame
  4. Creates a basic visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SpectDict import ESI_CONFIG, PLOT_STYLE_CONFIG
import random


def generate_synthetic_data(
    technique: str,
    config: dict,
    n_points: int = 2048,
    n_lines: int = 1,
    seed: int = None,
) -> dict:
    """
    Generate multiple random synthetic spectrum lines.
    
    Parameters
    ----------
    technique : str
        Spectroscopy technique (e.g., "XPS", "AES", "EDS", "EELS")
    config : dict
        Configuration dictionary for the technique
    n_points : int
        Number of data points per spectrum
    n_lines : int
        Number of independent lines to generate (1-5)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with keys as line_ids (1, 2, ...) and values as (x, y) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_lines = max(1, min(n_lines, 5))  # Clamp to 1-5 range
    
    # Generate x-axis values (shared across all lines)
    x_lo, x_hi = config["x_range"]
    x = np.linspace(x_lo, x_hi, n_points)
    
    # Get noise parameters
    noise = config.get("noise_profile", {})
    gaussian_sigma = noise.get("gaussian_sigma", 0.1)
    poisson_lambda = noise.get("poisson_lambda", 1000)
    
    # Generate multiple independent lines
    spectra = {}
    
    for line_id in range(1, n_lines + 1):
        # Initialize y-axis (baseline background)
        y = np.zeros_like(x)
        
        # Generate random peaks for this line
        num_peaks = np.random.randint(2, 6)
        peak_positions = np.random.uniform(x_lo, x_hi, num_peaks)
        peak_heights = np.random.uniform(10, 100, num_peaks)
        peak_widths = np.random.uniform(5, 50, num_peaks)
        
        for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
            y += height * np.exp(-((x - pos) ** 2) / (2 * width ** 2))
        
        # Add background (simple power-law or linear)
        background_type = config.get("background_type", "Linear")
        if "Power" in background_type:
            y += 50 * np.exp(-0.001 * (x - x_lo))
        else:
            y += np.linspace(5, 20, n_points)
        
        # Add noise
        # Gaussian broadening
        y += np.random.normal(0, gaussian_sigma, n_points)
        
        # Poisson shot noise
        y = np.maximum(y, 0)  # Ensure non-negative for Poisson
        y += np.random.poisson(poisson_lambda / 10000, n_points) / 100
        
        # Add slight vertical offset to distinguish lines
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


def plot_spectrum(df: pd.DataFrame, spectra: dict, technique: str, config: dict, style: dict) -> None:
    """
    Plot multiple synthetic spectrum lines with distinct styling or same color.
    
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
        Plot style configuration
    same_color : bool
        If True, all lines use the same color and are differentiated by line style only
    """
    def _plot_spectrum_inner(same_color=False):
        vs = style["visual_style"]  # visual style
        fig, ax = plt.subplots(figsize=(12, 7))
        # Color palette for multiple lines (distinct and professional)
        color_palette = [
            vs.get("line_color", "#1A3A6B"),  # use config color as default
            "#C41E3A",  # crimson
            "#2E8B57",  # sea green
            "#FF8C00",  # dark orange
            "#663399",  # purple
        ]
        linestyle_palette = ["-", "--", "-.", ":", "-"]
        # Plot each line
        for idx, (line_id, (x, y)) in enumerate(spectra.items()):
            if same_color:
                color = vs.get("line_color", "#1A3A6B")
            else:
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
        # Configure axes
        x_units = config.get("x_units", "eV")
        y_label = config.get("y_axis", "Intensity")
        y_units = config.get("y_units", "a.u.")
        ax.set_xlabel(f"{config.get('x_axis', 'Energy')} ({x_units})", fontsize=12)
        ax.set_ylabel(f"{y_label} ({y_units})", fontsize=12)
        ax.set_title(f"Synthetic {technique} Spectrum — {len(spectra)} Lines", fontsize=14, fontweight="bold")
        # Configure grid
        if vs.get("grid_visible", True):
            ax.grid(True, linestyle=vs.get("grid_linestyle", "--"), alpha=vs.get("grid_alpha", 0.35))
        # Set background colors
        ax.set_facecolor(vs.get("background_color", "#FFFFFF"))
        fig.patch.set_facecolor(vs.get("figure_facecolor", "#F7F7F7"))
        # Add watermark
        wm = style.get("watermark", {})
        if wm.get("enabled", True):
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
        # Enhanced legend with line information
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
        plt.tight_layout()
        # Save and show
        filename = f"spectrum_{technique.lower()}_multiline.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        print(f"✓ Plot saved: {filename}")
        # plt.show()  # Commented out so script completes when run from IDE
    return _plot_spectrum_inner


if __name__ == "__main__":
    print("=" * 80)
    print("  Synthetic Spectrum Generator — Multi-Line Instance")
    print("=" * 80)
    # Select a random technique
    technique = random.choice(list(ESI_CONFIG.keys()))
    config = ESI_CONFIG[technique]
    style = PLOT_STYLE_CONFIG[technique]
    # Randomly select 1-5 lines to generate
    num_lines = np.random.randint(1, 6)
    # Randomize whether all lines are the same color or not
    SAME_COLOR = bool(np.random.randint(0, 2))
    print(f"\nSelected technique: {technique}")
    print(f"Number of lines to generate: {num_lines}")
    print(f"Same color for all lines: {SAME_COLOR}")
    # Generate synthetic spectra (multiple lines)
    print(f"Generating synthetic spectra...")
    spectra = generate_synthetic_data(
        technique=technique,
        config=config,
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
    csv_filename = f"spectrum_data_{technique.lower()}_multiline.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Data saved: {csv_filename}")
    # Display line summary
    print(f"\nLine Summary:")
    for line_id in sorted(df['line_id'].unique()):
        line_data = df[df['line_id'] == line_id]
        print(f"  Line {line_id}: {len(line_data)} points | "
              f"Intensity range: {line_data['intensity'].min():.2f} – {line_data['intensity'].max():.2f}")
    # Create visualization
    print(f"\nGenerating multi-line plot...")
    plot_func = plot_spectrum(df, spectra, technique, config, style)
    plot_func(same_color=SAME_COLOR)
    print("\n" + "=" * 80)
    print("  Generation complete!")
    print("=" * 80)
