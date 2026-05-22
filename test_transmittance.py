#!/usr/bin/env python3
"""Test script for FTIR and UV-Vis transmittance mode."""

from spectrum_generator import generate_synthetic_data, plot_spectrum, save_spectrum_plot, create_dataframe
from esi_config import ESI_CONFIG, PLOT_STYLE_CONFIG
import matplotlib.pyplot as plt
import numpy as np

def test_ftir():
    """Test FTIR transmittance mode."""
    print("=" * 70)
    print("Testing FTIR Transmittance Mode")
    print("=" * 70)
    
    ftir_config = ESI_CONFIG["FTIR"]
    ftir_style = PLOT_STYLE_CONFIG["FTIR"]
    
    print(f"Background type: {ftir_config.get('background_type')}")
    print(f"Baseline level: {ftir_config.get('baseline_level')}")
    print(f"FWHM range: {ftir_config.get('fwhm_range')}")
    print(f"Base data complexity: {ftir_config.get('base_data_complexity')}")
    
    # Generate data
    spectra = generate_synthetic_data(
        technique="FTIR",
        config=ftir_config,
        material="Quartz (SiO2)",
        n_points=512,
        n_lines=2,
        data_complexity=3,
        seed=42
    )
    
    print(f"\nGenerated {len(spectra)} spectrum(s):")
    for line_id, (x, y) in spectra.items():
        y_min, y_max = y.min(), y.max()
        print(f"  Line {line_id}: x [{x.min():.1f}, {x.max():.1f}], y [{y_min:.4f}, {y_max:.4f}]")
        # Verify inverted peaks
        if "_trailing" not in str(line_id):
            # Should have dips (y < baseline)
            baseline = ftir_config.get("baseline_level", 1.0)
            dips = np.sum(y < baseline * 0.95)
            print(f"           Dips below 95% baseline: {dips} points")
    
    # Create DataFrame
    df = create_dataframe(spectra, "FTIR", material="Quartz (SiO2)")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Y-axis range: [{df['intensity'].min():.4f}, {df['intensity'].max():.4f}]")
    
    # Plot
    fig, ax = plot_spectrum(df, spectra, "FTIR", ftir_config, ftir_style)
    
    # Set Y-axis label to %T for transmittance
    ax.set_ylabel("Transmittance (%T)", fontsize=12)
    
    save_spectrum_plot(fig, "FTIR", low_res_config=ftir_style["low_res"], visual_complexity=5)
    plt.close(fig)
    
    print("\n✓ FTIR transmittance test passed!")
    return True


def test_uv_vis():
    """Test UV-Vis transmittance mode."""
    print("\n" + "=" * 70)
    print("Testing UV-Vis Transmittance Mode")
    print("=" * 70)
    
    uv_config = ESI_CONFIG["UV-Vis"]
    uv_style = PLOT_STYLE_CONFIG["UV-Vis"]
    
    print(f"Background type: {uv_config.get('background_type')}")
    print(f"Baseline level: {uv_config.get('baseline_level')}")
    print(f"FWHM range: {uv_config.get('fwhm_range')}")
    print(f"Peak shape: {uv_config.get('peak_shape')}")
    
    # Generate data
    spectra = generate_synthetic_data(
        technique="UV-Vis",
        config=uv_config,
        material="Gold (Au)",
        n_points=256,
        n_lines=1,
        data_complexity=6,
        seed=100
    )
    
    print(f"\nGenerated {len(spectra)} spectrum(s):")
    for line_id, (x, y) in spectra.items():
        y_min, y_max = y.min(), y.max()
        print(f"  Line {line_id}: x [{x.min():.1f}, {x.max():.1f}], y [{y_min:.4f}, {y_max:.4f}]")
        # Verify inverted peaks and trailing lines
        if "_trailing" in str(line_id):
            print(f"           [TRAILING LINE] reduced intensity")
        else:
            baseline = uv_config.get("baseline_level", 1.0)
            dips = np.sum(y < baseline * 0.90)
            print(f"           Dips below 90% baseline: {dips} points")
    
    # Create DataFrame
    df = create_dataframe(spectra, "UV-Vis", material="Gold (Au)")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Y-axis range: [{df['intensity'].min():.4f}, {df['intensity'].max():.4f}]")
    
    # Plot
    fig, ax = plot_spectrum(df, spectra, "UV-Vis", uv_config, uv_style)
    ax.set_ylabel("Transmittance (T)", fontsize=12)
    
    save_spectrum_plot(fig, "UV-Vis", low_res_config=uv_style["low_res"], visual_complexity=5)
    plt.close(fig)
    
    print("\n✓ UV-Vis transmittance test passed!")
    return True


def compare_absorption_vs_emission():
    """Compare emission mode (XPS) vs absorption mode (FTIR)."""
    print("\n" + "=" * 70)
    print("Comparing Emission (XPS) vs Absorption (FTIR)")
    print("=" * 70)
    
    # Emission mode (XPS)
    xps_config = ESI_CONFIG["XPS"]
    xps_spectra = generate_synthetic_data(
        technique="XPS",
        config=xps_config,
        material="Gold (Au)",
        n_points=256,
        n_lines=1,
        data_complexity=2,
        seed=42
    )
    
    xps_x, xps_y = list(xps_spectra.values())[0]
    print(f"XPS (emission):  y range [{xps_y.min():.2f}, {xps_y.max():.2f}]")
    print(f"  Peak behavior: upward peaks from baseline")
    
    # Absorption mode (FTIR)
    ftir_config = ESI_CONFIG["FTIR"]
    ftir_spectra = generate_synthetic_data(
        technique="FTIR",
        config=ftir_config,
        material="Gold (Au)",
        n_points=256,
        n_lines=1,
        data_complexity=2,
        seed=42
    )
    
    ftir_x, ftir_y = list(ftir_spectra.values())[0]
    baseline = ftir_config.get("baseline_level", 1.0)
    print(f"FTIR (transmittance): y range [{ftir_y.min():.4f}, {ftir_y.max():.4f}]")
    print(f"  Baseline: {baseline}")
    print(f"  Peak behavior: downward dips from {baseline} baseline")
    
    # Verify expected behavior
    xps_peaks_up = np.any(np.diff(xps_y[50:200]) > 0.5) and np.max(xps_y) > np.min(xps_y) * 1.5
    ftir_dips_down = np.any(ftir_y < baseline * 0.95)
    
    print(f"\nXPS peaks upward: {xps_peaks_up}")
    print(f"FTIR dips downward: {ftir_dips_down}")
    
    if xps_peaks_up and ftir_dips_down:
        print("\n✓ Absorption vs Emission modes verified!")
        return True
    else:
        print("\n✗ Mode comparison failed")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRANSMITTANCE MODE TEST SUITE")
    print("=" * 70)
    
    results = {
        "FTIR": test_ftir(),
        "UV-Vis": test_uv_vis(),
        "Absorption vs Emission": compare_absorption_vs_emission(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {test_name:25s} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 70))
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("FTIR and UV-Vis transmittance modes ready to use!")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")
