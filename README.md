# Synthetic Spectroscopy Data Generator

A comprehensive Python pipeline for generating highly realistic synthetic spectroscopy data with independent control over spectral complexity and visual degradation. Designed for robust machine learning model training across diverse measurement conditions.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Two-Axis Complexity System](#two-axis-complexity-system)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Agent Interface](#agent-interface)
7. [File Structure](#file-structure)
8. [Configuration & Material Library](#configuration--material-library)
9. [Advanced Features](#advanced-features)
10. [Technical Details](#technical-details)
11. [Examples & Use Cases](#examples--use-cases)

---

## Overview

This project generates synthetic electron spectroscopy data across multiple techniques (XPS, AES, EDS, EELS, IR, Raman) with physically accurate peak shapes, backgrounds, and material-specific signatures. The generator combines:

- **Physics-based peak synthesis** (Gaussian, Lorentzian, Voigt line shapes)
- **Realistic backgrounds** (Shirley for XPS, Bremsstrahlung for EDS)
- **Material library** (25+ realistic materials with characteristic peaks)
- **Two-axis complexity control** (independently controllable data vs. visual complexity)
- **Trailing lines** (secondary measurements simulating cross-polarization)
- **Visual degradation** (blur, downsampling, JPEG compression, scan lines)

Perfect for training deep learning models that must be robust to both spectral complexity and image quality variations.

---

## Key Features

### ✓ Physics-Based Peak Generation

- **Multiple line shapes**: Gaussian, Lorentzian, Voigt (via Faddeeva function)
- **Realistic backgrounds**: 
  - Shirley (XPS): Iterative integration background
  - Bremsstrahlung (EDS): Kramers' law X-ray background
  - Polynomial (IR/Raman): Baseline with polynomial coefficients
  - Power law (AES/EELS): Energy-dependent background

### ✓ Material Library (25+ Materials)

Organized into categories:
- **Noble & Transition Metals**: Au, Cu, Ag, Pt, Ni, Ti
- **Semiconductors**: Si, GaAs, SiC, ZnO, GaN
- **Oxides & Ceramics**: SiO2, Al2O3, TiO2, MoS2, WS2
- **Polymers**: PMMA, PTFE, Polystyrene, Polyethylene
- **Organics & Salts**: Graphite, Graphene Oxide, Fe2O3, NaN3

Each material includes characteristic peaks, FWHMs, and relative intensities for relevant spectroscopy techniques.

### ✓ Two-Axis Independent Complexity

**Data Complexity (1-10)**:
- Controls spectral properties: number of peaks, overlap, trailing lines
- Based on technique's inherent spectral complexity (EDS=2, Raman=8)
- Ranges from 1 (single isolated peak) to 10 (5 overlapping peaks + trailing lines)

**Visual Complexity (1-10)**:
- Controls post-rendering image degradation
- Ranges from 1 (pristine) to 10 (heavily degraded with artifacts)
- Independent from data complexity: same data at different visual qualities

### ✓ Trailing Lines

Simulate cross-polarized optical measurements or secondary correlated detections:
- Generated at high data complexity (≥6) with 40-70% probability
- Share exact peak positions with parent line but reduced intensity (0.3-0.7×)
- Rendered with dashed/dotted line style
- Physically accurate for polarization-dependent spectroscopy

### ✓ Visual Degradation Effects

Scaled by visual_complexity (1-10):
- **Gaussian blur**: 0→3.0 px sigma
- **Downsampling**: 1→3× with upsampling artifacts
- **JPEG compression**: Quality 95→40 with characteristic artifacts
- **Scan lines**: 0→0.1 alpha horizontal grid pattern
- **Paper grain**: 0→1.0 sigma texture

---

## Two-Axis Complexity System

### Why Decouple Data from Visual Complexity?

Traditional synthetic data generators conflate two independent concerns:
1. **Spectral content** (complexity of peaks, overlaps)
2. **Image quality** (blur, noise, artifacts)

This creates artificial correlations that don't exist in real data. For example:
- Complex spectra from high-quality instruments appear pristine
- Simple spectra from legacy instruments appear degraded

Our two-axis system enables:

```
Data Complexity 1 (simple) + Visual Complexity 1 (pristine)
→ Single peak, perfect image (modern simple measurement)

Data Complexity 9 (complex) + Visual Complexity 1 (pristine)
→ 5 overlapping peaks, perfect image (modern complex measurement)

Data Complexity 1 (simple) + Visual Complexity 10 (degraded)
→ Single peak, heavy artifacts (legacy simple measurement)

Data Complexity 9 (complex) + Visual Complexity 10 (degraded)
→ 5 overlapping peaks, heavy artifacts (legacy complex measurement)
```

### Data Complexity Scoring

| Score | Lines | Trailing | Overlap | Technique Examples |
|-------|-------|----------|---------|-------------------|
| 1-2 | 1 | No | None | EDS simple peak |
| 3-4 | 1-2 | No | Low | XPS isolated core level |
| 5-6 | 2-3 | Maybe | Moderate | AES, IR medium bands |
| 7-8 | 3-4 | Yes | High | EELS ionization edges |
| 9-10 | 4-5 | Yes | Very High | Raman polycrystal |

### Visual Complexity Scaling

All degradation parameters scale linearly:

```
degradation_scale = (visual_complexity - 1) / 9.0  ∈ [0, 1]
```

Examples:
- Complexity 1: degradation_scale = 0.0 → pristine (no blur, no JPEG)
- Complexity 5: degradation_scale = 0.44 → moderate degradation
- Complexity 10: degradation_scale = 1.0 → maximum degradation

---

## Installation & Setup

### Requirements

- Python 3.8+
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `pillow`

### Installation

```bash
# Clone or download the repository
cd /path/to/my-TrainingDataGen

# Install dependencies (if needed)
pip install numpy pandas matplotlib scipy pillow

# Verify imports work
python3 -c "from esi_config import ESI_CONFIG; from material_library import PEAK_LIBRARY; print('✓ All modules loaded')"
```

---

## Usage

### Quick Start: Batch Generation (Recommended)

Interactive batch generation with automatic folder organization:

```bash
python3 batch_generate.py
```

Prompts:
```
How many spectra would you like to generate? 
→ 50

Enter Visual Complexity range (1-10) [Press Enter for default 1,10]:
→ 2,8

Enter Data Complexity range (1-10) [Press Enter for default 1,10]:
→ 5,8
```

**Output**: Creates organized batch folder `batch_vis5.0_data6.5_20240331_123456/` with all PNG and CSV files.

### Command Line: Single Spectrum

Generate a single spectrum with specific complexity ranges:

```bash
# Complex data (9/10), pristine visuals (1/10)
python3 spectrum_generator.py 1 --min-data 9 --max-data 10 --min-vis 1 --max-vis 1

# Simple data (1/10), degraded visuals (10/10)
python3 spectrum_generator.py 2 --min-data 1 --max-data 2 --min-vis 9 --max-vis 10

# Balanced
python3 spectrum_generator.py 3 --min-data 5 --max-data 6 --min-vis 4 --max-vis 6
```

**Output**: 
- `spectrum_data_*_multiline_{index}.csv` — Data with technique and material columns
- `spectrum_*_multiline_{index}.png` — Rendered plot with visual degradation

### CSV Output Format

Generated CSV files include columns:

| Column | Description |
|--------|-------------|
| `energy` | X-axis value (eV, cm⁻¹, etc. depending on technique) |
| `intensity` | Y-axis intensity value |
| `line_id` | Peak identifier (1, 2, "1_trailing", "2_trailing", etc.) |
| `technique` | Spectroscopy technique (XPS, EDS, Raman, etc.) |
| `material` | Material name from library or "Unknown" |

Example:
```
energy,intensity,line_id,technique,material
0.0,10.2,1,XPS,Silicon (Si)
0.7,10.3,1,XPS,Silicon (Si)
0.5,8.9,1_trailing,XPS,Silicon (Si)
...
```

---

## Agent Interface

The pipeline supports direct AI agent interaction through the `SpectroscopyEnvironment` class in `agent_tools.py`. This enables programmatic sample generation with deterministic control and ground-truth metadata access.

### Quick Start

```python
from agent_tools import SpectroscopyEnvironment

# Create environment
env = SpectroscopyEnvironment(output_dir='my_batch', verbose=True)

# Generate a single spectrum with deterministic control
result = env.generate_custom_sample(
    technique='XPS',
    material='Gold (Au)',
    vis_complexity=5,      # Visual quality: 1 (pristine) to 10 (degraded)
    data_complexity=6,     # Spectral complexity: 1 (simple) to 10 (complex)
    seed=42                # Deterministic seed for reproducibility
)

# Result contains files, metadata, and ground truth
print(result['sample_id'])      # 'xps_gold_(au)_42'
print(result['csv_path'])       # 'my_batch/xps_gold_(au)_42.csv'
print(result['png_path'])       # 'my_batch/xps_gold_(au)_42.png'
print(result['spectra'])        # Ground truth peak data

# Batch generation
samples = [
    {'technique': 'XPS', 'material': 'Silicon (Si)', 'vis_complexity': 2, 'data_complexity': 4, 'seed': 100},
    {'technique': 'XPS', 'material': 'Gold (Au)', 'vis_complexity': 8, 'data_complexity': 8, 'seed': 101},
]
results = env.batch_generate(samples, verbose=True)

# Save metadata manifest
metadata_path = env.save_metadata()
# Creates: my_batch/metadata.json with all sample information
```

### Key Features

1. **Deterministic Control**: Same seed → identical output. Perfect for reproducible evaluation.
2. **Ground Truth Access**: Every sample returns complete metadata including peak positions, widths, and intensities.
3. **Decoupled Axes**: Data complexity and visual complexity are completely independent.
4. **Metadata Manifest**: JSON file tracks all generated samples with their exact parameters and random seeds.
5. **Batch Processing**: Generate multiple samples efficiently with metadata aggregation.

### Return Value

Each `generate_custom_sample()` call returns a dictionary with:

```python
{
    'sample_id': str,              # Unique identifier
    'technique': str,              # 'XPS', 'EDS', etc.
    'material': str,               # Material name from library
    'vis_complexity': int,         # 1-10
    'data_complexity': int,        # 1-10
    'seed': int,                   # Random seed used
    'csv_path': str,               # Path to data file
    'png_path': str,               # Path to rendered image
    'num_lines': int,              # Primary spectral lines
    'trailing_lines': list,        # Indices of secondary measurements
    'data': pd.DataFrame,          # Full CSV data in memory
    'spectra': dict,               # Ground truth decomposition
}
```

### Complete Documentation

See **[AGENT_INTERFACE.md](AGENT_INTERFACE.md)** for comprehensive API documentation, including:
- Full method signatures and parameters
- Metadata manifest structure
- Example training and evaluation loops
- Material and technique references
- Troubleshooting guide

---

## File Structure

```
my-TrainingDataGen/
├── esi_config.py                    # Core configuration (ESI_CONFIG, PLOT_STYLE_CONFIG)
├── material_library.py              # Material library (25+ materials with peaks)
├── spectrum_generator.py            # Main generation engine
├── batch_generate.py                # User-facing batch CLI interface
├── agent_tools.py                   # Agent-ready programmatic interface (NEW)
├── README.md                        # This file
├── AGENT_INTERFACE.md               # Complete agent API documentation (NEW)
├── TWO_AXIS_COMPLEXITY.md           # Detailed complexity system documentation
├── IMPLEMENTATION_NOTES.md          # Technical implementation details
└── batch_vis5.0_data6.5_20240331_123456/  # Generated batch folder
    ├── spectrum_data_*.csv
    ├── spectrum_*.png
    ├── metadata.json                # Manifest with all sample metadata (NEW)
    └── generated_csv_files.txt
```

---

## Configuration & Material Library

### esi_config.py

Defines spectroscopy techniques and their parameters:

```python
ESI_CONFIG: Dict[str, SpecConfig] = {
    "XPS": {
        "x_axis": "Binding Energy",
        "x_units": "eV",
        "x_range": (0.0, 1400.0),
        "base_data_complexity": 4,
        "noise_profile": {...},
        ...
    },
    # ... (AES, EDS, EELS, IR, Raman)
}

PLOT_STYLE_CONFIG: Dict[str, PlotStyle] = {
    "XPS": {
        "visual_style": {"color": "#1f77b4", ...},
        "low_res": {...},
        ...
    },
    # ... (for each technique)
}
```

### material_library.py

Contains 25+ materials with realistic peaks:

```python
PEAK_LIBRARY: Dict[str, Dict[str, Any]] = {
    "Silicon (Si)": {
        "description": "Semiconductor; Si 2p doublet in XPS...",
        "XPS": {
            "peaks": [
                {"position": 99.3, "intensity": 0.67, "fwhm": 0.85, ...},
                ...
            ]
        },
        "EDS": {...},
        "Raman": {...},
    },
    # ... (24+ more materials)
}
```

**Key functions**:
- `get_material_list()` — List all available materials
- `get_material_info(name)` — Retrieve material definition
- `get_techniques_for_material(name)` — List supported techniques

---

## Advanced Features

### Trailing Lines

Generate secondary correlated measurements for high data complexity:

```python
# Generated when data_complexity ≥ 6
# Probability: 40% (complexity 6) → 70% (complexity 10)

# In generated DataFrame:
line_id values: [1, 2, "1_trailing", "2_trailing", ...]
# Trailing lines have same peaks as parent, intensity scaled 0.3-0.7×
```

**Physics**: Simulates cross-polarized optical spectroscopy or dual-detector measurements where secondary signal is correlated but attenuated.

### Realistic Backgrounds

Different background models for each technique:

```python
# XPS: Shirley background (iterative integration)
background = shirley_background(x, intensity)

# EDS: Bremsstrahlung (Kramers' law)
background = bremsstrahlung_background(x)

# IR/Raman: Polynomial baseline
background = polynomial_background(x, degree=2)

# AES/EELS: Power law
background = power_law_background(x)
```

### Physics-Based Line Shapes

```python
# Gaussian: symmetric, fast computation
y = gaussian_peak(x, position, fwhm, intensity)

# Lorentzian: broader tails, natural line shape
y = lorentzian_peak(x, position, fwhm, intensity)

# Voigt: convolution of Gaussian and Lorentzian (most realistic)
y = voigt_peak(x, position, fwhm_g, fwhm_l, intensity)
```

---

## Technical Details

### Complexity Filtering

When generating data:
1. Generate random target_data_complexity in provided range
2. Filter techniques where `|base_data_complexity - target| ≤ 2`
3. Select random technique from suitable list
4. Generate peaks based on target_data_complexity

This ensures realistic technique selection — you won't get simple EDS data with 5 overlapping peaks.

### Degradation Scaling Formula

```python
degradation_scale = (visual_complexity - 1) / 9.0

# Scaled parameters:
blur_sigma = base_sigma * degradation_scale  # 0 → 3.0 px
downsample = 1 + (base_downsample - 1) * degradation_scale  # 1 → 3×
jpeg_quality = 95 - (95 - base_quality) * degradation_scale  # 95 → 40
scan_line_alpha = base_alpha * degradation_scale  # 0 → 0.1
```

### Performance Characteristics

- **Single spectrum**: ~0.5-1.0 seconds (depends on n_lines and visual degradation)
- **Batch of 50 spectra**: ~30-50 seconds
- **Memory usage**: ~50 MB per batch

---

## Examples & Use Cases

### Use Case 1: Training Dataset with Full Diversity

Generate spectra covering all combinations of complexity:

```bash
python3 batch_generate.py
# Inputs:
#   Spectra: 500
#   Visual range: 1,10
#   Data range: 1,10
```

Result: 500 spectra with all possible data/visual complexity combinations for robust ML training.

### Use Case 2: Instrument-Specific Training

Simulate data from different instrument generations:

```bash
# Modern instrument: good resolution (visual complexity 1-3), any data complexity
python3 batch_generate.py
# Inputs:
#   Spectra: 100
#   Visual range: 1,3
#   Data range: 1,10

# Legacy instrument: poor resolution (visual complexity 7-10), any data complexity
python3 batch_generate.py
# Inputs:
#   Spectra: 100
#   Visual range: 7,10
#   Data range: 1,10
```

### Use Case 3: Material-Specific Training

The material column in CSV enables filtering:

```bash
# Generate 50 spectra, keep only those with Silicon (Si)
python3 batch_generate.py
# Then filter CSV files where material == "Silicon (Si)"
```

### Use Case 4: Cross-Validation Study

Compare model robustness across different complexity axes:

```python
import pandas as pd

# Train on complex data (8-10), test on simple data (1-2)
train_files = glob.glob("batch_vis*_data8*/*.csv")
test_files = glob.glob("batch_vis*_data1*/*.csv")

# Train on pristine visuals (1-2), test on degraded (9-10)
train_files = glob.glob("batch_vis1*/*.csv")
test_files = glob.glob("batch_vis10*/*.csv")
```

---

## Contributing & Extending

### Adding New Materials

Edit `material_library.py`:

```python
PEAK_LIBRARY["New Material (Symbol)"] = {
    "description": "Brief description...",
    "XPS": {
        "peaks": [
            {"position": X.X, "intensity": 0.XX, "fwhm": X.X, "type": "Label"},
            ...
        ]
    },
    # Add more techniques as needed
}
```

### Adding New Techniques

Edit `esi_config.py`:

```python
ESI_CONFIG["NewTech"] = {
    "x_axis": "...",
    "x_units": "...",
    "x_range": (min, max),
    "base_data_complexity": 5,
    "noise_profile": {...},
    ...
}

PLOT_STYLE_CONFIG["NewTech"] = {
    "visual_style": {"color": "#...", ...},
    "low_res": {...},
    ...
}
```

---

## References & Literature

- **XPS**: Probing surface composition and bonding (Al Kα: 1486.6 eV)
- **AES**: Auger electron spectroscopy with KLL, LMM transitions
- **EDS**: Energy dispersive X-ray spectroscopy with characteristic K, L, M lines
- **EELS**: Electron energy loss spectroscopy with ionization edges and fine structure
- **IR**: Infrared absorption from 400-4000 cm⁻¹ vibrational modes
- **Raman**: Raman scattering with stokes/anti-stokes lines and overtones

Material peak data sourced from:
- NIST X-ray photoelectron spectroscopy database
- Atomic Spectra Database (physics.nist.gov)
- Published spectroscopy literature (Moulder, Wagner, et al.)

---

## License & Citation

MIT License. If you use this generator in research, please cite:

```bibtex
@software{synthetic_spectroscopy_2024,
  title={Synthetic Spectroscopy Data Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/my-TrainingDataGen}
}
```

---

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'esi_config'
```

**Solution**: Ensure all files (`esi_config.py`, `material_library.py`, `spectrum_generator.py`) are in the same directory.

### Memory Usage High

```
MemoryError on batch with 1000 spectra
```

**Solution**: Reduce batch size. Process in smaller batches (100-200 spectra per run).

### CSV Missing Material Column

```
CSV has only energy, intensity, line_id columns
```

**Solution**: Update `spectrum_generator.py` to latest version. Material column is added by `create_dataframe()`.

---

## Support & Questions

For issues, questions, or feature requests, please refer to the technical documentation in `TWO_AXIS_COMPLEXITY.md` and `IMPLEMENTATION_NOTES.md`.

**Happy spectroscopy data generation!** 🧪⚛️
