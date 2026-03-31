# Agent Interface Documentation

The synthetic spectroscopy pipeline now supports direct AI agent interaction through the `SpectroscopyEnvironment` class in `agent_tools.py`. This document describes the interface and how to use it.

## Overview

The agent interface enables AI models to:
- Generate synthetic spectroscopy samples with precise control over data and visual complexity
- Retrieve deterministic output (same seed = identical spectrum)
- Access complete ground-truth metadata in JSON format
- Batch process multiple samples efficiently
- Compute statistics over a batch

## Core Class: `SpectroscopyEnvironment`

Located in `agent_tools.py`, this class provides the main interface for agents.

### Initialization

```python
from agent_tools import SpectroscopyEnvironment

env = SpectroscopyEnvironment(
    output_dir='my_batch',      # Directory to save outputs
    verbose=True                # Print progress info
)
```

### Primary Method: `generate_custom_sample()`

Generate a single spectrum with complete deterministic control.

```python
result = env.generate_custom_sample(
    technique='XPS',              # Spectroscopy technique (string)
    material='Gold (Au)',         # Material from library (string)
    vis_complexity=5,             # Visual complexity: 1-10 (int)
    data_complexity=6,            # Data complexity: 1-10 (int)
    seed=42                       # Random seed for determinism (int)
)
```

**Parameters:**
- `technique`: One of `'XPS'`, `'EDS'`, `'AES'`, `'IR'`, `'Raman'`, `'EELS'`
- `material`: Any material name from `PEAK_LIBRARY` in `material_library.py`
- `vis_complexity`: 1-10, controls image degradation (blur, noise, scan lines, JPEG artifacts)
- `data_complexity`: 1-10, controls spectral features (number of lines, widths, overlaps)
- `seed`: Integer seed for numpy/random reproducibility

**Return Value (Dictionary):**
```python
{
    'sample_id': 'xps_gold_(au)_42',        # Unique identifier
    'technique': 'XPS',
    'material': 'Gold (Au)',
    'vis_complexity': 5,
    'data_complexity': 6,
    'seed': 42,
    'csv_path': 'my_batch/xps_gold_(au)_42.csv',
    'png_path': 'my_batch/xps_gold_(au)_42.png',
    'num_lines': 3,                         # Primary spectral lines
    'trailing_lines': [1, 2],               # Indices of lines with cross-pol component
    'data': pandas.DataFrame,               # Full CSV data in memory
    'spectra': {                            # Ground truth spectral decomposition
        'x_values': numpy.array,
        'lines': [                          # List of primary lines
            {
                'x_position': 1241.5,
                'fwhm': 0.8,
                'intensity': 45.2,
                'peak_shape': 'Voigt'
            },
            # ... more lines
        ]
    }
}
```

### Determinism Guarantee

The `seed` parameter ensures absolute reproducibility:

```python
result1 = env.generate_custom_sample(..., seed=999)
result2 = env.generate_custom_sample(..., seed=999)
# result1['data'] == result2['data']  ✓ Byte-for-byte identical CSV
# result1['spectra'] == result2['spectra']  ✓ Identical peak data
```

This is critical for agent evaluation and debugging. With the same seed, you can:
- Regenerate a challenging sample to analyze agent performance
- Create reproducible test suites
- Implement contrastive learning (same data + vis, different vis + data)

### Batch Processing

Generate multiple samples efficiently:

```python
samples = [
    {
        'technique': 'XPS',
        'material': 'Silicon (Si)',
        'vis_complexity': 2,
        'data_complexity': 4,
        'seed': 100
    },
    {
        'technique': 'XPS',
        'material': 'Gold (Au)',
        'vis_complexity': 8,
        'data_complexity': 8,
        'seed': 101
    },
]

results = env.batch_generate(samples, verbose=True)
```

Returns a list of result dictionaries (same format as `generate_custom_sample()`).

### Metadata Management

Save all batch metadata to a JSON manifest:

```python
metadata_path = env.save_metadata()
# Creates: my_batch/metadata.json
```

**Metadata Structure:**
```json
{
    "batch_info": {
        "output_dir": "my_batch",
        "timestamp": "2026-03-31T15:49:18.223990",
        "num_samples": 3
    },
    "samples": [
        {
            "sample_id": "xps_gold_(au)_42",
            "technique": "XPS",
            "material": "Gold (Au)",
            "vis_complexity": 5,
            "data_complexity": 6,
            "base_data_complexity": 4,
            "num_lines": 3,
            "trailing_lines": [1, 3],
            "num_trailing_lines": 2,
            "seed": 42,
            "csv_path": "my_batch/xps_gold_(au)_42.csv",
            "png_path": "my_batch/xps_gold_(au)_42.png",
            "n_points": 2048,
            "timestamp": "2026-03-31T15:49:18.140644"
        }
    ]
}
```

Load existing metadata:

```python
metadata = env.load_metadata()
# Returns the parsed JSON dict
```

### Sample Retrieval

Retrieve specific samples from a batch:

```python
sample = env.get_sample('xps_gold_(au)_42')
# Returns the same result dict as generate_custom_sample()

sample_ids = env.list_samples()
# Returns: ['xps_silicon_(si)_100', 'xps_gold_(au)_101', 'eds_copper_(cu)_102']
```

### Statistics & Analysis

Compute complexity distributions and material statistics:

```python
stats = env.get_statistics()
# Returns dict with:
#   - vis_complexity_distribution: counts by complexity level
#   - data_complexity_distribution: counts by complexity level
#   - material_counts: frequency of each material
#   - technique_counts: frequency of each technique
```

## Two-Axis Complexity System

The system maintains **complete decoupling** between data complexity and visual complexity:

- **Data Complexity (1-10):** Controls spectroscopy features
  - 1: Single, isolated peak
  - 5: Multiple lines with some overlap
  - 10: Highly overlapped, ambiguous peaks
  
- **Visual Complexity (1-10):** Controls image degradation
  - 1: Perfect image (no blur, noise, or artifacts)
  - 5: Moderate blur, some noise
  - 10: Heavy blur, JPEG compression, scan lines

You can generate:
- Complex data with pristine visuals: `data_complexity=10, vis_complexity=1`
- Simple data with terrible visuals: `data_complexity=1, vis_complexity=10`
- Any combination

## Trailing Lines (Cross-Polarization)

High-complexity samples may include "trailing lines" — secondary spectral features representing cross-polarization or other secondary measurements. These appear in the metadata:

```python
result['trailing_lines']  # e.g., [1, 3] means lines 1 and 3 have trailing components
result['num_trailing_lines']  # e.g., 2
```

Trailing lines are automatically included in the CSV/PNG output and peak data.

## Example: Agent Training Loop

```python
from agent_tools import SpectroscopyEnvironment

# Create environment
env = SpectroscopyEnvironment(output_dir='agent_training', verbose=False)

# Generate diverse training set
for i in range(100):
    seed = 1000 + i
    data_comp = (seed % 10) + 1  # 1-10
    vis_comp = ((seed // 10) % 10) + 1  # 1-10
    
    result = env.generate_custom_sample(
        technique='XPS',
        material='Gold (Au)',
        vis_complexity=vis_comp,
        data_complexity=data_comp,
        seed=seed
    )
    
    # Agent does something with result['png_path'] and result['csv_path']
    # Agent can access ground truth via result['spectra']

# Save metadata for reproducibility
metadata_path = env.save_metadata()
print(f"Training batch saved to: {metadata_path}")
```

## Example: Evaluation with Determinism

```python
# During evaluation, regenerate the same sample
env = SpectroscopyEnvironment(output_dir='eval_batch')

# Agent was challenged with seed=500
original = env.generate_custom_sample(
    technique='XPS',
    material='Copper (Cu)',
    vis_complexity=7,
    data_complexity=8,
    seed=500
)

# Agent provided predictions; now regenerate for detailed analysis
regenerated = env.generate_custom_sample(
    technique='XPS',
    material='Copper (Cu)',
    vis_complexity=7,
    data_complexity=8,
    seed=500  # Same seed = identical output
)

# Compare agent predictions to ground truth
assert regenerated['spectra'] == original['spectra']
# Now verify agent detected all peaks, measured correct positions, etc.
```

## API Reference

### Methods

| Method | Purpose |
|--------|---------|
| `__init__(output_dir, verbose=False)` | Initialize environment |
| `generate_custom_sample(...)` | Generate one spectrum with seed control |
| `batch_generate(samples, verbose=False)` | Generate multiple spectra |
| `save_metadata()` | Write metadata.json to output_dir |
| `load_metadata()` | Load metadata.json from output_dir |
| `get_sample(sample_id)` | Retrieve result dict by ID |
| `list_samples()` | List all sample IDs in batch |
| `get_statistics()` | Compute batch statistics |

### Material Library

Available materials in `material_library.py`:
- **Metals**: Copper (Cu), Gold (Au), Silver (Ag), Platinum (Pt), Nickel (Ni), Titanium (Ti), Aluminum (Al), Iron (Fe)
- **Semiconductors**: Silicon (Si), Germanium (Ge), Gallium Arsenide (GaAs), Silicon Carbide (SiC), Zinc Oxide (ZnO)
- **Polymers**: PMMA, PTFE, Polystyrene
- **Ceramics/Minerals**: Quartz (SiO2), Aluminum Oxide (Al2O3), Molybdenum Disulfide (MoS2), Titanium Oxide (TiO2)
- **Organics/Salts**: Sodium Azide (NaN3), Carbon

### Spectroscopy Techniques

| Technique | X-Axis | X-Units | Base Data Complexity |
|-----------|--------|---------|----------------------|
| XPS | Binding Energy | eV | 4 |
| EDS | Energy | keV | 2 |
| AES | Kinetic Energy | eV | 5 |
| IR | Wavenumber | cm⁻¹ | 6 |
| Raman | Raman Shift | cm⁻¹ | 8 |
| EELS | Energy Loss | eV | 7 |

## File Structure

```
output_dir/
├── metadata.json                          # Batch manifest
├── {sample_id}.csv                        # Spectral data (x, intensity, baseline, ...)
└── {sample_id}.png                        # Rendered spectrum image
```

## Notes for Agent Developers

1. **Always Use Seeds:** Never rely on random sample generation. Always specify a seed for reproducibility.

2. **CSV Format:** The CSV files contain columns for x-axis position, intensity, baseline, and component information. Reference `spectrum_generator.py` for exact column names.

3. **Determinism Requirement:** Your system must be able to regenerate exact samples. Use the seed from metadata.json.

4. **Ground Truth Access:** Use the `'spectra'` key in the result dict to access peak positions, widths, and intensities directly (avoiding any image analysis on the PNG).

5. **Batch Organization:** Always use metadata.json to track what was generated. Never rely on filename parsing.

## Troubleshooting

**Q: Why is my seed not deterministic?**
- Ensure you're using the same `vis_complexity` and `data_complexity`. Visual degradation is seeded separately.

**Q: Can I use a material not in the library?**
- The environment will raise a clear error. Check `material_library.py` for available materials or add your own.

**Q: How do I add a new material?**
- Edit `material_library.py` and add an entry to `PEAK_LIBRARY`. Reference existing entries for structure.

**Q: How do I add support for a new spectroscopy technique?**
- Add configuration to `esi_config.py` in `ESI_CONFIG` and `PLOT_STYLE_CONFIG`. Then update `material_library.py` to add peaks for that technique.
