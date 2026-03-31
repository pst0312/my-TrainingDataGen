# Quick Start: Agent Integration

## Import & Initialize

```python
from agent_tools import SpectroscopyEnvironment

env = SpectroscopyEnvironment(output_dir='my_batch')
```

## Generate One Sample

```python
result = env.generate_custom_sample(
    technique='XPS',           # XPS, EDS, AES, IR, Raman, EELS
    material='Gold (Au)',      # Any from material_library.py
    vis_complexity=5,          # 1-10: visual quality (1=pristine, 10=degraded)
    data_complexity=6,         # 1-10: spectral features (1=simple, 10=complex)
    seed=42                    # Random seed (same seed = identical output)
)

# Access outputs
csv_file = result['csv_path']          # e.g., 'my_batch/xps_gold_(au)_42.csv'
png_file = result['png_path']          # Rendered image
data_df = result['data']               # pandas DataFrame
sample_id = result['sample_id']        # 'xps_gold_(au)_42'
num_lines = result['num_lines']        # Primary spectral lines
trailing_lines = result['trailing_lines']  # Indices of secondary measurements
```

## Batch Generation

```python
samples = [
    {'technique': 'XPS', 'material': 'Silicon (Si)', 'vis_complexity': 2, 'data_complexity': 4, 'seed': 100},
    {'technique': 'XPS', 'material': 'Gold (Au)', 'vis_complexity': 8, 'data_complexity': 8, 'seed': 101},
    {'technique': 'EDS', 'material': 'Copper (Cu)', 'vis_complexity': 5, 'data_complexity': 2, 'seed': 102},
]

results = env.batch_generate(samples, verbose=True)
env.save_metadata()  # Creates 'my_batch/metadata.json'
```

## Reproducibility (Key Feature!)

```python
# Same parameters + seed = identical output
sample1 = env.generate_custom_sample(
    technique='XPS', material='Gold (Au)',
    vis_complexity=5, data_complexity=6, seed=999
)

sample2 = env.generate_custom_sample(
    technique='XPS', material='Gold (Au)',
    vis_complexity=5, data_complexity=6, seed=999
)

# Guaranteed: sample1['data'] == sample2['data']  (byte-identical CSV)
```

## Metadata Access

```python
# Load batch metadata
metadata = env.load_metadata()

# Access sample info
for sample in metadata['samples']:
    print(sample['sample_id'])        # 'xps_gold_(au)_42'
    print(sample['technique'])        # 'XPS'
    print(sample['material'])         # 'Gold (Au)'
    print(sample['vis_complexity'])   # 5
    print(sample['data_complexity'])  # 6
    print(sample['seed'])             # 42
    print(sample['num_lines'])        # 3
    print(sample['trailing_lines'])   # [1, 3]
```

## Decoupled Axes (Power Feature!)

### Same data, different visual quality
```python
pristine = env.generate_custom_sample(
    technique='XPS', material='Copper (Cu)',
    vis_complexity=1,  # Perfect image
    data_complexity=7, # Complex data
    seed=100
)

degraded = env.generate_custom_sample(
    technique='XPS', material='Copper (Cu)',
    vis_complexity=10,  # Heavily degraded
    data_complexity=7,  # Same complex data
    seed=100            # Same seed = same spectra
)

# Result: Same peak data, different image quality!
# pristine['data'] == degraded['data']  ✓
```

### Different data, same visual quality
```python
simple = env.generate_custom_sample(
    technique='XPS', material='Copper (Cu)',
    vis_complexity=5, data_complexity=1, seed=200
)

complex = env.generate_custom_sample(
    technique='XPS', material='Copper (Cu)',
    vis_complexity=5, data_complexity=9, seed=201
)

# Result: 1 vs 9 spectral lines, same visual quality
# simple['num_lines'] == 1
# complex['num_lines'] == 4-5
```

## Data Format

### CSV Columns
```
energy,intensity,line_id,technique,material
0.0,10.2,1,XPS,Gold (Au)
0.7,10.3,1,XPS,Gold (Au)
0.5,8.9,1_trailing,XPS,Gold (Au)
```

### Trailing Lines
Lines with `_trailing` suffix represent secondary measurements (cross-polarization, etc).
They share x-axis with parent line but have reduced intensity.

## Available Materials

**Metals**: Copper (Cu), Gold (Au), Silver (Ag), Platinum (Pt), Nickel (Ni), Titanium (Ti), Aluminum (Al), Iron (Fe)

**Semiconductors**: Silicon (Si), Germanium (Ge), Gallium Arsenide (GaAs), Silicon Carbide (SiC), Zinc Oxide (ZnO)

**Polymers**: PMMA, PTFE, Polystyrene

**Ceramics**: Quartz (SiO2), Aluminum Oxide (Al2O3), Molybdenum Disulfide (MoS2), Titanium Oxide (TiO2)

**Organics**: Carbon, Graphite, Graphene Oxide, Iron Oxide (Fe2O3), Sodium Azide (NaN3)

(Run `from material_library import PEAK_LIBRARY; print(list(PEAK_LIBRARY.keys()))` to see all 24+)

## Available Techniques

| Technique | X-Axis | Units | Base Complexity |
|-----------|--------|-------|-----------------|
| XPS | Binding Energy | eV | 4 |
| EDS | Energy | keV | 2 |
| AES | Kinetic Energy | eV | 5 |
| IR | Wavenumber | cm⁻¹ | 6 |
| Raman | Raman Shift | cm⁻¹ | 8 |
| EELS | Energy Loss | eV | 7 |

## Complexity Scoring

### Data Complexity (what's in the spectrum)
- **1-2**: Single isolated peak
- **3-4**: 1-2 lines, minimal overlap
- **5-6**: 2-3 lines, moderate overlap
- **7-8**: 3-4 lines, high overlap, maybe trailing
- **9-10**: 4-5 overlapping lines, definitely trailing lines

### Visual Complexity (how degraded is the image)
- **1**: Perfect (no blur, no noise, no artifacts)
- **5**: Moderate degradation
- **10**: Maximum (heavy blur, JPEG artifacts, scan lines)

## Common Use Cases

### Build training dataset with varied complexity
```python
for vis in range(1, 11):
    for data in range(1, 11):
        seed = vis * 100 + data
        result = env.generate_custom_sample(
            technique='XPS', material='Gold (Au)',
            vis_complexity=vis,
            data_complexity=data,
            seed=seed
        )
        # Train model on result['png_path']
```

### Test model robustness to degradation
```python
# Simple spectrum at different visual qualities
for vis in [1, 5, 10]:
    result = env.generate_custom_sample(
        technique='XPS', material='Silicon (Si)',
        vis_complexity=vis,
        data_complexity=2,
        seed=500
    )
    # Evaluate model on result['png_path']
    # Same data, different image quality = test robustness
```

### Regenerate exact sample for debugging
```python
# Agent failed on seed=1000
problematic = env.generate_custom_sample(
    technique='XPS', material='Platinum (Pt)',
    vis_complexity=9, data_complexity=9,
    seed=1000
)

# Later: regenerate with full ground truth
regenerated = env.generate_custom_sample(
    technique='XPS', material='Platinum (Pt)',
    vis_complexity=9, data_complexity=9,
    seed=1000
)

# Same data - can analyze why agent struggled
assert regenerated['csv_path'].read_text() == problematic['csv_path'].read_text()
```

## Statistics & Introspection

```python
# List all samples in batch
sample_ids = env.list_samples()

# Get specific sample
sample = env.get_sample('xps_gold_(au)_42')

# Compute statistics
stats = env.get_statistics()
# Returns: {
#     'vis_complexity_distribution': {1: 5, 2: 4, ...},
#     'data_complexity_distribution': {1: 3, 2: 5, ...},
#     'material_counts': {'Gold (Au)': 15, 'Silicon (Si)': 10, ...},
#     'technique_counts': {'XPS': 40, 'EDS': 20, ...}
# }
```

## Error Handling

```python
try:
    result = env.generate_custom_sample(
        technique='XPS',
        material='Unknown Material',  # Will fail
        vis_complexity=5,
        data_complexity=5,
        seed=42
    )
except ValueError as e:
    print(f"Invalid material: {e}")
```

## Performance Tips

1. **Batch Processing**: Use `batch_generate()` instead of loop of `generate_custom_sample()`
2. **Determinism**: Always set seed for reproducibility
3. **Metadata**: Load batch metadata once, not per sample
4. **Memory**: Results contain in-memory DataFrame - be mindful with very large batches

## Full Documentation

See `AGENT_INTERFACE.md` for:
- Complete method signatures
- Return value structures
- Metadata manifest format
- Advanced examples
- Troubleshooting guide

---

**Ready to integrate? Start with a simple batch:**

```python
from agent_tools import SpectroscopyEnvironment
env = SpectroscopyEnvironment(output_dir='test_batch')
result = env.generate_custom_sample(technique='XPS', material='Gold (Au)', seed=42)
print(f"Generated: {result['sample_id']}")
print(f"CSV: {result['csv_path']}")
print(f"Lines: {result['num_lines']}, Trailing: {result['num_trailing_lines']}")
```

All tests pass ✓. Your AI model can now generate and access deterministic spectroscopy data!
