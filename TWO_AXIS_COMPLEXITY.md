# Two-Axis Complexity System

## Overview

The synthetic spectroscopy generator now implements **complete decoupling** between **data complexity** and **visual complexity**. This enables the generation of diverse, realistic training datasets where spectral properties are independent from image quality degradation.

### Key Concept

- **Data Complexity** (1-10): The inherent complexity of the spectroscopic data number of peaks, overlapping lines, trailing lines from cross-polarized measurementsitself
- **Visual Complexity** (1-10): The post-rendering image blur, JPEG compression, downsampling, scan lines, graindegradation

A spectrum can have:
- Complex data with pristine visuals (9/10 data, 1/10 visual)
- Simple data with heavy degradation (1/10 data, 10/10 visual)
- Any combination in between

This is critical for training machine learning models that must be robust to both spectral complexity and image quality variations.

---

## Data Complexity (1-10)

### Scoring by Technique

Each spectroscopy technique has a **base_data_complexity** score reflecting its inherent spectral overlap:

| Technique | Base Score | Rationale |
|-----------|-----------|-----------|
| EDS | 2 | Elemental X-ray lines are well-separated |
| XPS | 4 | Core levels, Auger lines, can overlap moderately |
| AES | 5 | Auger energies overlap more frequently |
| IR | 6 | Vibrational modes across functional groups |
| EELS | 7 | Ionization edges overlap significantly |
| Raman | 8 | Multiple overlapping vibrational modes |

### What Drives Data Complexity

1. **Number of lines** (`n_lines`)
   - Complexity 1-3: 1 line
   - Complexity 4-5: 1-2 lines
   - Complexity 6-7: 2-3 lines
   - Complexity 8-10: 3-5 lines

2. **Peak overlap** (from base technique characteristics)
   - Techniques with higher base scores naturally have more overlapping peaks

3. **Trailing lines** (simulating cross-polarized measurements)
   - Generated when data_ 6 with probability up to 70%complexity 
   - Share exact x-axis positions and peak positions with primary line
   - Reduced intensity (0.3-0.7x of parent line)
   - Rendered with dashed/dotted line style to distinguish from primary

### Example

```python
# Generate 5 Raman lines with high data complexity
target_data_complexity = 9  # 9/10
# Result: 4-5 primary lines, likely trailing lines, high peak overlap

# Generate 1 EDS line with low data complexity
target_data_complexity = 1  # 1/10
# Result: 1 simple line, no trailing lines, pristine peaks
```

---

## Visual Complexity (1-10)

### Degradation Parameters

Visual complexity scales all image degradation post-rendering:

| Parameter | Complexity 1 | Complexity 5 | Complexity 10 | Formula |
|-----------|-------------|-------------|---------------|---------|
| blur_sigma_px | 0.0 | 1.5 | 3.0 | `0 + 3.0  scale` |
| downsample_factor | 1.0 | 1.5 | 3.0 | `1 + 2.0  scale` |
| JPEG quality | 95 | 67 | 40 | `95 - 55  scale` |
| scan_line_alpha | 0.0 | 0.05 | 0.1 | `0.1  scale` |
| paper_grain_sigma | 0.0 | 0.5 | 1.0 | `1.0  scale` |

Where `scale = (visual_complexity -  [0, 1]0` 

### Examples

**Visual Complexity 1** (pristine):
- No blur
- No downsampling
- No JPEG artifacts
- Clean, high-resolution image

**Visual Complexity 5** (moderate degradation):
- Slight blur (1.5 px Gaussian)
- Minor downsampling (1.5)
- Moderate JPEG compression
- Moderate scan line visibility

**Visual Complexity 10** (heavily degraded):
- Heavy blur (3.0 px)
- Significant downsampling (3)
- Heavy JPEG compression (quality 40)
- Prominent scan lines
- Paper grain texture

---

## Usage

### Command Line: `spectrum_generator.py`

```bash
python3 spectrum_generator.py <index> [--min-vis MIN] [--max-vis MAX] [--min-data MIN] [--max-data MAX]
```

**Arguments:**

- `index`: Output file index (for naming: `spectrum_XYZ_index.png`)
- `--min-vis`, `--max-vis`: Visual complexity range (default: 1, 10)
- `--min-data`, `--max-data`: Data complexity range (default: 1, 10)

**Example:**

```bash
# Complex data, pristine visuals
python3 spectrum_generator.py 1 --min-data 8 --max-data 10 --min-vis 1 --max-vis 1

# Simple data, degraded visuals (like historical analog data)
python3 spectrum_generator.py 2 --min-data 1 --max-data 3 --min-vis 8 --max-vis 10

# Balanced complexity
python3 spectrum_generator.py 3 --min-data 4 --max-data 6 --min-vis 4 --max-vis 6
```

### Batch Generation: `batch_generate.py`

Interactive prompt-based generation of multiple spectra:

```bash
python3 batch_generate.py
```

**Prompts:**

```
How many spectra would you like to generate? 
 10

Enter Visual Complexity range (1-10) [Press Enter for default 1,10]:
 3,7

Enter Data Complexity range (1-10) [Press Enter for default 1,10]:
 5,8
```

This generates 10 spectra with:
- Visual complexity randomly sampled from [3, 7]
- Data complexity randomly sampled from [5, 8]

---

## Implementation Details

### Trailing Line Generation

Trailing lines simulate cross-polarized optical measurements or secondary correlated detections.

**When generated:**
- Data  6complexity 
- Probability: `0.4 + 0.3  (data_complexity - 6) / 4`
  - Complexity 6: 40% probability
  - Complexity 7: 47% probability
  - Complexity 8: 55% probability
  - Complexity 9: 62% probability
  - Complexity 10: 70% probability

**Properties:**
- **Same x-axis** as parent line
- **Same peak positions** as parent line
- **Reduced intensity**: multiplied by factor in [0.3, 0.7]
- **Reduced background**: 50% of parent background
- **Visual style**: dashed/dotted line (technique-specific)
- **Legend**: "1_trailing", "2_trailing", etc.

**Physics accuracy:**
- Cross-polarized measurements in optical spectroscopy do share the same spectral features (peaks)
- Intensity reduction reflects the reduced signal from orthogonal polarization orientation
- Line style distinguishes secondary measurements from primary

### Visual Degradation Scaling

All degradation parameters are scaled by:
```python
degradation_scale = (visual_complexity - 1) / 9.0
```

This maps complexity 1-10 to scale 0.0-1.0, enabling smooth progression from pristine to heavily degraded.

**Implementation:**
```python
def apply_visual_degradation(image_array, low_res_config, visual_complexity):
    degradation_scale = (visual_complexity - 1) / 9.0
    
    # Scale blur sigma
    blur_sigma = low_res_config.get('blur_sigma_px', 3.0) * degradation_scale
    if blur_sigma > 0:
        image_array = scipy.ndimage.gaussian_filter(image_array, sigma=blur_sigma)
    
    # Scale downsampling
    downsample_factor = 1 + (low_res_config.get('downsample_factor', 3.0) - 1) * degradation_scale
    
    # Scale JPEG quality
    base_quality = low_res_config.get('jpeg_quality', 40)
    jpeg_quality = 95 - (95 - base_quality) * degradation_scale
    
    # ... apply scan lines, grain, etc. scaled by degradation_scale
```

---

## Training Dataset Diversity

### Recommended Generation Strategies

#### Strategy 1: Pristine Complex Data (Model training)
```bash
# Generate complex spectra with perfect visual clarity
--min-data 7 --max-data 10 --min-vis 1 --max-vis 1
```
Use for training the model on spectral patterns without image quality confounds.

#### Strategy 2: Degraded Simple Data (Edge cases)
```bash
# Generate simple spectra with heavy degradation (like historical instruments)
--min-data 1 --max-data 3 --min-vis 8 --max-vis 10
```
Use for robustness testing and historical data compatibility.

#### Strategy 3: Realistic Variation (Balanced)
```bash
# Generate realistic variation in both axes
--min-data 3 --max-data 7 --min-vis 3 --max-vis 7
```
Use for general training with diverse conditions.

#### Strategy 4: Cross-axis Coverage (Full range)
```bash
# Generate full coverage across both complexity axes
--min-data 1 --max-data 10 --min-vis 1 --max-vis 10
```
Use for comprehensive robustness testing and transfer learning.

---

## Technical Architecture

### File Structure

| File | Role |
|------|------|
| `SpectDict.py` | Configuration registry with `base_data_complexity` and `trailing_line_style` |
| `spectrum_generator.py` | Main generation engine with argparse, complexity filtering, trailing line generation |
| `batch_generate.py` | User-facing batch generation interface |

### Key Modifications

**SpectDict.py:**
- Added `base_data_complexity: int` (1-10) to each technique in `ESI_CONFIG`
- Added `trailing_line_style: str` ("--", ":", "-.", etc.) to each `visual_style` dict

**spectrum_generator.py:**
- `generate_synthetic_data(data_complexity)`: Scales `n_lines` and trailing line probability
- `apply_visual_degradation(image_array, low_res_config, visual_complexity)`: Scales all degradation parameters
- `main()`: Added argparse for `--min-vis`, `--max-vis`, `--min-data`, `--max-data`
- Technique filtering: Selects techniques             of `target_data_complexity`within 

**batch_generate.py:**
- `parse_complexity_range(input_str)`: Parses "min,max" format with validation
- Main loop: Prompts for complexity ranges, passes to `spectrum_generator.py`

---

## Validation & Testing

All functionality has been tested:

 Trailing line generation with correct intensity scaling
 Visual degradation scaling from complexity 1 (pristine) to 10 (heavily degraded)
 Argparse argument parsing with range specification
 Batch generation with complexity prompts and defaults
 Mixed line ID handling (int + string keys)
 Two-axis decoupling independence (confirmed with test cases)

---

## Future Extensions

1. **Quantitative complexity metrics**: Calculate actual spectral complexity (peak overlap, FWHM distribution) and correlate with complexity scores

2. **Multi-technique blending**: Generate "mixed" spectra combining features from multiple techniques

3. **Realistic physical variation**: Model instrument-specific noise profiles, field-dependent effects

4. **Interactive visualization**: Side-by-side comparison of same data at different visual complexity levels

---

## References

- **Two-Axis Complexity**: Independent control of data and visual properties
- **Trailing Lines**: Simulate cross-polarized optical measurements
- **Degradation Scaling**: Linear scaling of post-processing severity
- **Physics Basis**: All line shapes, backgrounds, and material peaks from literature

For physics details, see comments in `spectrum_generator.py` for:
- Lorentzian/Voigt line shapes
- Shirley/Bremsstrahlung backgrounds
- PEAK_LIBRARY material definitions
