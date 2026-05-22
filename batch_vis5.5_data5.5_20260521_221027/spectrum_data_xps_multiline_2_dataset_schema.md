# Dataset Schema for spectrum_data_xps_multiline_2.csv

This file documents the CSV layout produced by the synthetic spectrum generator.
Use this to map visual PNGs to the numeric columns programmatically.

## Column Definitions
- **energy**: float — X-axis value (units in `x_units` column or in the generator config).
- **intensity**: float — Measured intensity/counts at the corresponding energy/wavenumber.
- **line_id**: int|string — Identifier for the spectrum line (multiple lines may be present per file).
- **technique**: string — Spectroscopy technique (e.g., XPS, AES, EDS, EELS, IR, Raman).
- **material**: string — Material selection from the `PEAK_LIBRARY` used to inject peaks (or 'Unknown').
- **peak_metadata**: JSON string — Array of peak objects injected/detected for this `line_id`. Each object: `{position, amplitude, fwhm}`.

## Flattened Peak Columns (first 6 peaks)
- **peak_1_position**, **peak_1_amplitude**, **peak_1_fwhm**: floats — First (largest) peak for the line. Subsequent `peak_n_*` follow the same pattern up to 6.

## Notes
- Each row corresponds to a single point on a spectrum; to get per-spectrum metadata, group rows by `line_id`.
- `peak_metadata` is provided as a JSON string per-row but is identical for all rows sharing the same `line_id` (it describes the line, not the point).
- PNG visualizations produced by the generator contain only plot lines, axes, and grids (no textual annotations).

## Example JSON `peak_metadata`
```json
[{"position": 284.5, "amplitude": 1.0, "fwhm": 0.7}]
```