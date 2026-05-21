# Source Code

Python modules for seismic signal processing, phase detection, statistical analysis, and visualization.

---

## Module Structure

```
src/
├── preprocessing/          # Data loading and cleaning
│   ├── io.py
│   ├── cleaning_metadata.py
│   └── cleaning_signals.py
│
├── segmentation/          # Phase detection and windowing
│   ├── search_windows.py
│   ├── ar_pick.py
│   ├── phasenet_utils.py
│   ├── window_segmentation.py
│   └── window_validation.py
│
├── analysis/              # Statistical analysis
│   ├── signals_pdf.py
│   ├── signals_scaling_temporal.py
│   └── signals_scaling_spatial_ensemble.py
│
├── visualization/         # Plotting functions
│   ├── plot_settings.py
│   ├── plots_metadata.py
│   ├── plots_signals.py
│   ├── plots_segmentation.py
│   └── plot_moment_scaling.py
│
└── utils/                 # Export utilities
    └── latex_export.py
```

---

## Module Overview

### Preprocessing
Load raw data from ITACA .ASC archive and apply cleaning pipelines for metadata and signals.

### Segmentation
Detect seismic phase onsets (P-wave, S-wave, coda) using AR-AIC or PhaseNet, compute adaptive search windows based on CRUST1.0 velocity model, segment signals into four temporal windows, and validate detection quality.

### Analysis
Characterize probability distributions (Gaussian fit, heavy tails, power-law exponents), compute moment scaling exponents ζ(q) for anomalous diffusion analysis (both time-averaged and spatial ensemble methods), and perform sensitivity analysis for phase pick uncertainty.

### Visualization
Generate publication-quality figures following project-wide style conventions, including metadata exploration, signal preprocessing validation, phase detection diagnostics, and moment scaling curves.

### Utilities
Export analysis results to LaTeX table format for thesis integration.

---

## Quick Reference

### Import Patterns

```python
# Preprocessing
from src.preprocessing import build_dataframes, clean_metadata, preprocess_signals

# Segmentation
from src.segmentation import (
    add_crustal_velocities,
    detect_onsets_arpick,
    segment_all_signals,
    quality_control_all_stations
)

# Analysis
from src.analysis import (
    gaussian_fit_analysis,
    heavy_tail_assessment,
    analyze_all_windows,
    save_results_parquet
)

# Visualization
from src.visualization import (
    set_plot_style,
    plot_station_map,
    plot_single_signal_with_windows,
    plot_scaling_exponents
)
```

---

## Example Workflows

### Phase Detection and Windowing

```python
# Load and preprocess
df_meta, df_acc = build_dataframes('../data/raw/query.zip')
df_meta = clean_metadata(df_meta)
df_acc = preprocess_signals(df_acc, normalize=False)

# Build signals dictionary
signals_dict = {}
for file in df_acc['file'].unique():
    signal = df_acc[df_acc['file'] == file]['acceleration'].values
    signals_dict[file] = signal

# Add theoretical arrivals
df_stations = add_crustal_velocities(df_meta, hypo_depth_km=10.4)
df_stations = add_theoretical_arrivals(df_stations, hypo_depth_km=10.4)

# Detect onsets
df_stations = detect_onsets_arpick(signals_dict, df_stations)
df_full = add_coda_onsets_to_dataframe(df_stations, signals_dict)

# Segment into windows
windowed = segment_all_signals(signals_dict, df_full, coda_method='rautian')

# Quality control
qc = quality_control_all_stations(windowed, df_full, df_stations,
                                   peak_column='PGA_CM/S^2',
                                   time_peak_column='TIME_PGA_S')
```

---

### Moment Scaling Analysis

```python
# Spatial ensemble averaging
results = analyze_all_windows(
    windowed,
    tau_min=0.01,
    tau_max_fraction=0.5,
    q_values=np.arange(0.5, 5.5, 0.25),
    sampling_rate=200.0
)

# Save and visualize
save_results_parquet(results, output_dir='../data/processed/ensemble_spatial')
plot_scaling_curves(results, output_dir='../figures/scaling')
plot_scaling_exponents(results, output_dir='../figures/scaling')
```

---

### PDF Analysis

```python
# Preprocess with normalization
df_clean = preprocess_signals(df_acc, normalize=True)

# Gaussian fit analysis
df_gaussian = gaussian_fit_analysis(
    df_clean,
    signal_column='acceleration',
    normalized=True,
    output_dir='../figures/pdf_analysis/gaussian'
)

# Heavy tail assessment
df_tail = heavy_tail_assessment(
    df_clean,
    signal_column='acceleration',
    normalized=True,
    output_dir='../figures/pdf_analysis/heavy_tail'
)
```

---

## Documentation Standards

All modules follow consistent documentation conventions:

**Module-level docstrings:**
- Purpose and scope
- List of functions organized by category
- Technical details and implementation notes
- Usage examples
- References

**Function docstrings (NumPy style):**
- Brief description
- Parameters with types and defaults
- Returns with type
- Notes section for technical details
- Examples section with executable code

**Type hints:**
- All function parameters
- All return types
- Use `Union[str, Path]` for path parameters
- Use `Optional[...]` for nullable parameters

**Dual representation:**
- Temporal data stored in both samples and seconds
- Primary computation in sample domain (avoids rounding)
- Seconds for output and visualization
- Column naming: `*_samples`, `*_seconds`
- Legacy columns point to preferred unit

---

## Coding Conventions

### File Operations
```python
from pathlib import Path

# Always use Path objects
output_path = Path(output_path)
if output_path.suffix == '':
    output_path = output_path.with_suffix('.pdf')
output_path.parent.mkdir(parents=True, exist_ok=True)
```

### Parameter Patterns
```python
# Single output file → full path
def plot_function(..., output_path: Optional[Union[str, Path]] = None):
    
# Multiple output files → directory only
def analysis_function(..., output_dir: Union[str, Path] = '../figures/...'):
```

### Column Naming
```python
# Detected onsets
't_p_detected_samples', 't_p_detected_seconds'
't_s_detected_samples', 't_s_detected_seconds'

# Theoretical arrivals
't_p_theo_samples', 't_p_theo_seconds'
't_s_theo_samples', 't_s_theo_seconds'

# Window boundaries
'p_window_start_samples', 'p_window_start_seconds'
'p_window_end_samples', 'p_window_end_seconds'

# Legacy (backward compatibility)
't_p_detected' → points to 't_p_detected_seconds'
```

---

## Key Dependencies

```python
# Core scientific computing
import numpy as np
import pandas as pd
from scipy import stats

# Seismology
from obspy.signal.trigger import ar_pick
import seisbench.models as sbm

# Visualization
import matplotlib.pyplot as plt
from src.visualization.plot_settings import set_plot_style

# File operations
from pathlib import Path
```

---

## Module-Specific Notes

### `preprocessing/cleaning_signals.py`
**Critical choice:** `normalize=True` for PDF analysis, `normalize=False` for moment scaling (preserves physical units and variance structure).

### `segmentation/search_windows.py`
Uses CRUST1.0 model from `data/Crust1.0/model/`. Adaptive windows scale with epicentral distance to account for accumulated velocity model errors.

### `segmentation/ar_pick.py`
Four coda detection methods: Rautian (theoretical, 2×t_S), Arias (D5-95), Envelope (amplitude threshold), Median (robust combination). Rautian is recommended for consistency.

### `segmentation/window_segmentation.py`
Four non-overlapping windows: pre-event (noise reference), P-wave (first arrival to S), S-wave (S to coda), coda (scattered waves). Each window stored with dual representation (samples + seconds).

### `analysis/signals_scaling_spatial_ensemble.py`
**Tau generation:** Logarithmic spacing computed directly in sample domain, then converted to seconds only for output. Avoids rounding artifacts from seconds → samples → seconds conversion.

### `visualization/plot_settings.py`
Sets global matplotlib rcParams. Call `set_plot_style()` at the start of each notebook/script to ensure consistent styling across all figures.

---

## Getting Help

View module documentation:
```python
import src.segmentation
help(src.segmentation)
```

View function documentation:
```python
from src.segmentation import detect_onsets_arpick
help(detect_onsets_arpick)
```

View submodule documentation:
```python
from src.segmentation import ar_pick
help(ar_pick)
```

---

## Development Workflow

1. **Before modifying code:** Read module docstring and relevant function docstrings
2. **Test changes:** Run corresponding notebook (see main project README for notebook order)
3. **Update docstrings:** If changing function behavior or adding parameters
4. **Follow conventions:** Use type hints, dual representation, Path objects, consistent naming
5. **Avoid duplication:** Create small, reusable functions instead of copy-pasting logic
