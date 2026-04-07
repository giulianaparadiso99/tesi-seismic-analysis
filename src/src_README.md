# Source Code

Python modules for seismic signal processing, analysis, and visualization.

---

## Module Organization

The source code is organized into four functional categories:

### 1. Data I/O & Preprocessing
**`io.py`** - Load raw seismic data from .ASC archive  
**`cleaning_metadata.py`** - Clean and preprocess metadata  
**`cleaning_signals.py`** - Preprocess acceleration time series

### 2. Statistical Analysis
**`signals_pdf.py`** - Probability density function analysis  
**`signals_scaling.py`** - Moment scaling and multifractal analysis  
**`signals_autocorrelation.py`** - Temporal correlation analysis (work in progress)

### 3. Visualization
**`plots_metadata.py`** - Metadata visualization (maps, distributions, correlations)  
**`plots_signals.py`** - Signal visualization (time series, distributions, diagnostics)  
**`plot_settings.py`** - Global matplotlib style configuration

### 4. Export Utilities
**`latex_export.py`** - Export analysis results to LaTeX tables

---

## Module Documentation

Each module contains detailed docstrings at the file level and for individual functions. Below is a brief overview of each module's purpose and main functions.

---

### Data I/O & Preprocessing

#### `io.py`
Load seismic waveforms and metadata from the .ASC archive.

**Main functions:**
- `build_metadata(zip_path)` - Extract metadata from .ASC headers
- `build_accelerations(zip_path)` - Extract acceleration time series
- `build_dataframes(zip_path)` - Load both metadata and accelerations

**Usage:**
```python
from src.io import build_dataframes

df_meta, df_acc = build_dataframes('../data/raw/query.zip')
```

---

#### `cleaning_metadata.py`
Five-step preprocessing pipeline for metadata.

**Pipeline steps:**
1. Replace missing values (empty strings → NaN)
2. Drop uninformative columns
3. Convert data types (numeric, datetime)
4. Normalize strings (strip whitespace)
5. Remove duplicates

**Usage:**
```python
from src.cleaning_metadata import clean_metadata

df_meta_clean = clean_metadata(df_meta_raw)
```

---

#### `cleaning_signals.py`
Flexible preprocessing for acceleration signals with independent control over each step.

**Main function:**
- `preprocess_signals()` - Configurable pipeline with three optional steps:
  - Length filtering (retain long signals for moment scaling)
  - Baseline correction (remove per-signal mean)
  - Normalization (standardize to unit variance)

**Critical choice:** Normalization ON for PDF analysis, OFF for moment scaling.

**Usage:**
```python
from src.cleaning_signals import preprocess_signals

# PDF analysis: normalize
df_pdf = preprocess_signals(df_acc_raw,
                             filter_length=False,
                             baseline_correction=True,
                             normalize=True)

# Moment scaling: preserve physical units
df_scaling = preprocess_signals(df_acc_raw,
                                 filter_length=True,
                                 baseline_correction=True,
                                 normalize=False)
```

---

### Statistical Analysis

#### `signals_pdf.py`
Assess statistical distributions of seismic increments.

**Main functions:**
- `gaussian_fit_analysis()` - Test for Gaussianity (Anderson-Darling, kurtosis, skewness)
- `heavy_tail_analysis()` - Characterize heavy tails (Lévy-stable, Student-t, power-law)

**Outputs:** Individual plots per signal + summary aggregations

**Usage:**
```python
from src.signals_pdf import gaussian_fit_analysis, heavy_tail_analysis

# Gaussian fit with normality tests
df_gaussian = gaussian_fit_analysis(
    df_acc_clean,
    bins=100,
    normalized=True,
    output_dir='../figures/03_single_signal/03a_pdf_analysis/gaussian_fit'
)

# Heavy tail assessment
df_tail = heavy_tail_analysis(
    df_acc_clean,
    normalized=True,
    output_dir='../figures/03_single_signal/03a_pdf_analysis/heavy_tail'
)
```

---

#### `signals_scaling.py`
Multifractal moment scaling analysis.

**Main functions:**
- `integrate_to_velocity()`, `integrate_to_displacement()` - Process integration
- `compute_increments()` - Calculate Δx(τ, t₀) for multiple time lags
- `compute_moments_from_increments()` - Compute M_q(τ) = ⟨|Δx|^q⟩
- `compute_scaling_exponents()` - Fit ζ(q) from M_q(τ) ~ τ^ζ(q)
- `test_scaling_linearity()` - Test if ζ(q) is linear in q
- `fit_piecewise_scaling()` - Detect breakpoints in ζ(q)

**Usage:**
```python
from src.signals_scaling import (
    integrate_to_displacement,
    compute_increments,
    compute_moments_from_increments,
    compute_scaling_exponents
)

# Integration
df_disp = integrate_to_displacement(df_acc, method='trapz')

# Increments
df_inc = compute_increments(df_disp, tau_values=[1, 2, 5, 10, 20])

# Moments
df_mom = compute_moments_from_increments(df_inc, q_values=[0.5, 1, 2, 3, 4, 5])

# Scaling exponents
df_zeta = compute_scaling_exponents(df_mom)
```

---

#### `signals_autocorrelation.py`
Temporal correlation analysis (module under development).

---

### Visualization

#### `plots_metadata.py`
Visualization suite for metadata exploration.

**Main functions:**
- `plot_column_types_pie()` - Data type distribution
- `plot_numerical_distributions()` - Histograms for numeric variables
- `plot_categorical_distributions()` - Bar charts for categorical variables
- `plot_correlation_matrix()` - Heatmap of correlations
- `plot_station_map()` - Geographic map with PGA overlay
- `plot_pga_and_duration_by_component()` - Component-wise analysis

**Usage:**
```python
from src.plots_metadata import plot_station_map

plot_station_map(
    df_meta_clean,
    event_lat=42.8,
    event_lon=13.1,
    output_path='../figures/01_metadata/station_map.pdf'
)
```

---

#### `plots_signals.py`
Visualization for signal preprocessing and validation.

**Main functions:**
- `plot_signal_length_distribution()` - Sample count histogram
- `plot_example_signals()` - Representative time series per component
- `plot_acceleration_distributions()` - Global and per-component distributions
- `plot_postcheck_pdf()` - Validation for PDF analysis pipeline
- `plot_postcheck_moment_scaling()` - Validation for moment scaling pipeline
- `plot_increments_histograms_dual_view()` - Dual-view increment distributions

**Usage:**
```python
from src.plots_signals import plot_postcheck_pdf

plot_postcheck_pdf(
    df_acc_raw,
    df_acc_clean,
    output_dir='../figures/02_signals'
)
```

---

#### `plot_settings.py`
Global matplotlib style configuration.

**Function:**
- `set_plot_style()` - Apply project-wide rcParams and return color palette

**Usage:**
```python
from src.plot_settings import set_plot_style

colors = set_plot_style()
# colors[0-3] for HNE/HNN/HNZ components or distance groups
```

---

### Export Utilities

#### `latex_export.py`
Export analysis results to LaTeX table format.

**Main functions:**
- `corr_diff_to_latex()` - Correlation difference table
- `preprocess_checks_to_latex()` - Quality check summary table
- `heavy_tail_to_latex()` - Heavy tail assessment longtable rows

**Usage:**
```python
from src.latex_export import heavy_tail_to_latex

latex_rows = heavy_tail_to_latex(
    df_heavy_tail,
    output_path='../data/processed/latex_tables/heavy_tail_table.tex'
)
```

---

## Quick Start Examples

### Complete PDF Analysis Workflow
```python
# 1. Load data
from src.io import build_dataframes
df_meta, df_acc = build_dataframes('../data/raw/query.zip')

# 2. Clean metadata
from src.cleaning_metadata import clean_metadata
df_meta_clean = clean_metadata(df_meta)

# 3. Preprocess signals (with normalization)
from src.cleaning_signals import preprocess_signals
df_acc_clean = preprocess_signals(df_acc, normalize=True)

# 4. Analyze distributions
from src.signals_pdf import gaussian_fit_analysis
df_gaussian = gaussian_fit_analysis(df_acc_clean, normalized=True)
```

---

### Complete Moment Scaling Workflow
```python
# 1. Load and preprocess (WITHOUT normalization)
from src.io import build_accelerations
from src.cleaning_signals import preprocess_signals

df_acc_raw = build_accelerations('../data/raw/query.zip')
df_acc = preprocess_signals(df_acc_raw, filter_length=True, normalize=False)

# 2. Compute increments and moments
from src.signals_scaling import compute_increments, compute_moments_from_increments

df_inc = compute_increments(df_acc, tau_values=range(1, 101))
df_mom = compute_moments_from_increments(df_inc, q_values=[0.5, 1, 2, 3, 4, 5])

# 3. Extract scaling exponents
from src.signals_scaling import compute_scaling_exponents
df_zeta = compute_scaling_exponents(df_mom)
```

---

## Data Flow

```
Raw .ASC files (query.zip)
    ↓
[io.py] Load data
    ↓
df_meta_raw + df_acc_raw
    ↓
[cleaning_metadata.py + cleaning_signals.py] Preprocess
    ↓
df_meta_clean + df_acc_clean
    ↓
    ├─→ [signals_pdf.py] PDF analysis → figures + tables
    ├─→ [signals_scaling.py] Moment scaling → exponents + plots
    └─→ [signals_autocorrelation.py] Autocorrelation → correlation functions
```

---

## Visualization Workflow

```
Preprocessed data
    ↓
[plot_settings.py] Apply global style
    ↓
    ├─→ [plots_metadata.py] Metadata figures
    └─→ [plots_signals.py] Signal figures
         ↓
    PDF files in ../figures/
```

---

## Documentation Standards

All modules follow these documentation conventions:

1. **File-level docstring** - Module purpose, organization, usage examples
2. **Function docstrings** - Parameters, returns, examples (NumPy style)
3. **Type hints** - Where applicable
4. **Inline comments** - For complex logic only

---

## Development Guidelines

### Code Style
- Follow PEP 8
- Use descriptive variable names
- Keep functions focused (single responsibility)
- Avoid hardcoded paths (use parameters)

### Testing
Run notebooks in order to verify module functionality:
1. `01_metadata.ipynb` → tests io.py, cleaning_metadata.py, plots_metadata.py
2. `02_signals.ipynb` → tests cleaning_signals.py, plots_signals.py
3. `03a_pdf_analysis.ipynb` → tests signals_pdf.py
4. `03b_moment_scaling.ipynb` → tests signals_scaling.py

---

## Dependencies

See `requirements.txt` in project root for complete list.

**Core dependencies:**
- numpy, pandas - Data manipulation
- matplotlib, seaborn - Visualization
- scipy - Statistical functions
- contextily - Basemap tiles for geographic plots

---

## Questions?

For detailed documentation, see individual module docstrings:
```python
import src.signals_scaling
help(src.signals_scaling)
help(src.signals_scaling.compute_scaling_exponents)
```
