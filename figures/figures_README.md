# Analysis Figures

This directory contains all figures generated.

**Quick Download Options:**
- [Download entire figures folder as ZIP](https://github.com/giulianaparadiso99/tesi-seismic-analysis/archive/refs/heads/main.zip) (extract `figures/` from archive)
- Browse individual files below

---

## Directory Structure

```
figures/
├── 01_metadata/          # Metadata analysis and catalog statistics
├── 02_signals/           # Signal processing and preprocessing
├── 03_single_signal/     # Single signal analysis (main analysis)
│   ├── acceleration_pdfs_individual_data/
│   ├── autocorrelation/
│   ├── gaussian_fit/
│   ├── heavy_tail/
│   ├── pdf_single/
│   └── scaling/
└── 04_aggregated/        # Aggregated analysis across multiple signals
    ├── gaussian_fit/
    └── pdf/
```

---

## 1. Metadata Analysis

**Folder:** `01_metadata/`

Contains catalog statistics, event distribution, and metadata visualizations.

**Download:**
- [View folder on GitHub](01_metadata)
- Individual files:
  - Coming soon (add specific files as generated)

---

## 2. Signal Processing

**Folder:** `02_signals/`

Signal preprocessing, baseline correction, and quality control.

**Download:**
- [View folder on GitHub](02_signals)
- Individual files:
  - Coming soon (add specific files as generated)

---

## 3. Single Signal Analysis

**Folder:** `03_single_signal/`

Main analysis of individual seismic signals.

### 3.1 Acceleration PDFs (Individual Data)

**Folder:** `03_single_signal/acceleration_pdfs_individual_data/`

Probability density functions for acceleration data, individual files.

**Download:**
- [View folder on GitHub](03_single_signal/acceleration_pdfs_individual_data)

---

### 3.2 Autocorrelation Analysis

**Folder:** `03_single_signal/autocorrelation/`

Temporal correlation analysis of seismic signals.

**Download:**
- [View folder on GitHub](03_single_signal/autocorrelation)

**Key Figures:**
- [Add specific PDF links here as generated]

---

### 3.3 Gaussian Fit Analysis

**Folder:** `03_single_signal/gaussian_fit/`

Gaussian distribution fitting for signal increments.

**Download:**
- [View folder on GitHub](03_single_signal/gaussian_fit)

**Key Figures:**
- [Add specific PDF links here as generated]

---

### 3.4 Heavy Tail Analysis

**Folder:** `03_single_signal/heavy_tail/`

Heavy tail assessment and power-law fitting.

**Download:**
- [View folder on GitHub](03_single_signal/heavy_tail)

**Key Figures:**
- [Add specific PDF links here as generated]

---

### 3.5 Probability Density Functions (Single)

**Folder:** `03_single_signal/pdf_single/`

Individual PDF analysis for single signals.

**Download:**
- [View folder on GitHub](03_single_signal/pdf_single)

**Key Figures:**
- [Add specific PDF links here as generated]

---

### 3.6 Moment Scaling Analysis

**Folder:** `03_single_signal/scaling/`

Moment scaling, scaling exponents ζ(q), and multifractal analysis.

**Subfolders:**

#### 3.6.1 Acceleration Analysis

**Download full folder:**
- [View acceleration/ on GitHub](03_single_signal/scaling/acceleration)

**Subfolders:**
- **Full Signal**
  - [Exponents](03_single_signal/scaling/acceleration/full_signal/exponents) - Scaling exponent ζ(q) estimation
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/acceleration/full_signal/exponents/scaling_exponents_overlay.pdf?raw=true)
    - [scaling_exponents_individual.pdf](03_single_signal/scaling/acceleration/full_signal/exponents/scaling_exponents_individual.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/acceleration/full_signal/linearity) - Linearity test ζ(q) vs q
    - [scaling_linearity_test.pdf](03_single_signal/scaling/acceleration/full_signal/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/acceleration/full_signal/piecewise) - Piecewise scaling fit
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/acceleration/full_signal/piecewise/scaling_piecewise_fit.pdf?raw=true)

- **Event Window**
  - [Exponents](03_single_signal/scaling/acceleration/event_window/exponents)
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/acceleration/event_window/exponents/scaling_exponents_overlay.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/acceleration/event_window/linearity)
    - [scaling_linearity_test.pdf](03_single_signal/scaling/acceleration/event_window/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/acceleration/event_window/piecewise)
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/acceleration/event_window/piecewise/scaling_piecewise_fit.pdf?raw=true)

#### 3.6.2 Velocity Analysis

**Download full folder:**
- [View velocity/ on GitHub](03_single_signal/scaling/velocity)

**Subfolders:**
- **Full Signal**
  - [Exponents](03_single_signal/scaling/velocity/full_signal/exponents)
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/velocity/full_signal/exponents/scaling_exponents_overlay.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/velocity/full_signal/linearity)
    - [scaling_linearity_test.pdf](03_single_signal/scaling/velocity/full_signal/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/velocity/full_signal/piecewise)
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/velocity/full_signal/piecewise/scaling_piecewise_fit.pdf?raw=true)

- **Event Window**
  - [Exponents](03_single_signal/scaling/velocity/event_window/exponents)
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/velocity/event_window/exponents/scaling_exponents_overlay.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/velocity/event_window/linearity)
    - [scaling_linearity_test.pdf](03_single_signal/scaling/velocity/event_window/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/velocity/event_window/piecewise)
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/velocity/event_window/piecewise/scaling_piecewise_fit.pdf?raw=true)

#### 3.6.3 Displacement Analysis

**Download full folder:**
- [View displacement/ on GitHub](03_single_signal/scaling/displacement)

**Subfolders:**
- **Full Signal**
  - [Exponents](03_single_signal/scaling/displacement/full_signal/exponents)
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/displacement/full_signal/exponents/scaling_exponents_overlay.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/displacement/full_signal/linearity)
    - [scaling_linearity_test.pdf](03_single_signal/scaling/displacement/full_signal/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/displacement/full_signal/piecewise)
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/displacement/full_signal/piecewise/scaling_piecewise_fit.pdf?raw=true)

- **Event Window**
  - [Exponents](03_single_signal/scaling/displacement/event_window/exponents)
    - [scaling_exponents_overlay.pdf](03_single_signal/scaling/displacement/event_window/exponents/scaling_exponents_overlay.pdf?raw=true)
  - [Linearity](03_single_signal/scaling/displacement/event_window/linearity)
    - [scaling_linearity_test.pdf](03_single_signal/scaling/displacement/event_window/linearity/scaling_linearity_test.pdf?raw=true)
  - [Piecewise](03_single_signal/scaling/displacement/event_window/piecewise)
    - [scaling_piecewise_fit.pdf](03_single_signal/scaling/displacement/event_window/piecewise/scaling_piecewise_fit.pdf?raw=true)

---

## 4. Aggregated Analysis

**Folder:** `04_aggregated/`

Analysis aggregated across multiple signals.

### 4.1 Gaussian Fit (Aggregated)

**Folder:** `04_aggregated/gaussian_fit/`

Gaussian distribution fitting for aggregated data.

**Download:**
- [View folder on GitHub](04_aggregated/gaussian_fit)

**Key Figures:**
- [Add specific PDF links here as generated]

---

### 4.2 PDF Analysis (Aggregated)

**Folder:** `04_aggregated/pdf/`

Probability density function analysis for aggregated signals.

**Download:**
- [View folder on GitHub](04_aggregated/pdf)

**Key Figures:**
- [Add specific PDF links here as generated]

---

## Exploratory Analysis

**Folder:** `exploratory/`

Quality control and exploratory analysis plots.

### Process Analysis

**Folder:** `exploratory/processes/`

Analysis of integrated processes (velocity, displacement).

**Download:**
- [View folder on GitHub](exploratory/processes)

**Key Figures:**
- [process_analysis_velocity.pdf](exploratory/processes/process_analysis_velocity.pdf?raw=true) - Velocity distribution, drift check, time series
- [process_analysis_displacement.pdf](exploratory/processes/process_analysis_displacement.pdf?raw=true) - Displacement distribution, drift check, time series

---

### Increment Analysis

**Folder:** `exploratory/increments/`

Analysis of increment distributions, symmetry, and scaling.

**Download:**
- [View folder on GitHub](exploratory/increments)

**Key Figures:**
- [increment_analysis_displacement.pdf](exploratory/increments/increment_analysis_displacement.pdf?raw=true) - Increment distributions, scaling with τ, heavy tails

---

## How to Use

### Download Individual Files
Click on any `.pdf` link → File downloads automatically → Open in your PDF viewer

### Download Entire Folders
1. Click "View folder on GitHub" link
2. Click green "Code" button → "Download ZIP"
3. Or use git: `git clone https://github.com/yourusername/tesi-seismic-analysis.git`

### Browse on GitHub
GitHub doesn't render PDFs directly, but you can:
1. Click on any folder link to browse
2. Click on individual files to see metadata
3. Download files individually

---

## 🔄 Auto-Generated

This README is maintained alongside the analysis code. Files are added automatically as analyses complete.

**Last updated:** April 2026

---

## Reference

For methodology and theoretical background, see:
- `notebooks/03b_moment_scaling.ipynb` - Main analysis notebook
- `docs/methodology.md` - Detailed methodology
- Vollmer et al. (2024) - Theoretical framework

---

## Contact

For questions about specific figures or analysis methods:
- Open an issue on GitHub
- Contact: giulianaparadiso99@gmail.com
