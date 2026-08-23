# Analysis Notebooks

This directory contains Jupyter notebooks for seismic signal analysis and moment scaling characterization.

---

## Notebook Execution Order

Run notebooks in this sequence:

### Phase 1: Data Preparation

#### **01a. Metadata Processing**
**`01a_metadata.ipynb`** — Clean and prepare event metadata
- Load ITACA metadata and filter valid entries
- Quality control: remove duplicates, handle missing values
- Add derived columns (hypocentral distance, origin time)
- Export cleaned metadata to Parquet format

**Output:** `data/processed/01a_metadata/{signal_type}/metadata_clean_{signal_type[:3]}.parquet`

---

#### **01b. Signal Preprocessing**
**`01b_signals.ipynb`** — Baseline correction and signal preparation
- Load raw accelerometric traces from ITACA
- Apply baseline correction (polynomial detrending)
- Integrate to velocity and displacement (cumulative trapezoidal)
- Quality control: check for NaN, infinite values
- Export two versions:
  - **Normalized** (for PDF analysis): signals scaled to unit variance
  - **Unnormalized** (for phase detection and moment scaling): preserves physical amplitudes

**Output:**
- `data/processed/01b_signals/{signal_type}/{signal_type[:3]}_preprocessed_pdf.parquet` (normalized)
- `data/processed/01b_signals/{signal_type}/{signal_type[:3]}_preprocessed_scaling.parquet` (unnormalized)

---

### Phase 2: Statistical Characterization

#### **02. PDF Analysis**
**`02_pdf_analysis.ipynb`** — Probability density function fitting
- Empirical PDF computation with kernel density estimation
- Gaussian fit assessment (Anderson-Darling normality test)
- Heavy-tail model comparison (Gaussian, Laplace, Student-t, Lévy stable)
- Best-fit selection via AIC/BIC
- Hill estimator for power-law tail exponents

**Output:**
- `data/processed/02_pdf_analysis/{signal_type}/gaussian_fit_results.parquet`
- `data/processed/02_pdf_analysis/{signal_type}/heavy_tail_results.parquet`
- Figures: `figures/02_pdf_analysis/{signal_type}/`

---

### Phase 3: Phase Detection and Window Segmentation

#### **03a. AR-AIC Phase Picking**
**`03a_phase_identification_ar_pick.ipynb`** — Traditional phase detection
- Query CRUST1.0 for crustal velocities (depth-weighted averaging)
- Compute theoretical P/S arrival times (hypocentral distance + 1D ray model)
- Detect P and S onsets using AR-AIC method (ObsPy implementation)
- Detect coda onset (4 methods: Rautian, Arias D5-95, Envelope decay, Median)
- Segment signals into 4 windows: pre-event, P-wave, S-wave, coda
- Quality control: peak timing, monotonicity with distance, SNR ≥ 3

**Output:**
- `data/processed/03a_phase_identification_ar_pick/{signal_type}/df_full_{signal_type}_ar_pick.parquet`
- `data/processed/03a_phase_identification_ar_pick/{signal_type}/windowed_{signal_type}_{coda_method}_ar_pick.pkl` (4 files)
- Figures: `figures/03a_phase_identification_ar_pick/{signal_type}/`

---

#### **03b. PhaseNet Phase Picking**
**`03b_phase_identification_phasenet.ipynb`** — Deep learning phase detection
- Load PhaseNet model (INSTANCE pre-trained via SeisBench)
- Resample signals 200 Hz → 100 Hz (model training rate)
- Detect P and S onsets via CNN (U-Net architecture)
- Detect coda onset (same 4 methods as AR-AIC for comparison)
- Segment signals into 4 windows
- Quality control (same checks as AR-AIC)

**Output:**
- `data/processed/03b_phase_identification_phasenet/{signal_type}/df_full_{signal_type}_phasenet.parquet`
- `data/processed/03b_phase_identification_phasenet/{signal_type}/windowed_{signal_type}_{coda_method}_phasenet.pkl` (4 files)
- Figures: `figures/03b_phase_identification_phasenet/{signal_type}/`

**Note:** PhaseNet picks are typically 3-5 seconds earlier than AR-AIC. Both methods are analyzed to assess impact on moment scaling results.

---

### Phase 4: Moment Scaling Analysis

#### **04a. Spatial Ensemble Scaling**
**`04a_moment_scaling_spatial.ipynb`** — Multifractal moment scaling analysis
- Load windowed signals from 03a or 03b (select picker via configuration)
- Compute signal increments Δx(τ) for logarithmically-spaced time lags τ
- Calculate q-th order moments across spatial ensemble: M_q(τ) = ⟨|Δx(τ)|^q⟩
- Fit scaling exponents: M_q(τ) ~ τ^ζ(q) via linear regression in log-log space
- Analyze ζ(q) spectrum:
  - Linearity test (normal vs. anomalous diffusion)
  - Comparison to Brownian expectation ζ(q) = q/2
  - Multifractal signature (nonlinear ζ(q))
- Process all 4 windows (pre-event, P, S, coda) and all 3 signal types (acceleration, velocity, displacement)

**Output:**
- `data/processed/04a_moment_scaling_spatial/{picker}/{signal_type}/{coda_method}/moments_*.parquet`
- `data/processed/04a_moment_scaling_spatial/{picker}/{signal_type}/{coda_method}/exponents_*.parquet`
- Figures: `figures/04a_moment_scaling_spatial/{picker}/{signal_type}/{coda_method}/`

**Total runs:** 3 signal types × 2 pickers × 4 coda methods = **24 complete analyses**

**This is the core analysis notebook for the thesis.**

---

## Quick Start

### First Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda env create -f environment.yml
conda activate tesi-seismic

# Start Jupyter
jupyter notebook
```

### Minimal Pipeline (Single Signal Type)

```bash
# 1. Prepare data
jupyter notebook 01a_metadata.ipynb  # Execute all cells
jupyter notebook 01b_signals.ipynb   # Execute all cells

# 2. Phase detection (choose one)
jupyter notebook 03a_phase_identification_ar_pick.ipynb

# 3. Moment scaling
jupyter notebook 04a_moment_scaling_spatial.ipynb
```

Set `DATA_TYPE = 'acceleration'` in configuration cells to process acceleration data only.

### Complete Analysis (All Signal Types, Both Pickers)

Run each notebook 3 times (acceleration, velocity, displacement):
1. `01a_metadata.ipynb` (once per signal type)
2. `01b_signals.ipynb` (once per signal type)
3. `03a_phase_identification_ar_pick.ipynb` (once per signal type)
4. `03b_phase_identification_phasenet.ipynb` (once per signal type)
5. `04a_moment_scaling_spatial.ipynb` (6 times: 3 signal types × 2 pickers)

**Total:** ~24 notebook runs for complete analysis matrix.

---

## Expected Execution Times

| Notebook | Runtime | Memory | Notes |
|----------|---------|--------|-------|
| 01a_metadata.ipynb | ~30 sec | Low | Fast DataFrame operations |
| 01b_signals.ipynb | ~2-3 min | Medium | Integration to vel/disp |
| 02_pdf_analysis.ipynb | ~10-15 min | Medium | Lévy fitting slow |
| 03a_phase_identification_ar_pick.ipynb | ~5-10 min | Medium | AR-AIC + windowing |
| 03b_phase_identification_phasenet.ipynb | ~3-5 min | High | CNN inference (GPU helps) |
| **04a_moment_scaling_spatial.ipynb** | **~15-20 min** | **High** | **Core scaling analysis** |

Times are approximate for 66 traces (22 stations × 3 components). GPU accelerates PhaseNet (~2× speedup).

---

## Configuration

Each notebook has a `Configuration` section near the top:

```python
# Select data type
DATA_TYPE = 'acceleration'  # Options: 'acceleration', 'velocity', 'displacement'

# For 04a: also select phase picker
PICKING_METHOD = 'ar_pick'  # Options: 'ar_pick', 'phasenet'
```

Change `DATA_TYPE` to process different signal types. All paths and column names adjust automatically.

---

## Dependencies

### Data Dependencies
- **01a, 01b:** Require raw ITACA data in `data/raw/`
- **02:** Requires output from `01b_signals.ipynb`
- **03a, 03b:** Require output from `01b_signals.ipynb` (unnormalized signals)
- **04a:** Requires output from `03a` or `03b` (windowed signals)

### Code Dependencies
All notebooks import from `src/`:
- `src/preprocessing/` — Baseline correction, integration, metadata cleaning
- `src/segmentation/` — Phase detection (AR-AIC, PhaseNet), windowing
- `src/analysis/` — Moment scaling, PDF fitting, autocorrelation
- `src/visualization/` — Plotting functions for all analysis types
- `src/utils/` — LaTeX table export, logging utilities

See [`src/README.md`](../src/README.md) for module documentation.

### External Dependencies
See `requirements.txt` or `environment.yml` for complete package list. Key packages:
- **pandas, numpy, scipy** — Data processing and statistics
- **matplotlib, seaborn** — Visualization
- **obspy** — Seismic data handling and AR-AIC picker
- **seisbench** — PhaseNet model and inference
- **pyarrow** — Parquet file I/O

---

## Workflow Summary

```
01a_metadata.ipynb  →  metadata_clean_*.parquet
        ↓
01b_signals.ipynb  →  *_preprocessed_pdf.parquet (normalized)
                   →  *_preprocessed_scaling.parquet (unnormalized)
        ↓
        ├→ 02_pdf_analysis.ipynb  →  figures/02_pdf_analysis/
        │
        ├→ 03a_phase_identification_ar_pick.ipynb
        │       ↓
        │   windowed_*_ar_pick.pkl (4 coda methods)
        │       ↓
        │   04a_moment_scaling_spatial.ipynb (picker='ar_pick')
        │       ↓
        │   exponents_*.parquet, figures/04a_moment_scaling_spatial/ar_pick/
        │
        └→ 03b_phase_identification_phasenet.ipynb
                ↓
            windowed_*_phasenet.pkl (4 coda methods)
                ↓
            04a_moment_scaling_spatial.ipynb (picker='phasenet')
                ↓
            exponents_*.parquet, figures/04a_moment_scaling_spatial/phasenet/
```

---

## Analysis Strategy

### Signal Types
- **Acceleration:** High-frequency content, sensitive to spikes
- **Velocity:** Intermediate, smoother than acceleration
- **Displacement:** Low-frequency, long-period motions

All three are analyzed to compare scaling behavior across kinematic quantities.

### Phase Pickers
- **AR-AIC:** Traditional method, velocity-model-guided, interpretable
- **PhaseNet:** Deep learning, no velocity model needed, robust to noise

Both are analyzed to assess robustness of moment scaling exponents to window definition.

### Coda Detection Methods
- **Rautian (1978):** Theoretical definition, t_coda = 2·t_S - t_0
- **Arias D5-95:** Energy-based, 95% cumulative Arias Intensity
- **Envelope decay:** Amplitude-based, 25% of peak envelope threshold
- **Median:** Robust combination of the three above

All four are analyzed to quantify sensitivity of ζ(q) to coda onset timing.

### Recommended Analysis Path

**For thesis main results:**
1. Run **acceleration + AR-AIC + Rautian** first (most standard combination)
2. Verify ζ(q) shows expected behavior (superdiffusion in coda, multifractal signature)
3. Extend to other signal types (velocity, displacement)
4. Compare AR-AIC vs. PhaseNet (assess picker impact)
5. Compare coda methods (report variation as uncertainty estimate)

**For robustness checks:**
- Use **Median coda method** as alternative to Rautian
- Report uncertainty as: ζ(q) = X ± σ where σ spans all 4 coda methods

---

## Outputs

### Data Outputs
Saved to `data/processed/`:
- **01a:** Cleaned metadata (Parquet)
- **01b:** Preprocessed signals (Parquet, two versions)
- **02:** PDF fit results (Parquet)
- **03a/03b:** Full metadata with onsets (Parquet), windowed signals (Pickle)
- **04a:** Moments M_q(τ) and exponents ζ(q) (Parquet)

### Figure Outputs
Saved to `figures/`:
- **01a_metadata/** — Catalog statistics, station maps
- **01b_signals/** — Signal examples, preprocessing diagnostics
- **02_pdf_analysis/** — Empirical PDFs, Gaussian fits, heavy-tail fits
- **03a_phase_identification_ar_pick/** — Onset detection plots, residuals, QC
- **03b_phase_identification_phasenet/** — Same as 03a but for PhaseNet
- **04a_moment_scaling_spatial/** — Scaling curves M_q(τ), exponent spectra ζ(q)

See [`figures/README.md`](../figures/README.md) for complete figure catalog.

---

## Notebook Structure

Each notebook follows this template:

1. **Title and abstract** — Overview of analysis and outputs
2. **Imports and visualization settings** — Load packages, set plot style
3. **Configuration** — Select DATA_TYPE, PICKING_METHOD, define paths
4. **Data loading** — Read input files from previous steps
5. **Main analysis** — Core computations with progress logging
6. **Visualization** — Generate publication-quality figures
7. **Export results** — Save Parquet/Pickle files, call present_files
8. **Summary** — Key findings and next steps

All notebooks use consistent:
- **Logging:** `logger.info()` for progress tracking
- **Path handling:** `pathlib.Path` objects (no `os.path`)
- **File formats:** Parquet for tabular data, Pickle for nested structures
- **Naming conventions:** Dual representation (samples + seconds) for all time values

---

## Troubleshooting

### "Module not found" errors

```bash
# Ensure you're in project root
cd /path/to/tesi-seismic-analysis

# Install requirements
pip install -r requirements.txt

# Or activate conda environment
conda activate tesi-seismic
```

### "File not found" errors

- Check that previous notebooks have been run
- Verify `DATA_TYPE` matches in current and previous notebooks
- Check file paths in `data/processed/`

### Memory errors (notebook crashes)

- Close other applications
- Reduce ensemble size (modify code to process subset of stations)
- Use smaller q_values range in 04a

### PhaseNet inference slow

- Install PyTorch with GPU support (see PyTorch website for instructions)
- Reduce batch size in `apply_phasenet_to_signals()` if GPU memory limited

### Plots not rendering

```python
# Add at notebook start
%matplotlib inline
```

### "No module named 'src'" error

```python
# Verify path setup in notebook (should be automatic)
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

---

## Code Style

All notebooks follow these conventions:

### Cell Organization
- **Markdown cells:** Section headers and theory explanations only
- **Code cells:** Action labels as comments (`# Run analysis`, `# Display results`, `# Save results`)
- **No excessive formatting:** Minimize bold, bullets, emoji in markdown

### Code Patterns
```python
# Action comment
logger.info("Starting analysis...")
results = analyze_function(data, param1=value1, param2=value2)

# Display results
display(results.head())
logger.info(f"Analysis complete: {len(results)} rows")

# Save results
output_file = OUTPUT_DIR / 'results.parquet'
results.to_parquet(output_file, index=False)
logger.info(f"Saved: {output_file}")
```

### Naming Conventions
- **Variables:** `snake_case` (e.g., `df_signals`, `window_data`)
- **Constants:** `UPPER_CASE` (e.g., `DATA_TYPE`, `FIGURES_DIR`)
- **Functions:** `snake_case` with descriptive verbs (e.g., `compute_moments`, `plot_scaling_curves`)

---

## References

### Theoretical Framework
- **Beck & Cohen (2003)** — Superstatistics and anomalous diffusion
- **Vollmer et al. (2024)** — Moment scaling functions for correlated time series
- **Rautian & Khalturin (1978)** — Coda window definition

### Phase Detection Methods
- **Leonard & Kennett (1999)** — AR-AIC phase picking
- **Zhu & Beroza (2019)** — PhaseNet deep learning picker
- **Woollam et al. (2022)** — SeisBench framework

### Data Sources
- **ITACA database** — Italian strong-motion records
- **CRUST1.0** — Global crustal velocity model (Laske et al., 2013)

### Code Documentation
- Function docstrings: `src/` modules (NumPy style)
- Inline comments: Notebooks (explain non-obvious steps)
- Module-level documentation: `src/README.md`, `data/README.md`, `figures/README.md`

---

## Contributing

If you modify notebooks:

1. **Clear all outputs** before committing:
   ```
   Cell → All Output → Clear
   ```

2. **Test complete execution:**
   ```
   Cell → Run All
   ```
   Verify no errors from top to bottom.

3. **Update this README** if you:
   - Add/remove notebooks
   - Change workflow dependencies
   - Modify configuration options

4. **Document new features:**
   - Add docstrings to new functions in `src/`
   - Update relevant README files
   - Add examples in notebook markdown cells

---

## Environment

Notebooks tested with:
- **Python:** 3.9, 3.10, 3.11, 3.12
- **Jupyter:** Notebook 6.x, JupyterLab 3.x
- **Operating Systems:** macOS, Linux, Windows (WSL recommended)
- **Hardware:** 8GB RAM minimum, 16GB recommended for 04a

See `pyproject.toml` for exact package versions.

---

## Questions?

- **Code questions:** Check function docstrings in `src/` modules
- **Method questions:** See markdown explanations in notebooks
- **Bug reports:** Open issue on GitHub with error message and traceback
- **Feature requests:** Describe use case and expected behavior

---

**Last updated:** May 2026  
**Notebook structure version:** 1.0 (standardized pipeline)
