# Statistical Modeling and Scaling Analysis of Seismic Acceleration Signals

This repository contains the complete analysis pipeline for my master's thesis at Politecnico di Torino, investigating **statistical properties and moment scaling behavior** of seismic signals from a single regional earthquake event.

---

## Research Overview

The project characterizes seismic ground motion signals through statistical and scaling analysis to detect signatures of **anomalous diffusion** and **out-of-equilibrium dynamics** in earthquake recordings.

### Key Research Questions

1. Do seismic signals exhibit **heavy-tailed distributions** compatible with Lévy stable or Student-t models?
2. Does the **moment scaling spectrum** ζ(q) deviate from normal diffusion expectations?
3. Is there evidence of **multifractality** (nonlinear ζ(q)) in seismic coda?
4. How robust are scaling exponents to **phase picker choice** (AR-AIC vs. PhaseNet)?
5. How sensitive are results to **coda window definition**?

### Methodological Framework

**Moment scaling analysis** (Beck & Cohen 2003; Vollmer et al. 2024):

$$M_q(\tau) = \langle |\Delta x(\tau)|^q \rangle \sim \tau^{\zeta(q)}$$

where:
- Δx(τ) = signal increment at time lag τ
- ζ(q) = scaling exponent (quantifies diffusion regime)
- **Normal diffusion:** ζ(q) = q/2 (linear)
- **Anomalous diffusion:** ζ(q) ≠ q/2 (sublinear or superlinear)
- **Multifractal:** ζ(q) nonlinear (intermittency, heavy tails)

### Dataset

**Event:** Mw 3.8 earthquake on 2024-12-09 near Italy-France border (Parco Naturale Regionale del Queyras)
- Depth: 10.4 km
- Stations: 22 accelerometric stations (ITACA database)
- Components: 3 per station (HNE, HNN, HNZ) → 66 independent traces
- Sampling rate: 200 Hz
- Distance range: 4.8 - 109.5 km (hypocentral: 11.4 - 110.0 km)

**Supervisors:**
- Prof. Lamberto Rondoni (Politecnico di Torino)
- Dr. Matteo Colangeli (University of L'Aquila)
- Dr. Federica Di Michele (INGV Milan)

---

## Repository Structure

```
tesi-seismic-analysis/
│
├── data/                    # Raw and processed data
│   ├── raw/                # ITACA database files (not in repo)
│   ├── processed/          # Generated outputs from notebooks
│   └── README.md          # Data directory structure and formats
│
├── notebooks/              # Jupyter analysis notebooks
│   ├── 01a_metadata.ipynb               # Metadata preprocessing
│   ├── 01b_signals.ipynb                # Signal preprocessing
│   ├── 02_pdf_analysis.ipynb            # PDF fitting
│   ├── 03a_phase_identification_ar_pick.ipynb     # AR-AIC phase detection
│   ├── 03b_phase_identification_phasenet.ipynb    # PhaseNet phase detection
│   ├── 04a_moment_scaling_spatial.ipynb           # Moment scaling analysis ★
│   └── README.md          # Notebook execution guide
│
├── src/                    # Python modules
│   ├── preprocessing/     # Baseline correction, integration
│   ├── segmentation/      # Phase detection, windowing
│   ├── analysis/          # Moment scaling, PDF fitting
│   ├── visualization/     # Plotting functions
│   ├── utils/             # LaTeX export, helpers
│   └── README.md          # Module documentation
│
├── figures/                # Generated plots (organized by notebook)
│   └── README.md          # Figure catalog
│
├── docs/                   # Documentation
│   └── references.md      # Bibliography
│
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

**Detailed documentation:** See README files in each subdirectory for complete information on data formats, notebook workflows, module APIs, and figure organization.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/giulianaparadiso99/tesi-seismic-analysis.git
cd tesi-seismic-analysis

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate tesi-seismic
```

### Minimal Analysis Pipeline

Run notebooks in sequence (single signal type):

```bash
jupyter notebook notebooks/01a_metadata.ipynb          # Clean metadata
jupyter notebook notebooks/01b_signals.ipynb           # Preprocess signals
jupyter notebook notebooks/03a_phase_identification_ar_pick.ipynb  # Detect phases
jupyter notebook notebooks/04a_moment_scaling_spatial.ipynb        # Compute ζ(q)
```

Set `DATA_TYPE = 'acceleration'` in each notebook configuration cell.

**Complete documentation:** See [`notebooks/README.md`](notebooks/README.md) for full pipeline details, execution times, and troubleshooting.

---

## Key Features

### Multi-Signal Analysis
- **Acceleration:** High-frequency content, spikes
- **Velocity:** Intermediate smoothness
- **Displacement:** Long-period motions

All three kinematic quantities analyzed to compare scaling behavior.

### Dual Phase Picker Comparison
- **AR-AIC (Leonard & Kennett 1999):** Traditional method, velocity-model-guided
- **PhaseNet (Zhu & Beroza 2019):** Deep learning (INSTANCE pre-trained)

Both methods processed to assess robustness of scaling exponents to window definition.

### Multiple Coda Detection Methods
- **Rautian (1978):** Theoretical, t_coda = 2·t_S - t_0
- **Arias D5-95:** Energy-based (95% cumulative Arias Intensity)
- **Envelope decay:** Amplitude-based (25% threshold)
- **Median:** Robust combination

Four methods compared to quantify sensitivity of ζ(q) to coda onset timing.

### Spatial Ensemble Averaging
- N = 66 independent traces (22 stations × 3 components)
- Ensemble-averaged moments: robust estimation, reduced fluctuations
- Components treated as independent realizations (justified by different wave polarizations)

---

## Analysis Workflow

**Complete workflow:** See ASCII diagram in [`notebooks/README.md`](notebooks/README.md)

```
Data Preparation → Statistical Characterization → Phase Detection → Moment Scaling
     (01a/01b)            (02)                      (03a/03b)           (04a)
```

**Core analysis:** `04a_moment_scaling_spatial.ipynb`
- Input: Windowed signals from phase detection
- Output: Scaling exponents ζ(q) for all windows, signal types, and coda methods
- Total runs: 3 signal types × 2 pickers × 4 coda methods = **24 complete analyses**

---

## Main Outputs

### Data Products
Generated and saved in `data/processed/`:
- Cleaned metadata (Parquet)
- Preprocessed signals (Parquet)
- PDF fit results (Parquet)
- Phase onset times and windowed signals (Parquet/Pickle)
- **Moment scaling exponents ζ(q)** (Parquet) — primary research output

**Format specifications:** See [`data/README.md`](data/README.md)

### Figures
Publication-quality plots in `figures/`:
- Metadata exploration (station maps, distance distributions)
- Signal preprocessing diagnostics
- PDF fits and heavy-tail assessment
- Phase detection validation
- **Moment scaling curves and ζ(q) spectra** — key thesis figures

**Figure catalog:** See [`figures/README.md`](figures/README.md) for complete list (~1000-1500 PDFs generated)

---

## Code Organization

### Module Structure

```python
# Import example
from src import (
    add_crustal_velocities,        # preprocessing
    detect_onsets_arpick,          # segmentation
    analyze_all_windows,           # analysis (core)
    plot_scaling_exponents         # visualization
)
```

**Detailed API:** See [`src/README.md`](src/README.md) for function documentation, examples, and coding conventions.

### Key Modules
- **`src/segmentation/`** — Phase detection (AR-AIC, PhaseNet), window segmentation
- **`src/analysis/`** — Moment scaling computation, PDF fitting
- **`src/visualization/`** — Consistent plotting across all notebooks

All modules follow:
- Type hints (Python 3.9+)
- NumPy-style docstrings
- Dual representation (samples + seconds) for time values

---

## Technical Details

### Dependencies
- **Python:** 3.9, 3.10, 3.11, 3.12 (tested)
- **Core:** pandas, numpy, scipy, matplotlib
- **Seismology:** obspy (AR-AIC), seisbench (PhaseNet)
- **Format:** pyarrow (Parquet I/O)

**Full list:** See `requirements.txt` or `environment.yml`

### Hardware Requirements
- **Minimum:** 8GB RAM
- **Recommended:** 16GB RAM (for `04a_moment_scaling_spatial.ipynb`)
- **GPU:** Optional, accelerates PhaseNet inference (~2× speedup)

### Execution Times
- **Full pipeline (single signal type):** ~30-40 minutes
- **Core scaling analysis (04a):** ~15-20 minutes
- **Complete matrix (24 runs):** ~6-8 hours

**Detailed benchmarks:** See [`notebooks/README.md`](notebooks/README.md)

---

## Key References

### Theoretical Framework
- **Beck, C., & Cohen, E. G. D.** (2003). Superstatistics. *Physica A*, 322, 267-275.
- **Vollmer, J., et al.** (2024). Moment scaling functions of long-range correlated time series. *Physical Review E*, 109(3), 034117.

### Phase Detection
- **Leonard, M., & Kennett, B. L. N.** (1999). Multi-component autoregressive techniques. *Physics of the Earth and Planetary Interiors*, 113(1-4), 247-263.
- **Zhu, W., & Beroza, G. C.** (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International*, 216(1), 261-273.

### Coda Analysis
- **Rautian, T. G., & Khalturin, V. I.** (1978). The use of the coda for determination of the earthquake source spectrum. *BSSA*, 68(4), 923-948.

**Complete bibliography:** See [`docs/references.md`](docs/references.md)

---

## Results Summary

**Note:** Final results to be completed after thesis defense. Preliminary findings:

### PDF Analysis (Acceleration)
- **Pre-event:** Gaussian noise (instrumental)
- **S-wave/Coda:** Heavy-tailed distributions (Lévy stable, Student-t)
- **Tail exponent:** α ~ 1.5-2.0 (power-law regime)

### Moment Scaling (Preliminary)
- **Pre-event:** ζ(q) ≈ q/2 (normal diffusion baseline)
- **P/S-wave:** ζ(q) > q/2 (superdiffusion)
- **Coda:** ζ(q) nonlinear (multifractal signature)
- **Robustness:** Δζ(q) ~ 0.05-0.10 across coda methods

### Phase Picker Comparison
- **AR-AIC vs. PhaseNet:** Systematic offset ~3-5 seconds (PhaseNet earlier)
- **Impact on ζ(q):** Under investigation

---

## Repository Status

**Current version:** Analysis pipeline complete, thesis in preparation (May 2026)

**Submitted abstract:** Entropy 2026 conference (Barcelona)
- Session: Non-Equilibrium Systems and Entropy Production / Statistical Physics and Stochastic Processes

---

## Contributing

This is a thesis project repository. If you find issues or have suggestions:
1. Open an issue with detailed description
2. Include error messages, traceback, and system info
3. Specify which notebook and configuration (`DATA_TYPE`, `PICKING_METHOD`)

---

## License

MIT License — See LICENSE file for details.

---

## Contact

**Giuliana Paradiso**  
Master's Student in Mathematical Engineering  
Politecnico di Torino  
📧 s319688@studenti.polito.it  
🔗 GitHub: [giulianaparadiso99](https://github.com/giulianaparadiso99)

---

## Acknowledgments

- **ITACA database** (INGV) for providing accelerometric data
- **Politecnico di Torino** Department of Mathematical Sciences
- **SeisBench** project for PhaseNet implementation
- **ObsPy** community for seismic analysis tools

---

**Last updated:** May 2026  
**Repository version:** 1.0 (standardized pipeline)
