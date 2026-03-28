# Seismic acceleration analysis

This repository contains the code used for the analysis of ground motion recordings in my master's thesis at Politecnico di Torino.

The project investigates **strong anomalous diffusion** in seismic displacement signals by analyzing accelerometric data from a single earthquake event recorded by 22 stations along the Italy-France border.

Following the theoretical framework of Vollmer et al. (2024), the analysis characterizes the scaling behavior of statistical moments to detect signatures of anomalous transport processes in earthquake ground motion. Key findings include:
- Evidence of **nonlinear moment scaling** in seismic displacement
- Detection of **two-regime structure** with distinct scaling exponents
- Heavy-tailed distributions in acceleration signals (Lévy stable, Student-t)
- Strong dependence on event window selection (pre-event noise vs. seismic signal)

The project performs comprehensive statistical analysis on both metadata and acceleration time series, exploring the relationship between seismic signal properties, event characteristics, and station location.

---

## Project Structure
```
tesi-seismic-analysis/
│
├── data/
│   ├── raw/
│   │   └── query.zip
│   └── processed/
│
├── notebooks/
│   ├── 01_metadata_preprocessing_exploration.ipynb
│   ├── 02_seismic_signals_preprocessing_exploration.ipynb
│   ├── 03a_pdf_analysis.ipynb
│   ├── 03b_moment_scaling.ipynb
│   ├── 03c_autocorrelation.ipynb
│   ├── 04a_pdf_aggregated_analysis.ipynb
│   ├── 04b_moment_scaling_aggregated.ipynb
│   ├── 04c_autocorrelation_aggregated.ipynb
│   ├── 05a_pdf_distance_groups.ipynb
│   ├── 05b_moment_scaling_distance_groups.ipynb
│   └── 05c_autocorrelation_distance_groups.ipynb
│
├── src/
│   ├── __init__.py
│   ├── io.py
│   ├── cleaning_metadata.py
│   ├── cleaning_signals.py
│   ├── plot_settings.py
│   ├── metadata.py
│   ├── signals.py
│   ├── signals_scaling.py
│   ├── signals_autocorrelation.py
│   ├── signals_pdf.py
│   ├── latex_export.py
│   └── plots.py
│
├── figures/
│   ├── 01_metadata/
│   ├── 02_signals/
│   ├── 03_single_signal/
│   ├── 04_aggregated/
│   └── 05_distance_groups/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:
```bash
git clone https://github.com/giulianaparadiso99/tesi-seismic-analysis.git
cd tesi-seismic-analysis
```

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

Install the project in editable mode:
```bash
pip install -e .
```

---

## Data

The dataset is not included in the repository due to size constraints.

Place the data archive in the following location:
```
data/raw/query.zip
```

The raw dataset should contain the .ASC files used in the analysis.

---

## Usage

The analysis is performed through Jupyter notebooks.

Start Jupyter:
```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory in order:

### 1. Preprocessing and exploration
- `01_metadata_preprocessing_exploration.ipynb` — metadata loading, preprocessing and exploration
- `02_seismic_signals_preprocessing_exploration.ipynb` — signal loading, preprocessing and exploration

### 2. Single signal analysis
- `03a_pdf_analysis.ipynb` — probability density function analysis and heavy-tail assessment
- `03b_moment_scaling.ipynb` — moment scaling analysis following Vollmer et al. (2024) framework
- `03c_autocorrelation.ipynb` — displacement autocorrelation analysis

### 3. Aggregated analysis
- `04a_pdf_aggregated_analysis.ipynb` — PDF comparison across distance groups
- `04b_moment_scaling_aggregated.ipynb` — moment scaling comparison across distance groups
- `04c_autocorrelation_aggregated.ipynb` — autocorrelation comparison across distance groups

Example code snippet:
```python
from src.io import build_metadata, build_accelerations

df_meta = build_metadata("../data/raw/query.zip")
df_acc = build_accelerations("../data/raw/query.zip")
```

---

## Key Analyses

### Moment Scaling Analysis
The moment scaling analysis investigates signatures of anomalous diffusion in seismic displacement signals, following the theoretical framework of Vollmer et al. (2024).

**Key features:**
- Analysis of acceleration, velocity, and displacement processes
- Computation of q-th order moments: `M_q(τ) = ⟨|Δx(τ)|^q⟩`
- Estimation of scaling exponents `ζ(q)` from `M_q(τ) ~ τ^ζ(q)`
- Detection of strong anomalous diffusion via piecewise linear scaling spectrum
- Comparison between full signal and event window (post-onset)

**Main functions:** `signals_scaling.py`
- `compute_moment_scaling_disp()` — compute moments of displacement increments
- `compute_scaling_exponents()` — estimate scaling exponents ζ(q)
- `test_scaling_linearity()` — test for nonlinearity in ζ(q)
- `fit_piecewise_scaling()` — detect two-regime structure

### PDF Analysis
Statistical characterization of acceleration distributions:
- Gaussian fit and Anderson-Darling normality test
- Heavy-tail assessment (Gaussian, Laplace, Student-t, Lévy stable)
- Hill estimator for power-law tail exponent

**Main functions:** `signals_pdf.py`

### Autocorrelation Analysis
Time-domain correlation analysis of displacement signals.

**Main functions:** `signals_autocorrelation.py`

---

## Output

Generated figures are saved in:
```
figures/
├── 01_metadata/
├── 02_signals/
├── 03_single_signal/
│   ├── pdf/
│   ├── scaling/
│   └── autocorrelation/
└── 04_aggregated/
```

Processed datasets are stored in:
```
data/processed/
```

---

## Requirements

Main Python libraries used in the project:
- `pandas` — data manipulation
- `numpy` — numerical computing
- `matplotlib` — plotting
- `seaborn` — statistical visualization
- `scipy` — scientific computing
- `jupyter` — interactive notebooks
- `contextily` — basemap tiles
- `adjustText` — label placement
- `pyarrow` — parquet file format

---

## References

The moment scaling analysis is based on:

Vollmer, J., et al. (2024). "Framework for strong anomalous diffusion." *Physical Review E*.

---

## Author

**Giuliana Paradiso**  
Politecnico di Torino

---

## License

This project is licensed under the MIT License.