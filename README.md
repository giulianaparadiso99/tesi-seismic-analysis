# Seismic acceleration analysis
This repository contains the code used for the analysis of ground motion recordings in my master's thesis at Politecnico di Torino.
The project processes accelerometric data stored in .ASC files and performs statistical analysis on metadata and acceleration time series.
The goal is to explore statistical properties of seismic signals and their relationship with event and station metadata.

--------------------------------------------------
## Project Structure
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
│   ├── 03_single_signal_analysis.ipynb
│   └── 04_aggregated_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── io.py
│   ├── cleaning_metadata.py
│   ├── cleaning_signals.py
│   ├── plot_settings.py
│   ├── metadata.py
│   ├── signals.py
│   └── plots.py
│
├── figures/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore

--------------------------------------------------
## Installation
Clone the repository:
git clone https://github.com/giulianaparadiso99/tesi-seismic-analysis.git
cd tesi-seismic-analysis

Install the required Python libraries:
pip install -r requirements.txt

Install the project in editable mode:
pip install -e .

--------------------------------------------------
## Data
The dataset is not included in the repository due to size constraints.
Place the data archive in the following location:
data/raw/query.zip

The raw dataset should contain the .ASC files used in the analysis.

--------------------------------------------------
## Usage
The analysis is performed through Jupyter notebooks.
Start Jupyter:
jupyter notebook

Then open the notebooks in the notebooks directory in order:
1. 01_metadata_preprocessing_exploration.ipynb — metadata loading, preprocessing and exploration
2. 02_seismic_signals_preprocessing_exploration.ipynb — signal loading, preprocessing and exploration
3. 03_single_signal_analysis.ipynb — PDF, moment scaling and autocorrelation analysis per signal
4. 04_aggregated_analysis.ipynb — aggregated analysis and comparison across distance groups

Example code snippet:
from src.io import build_metadata, build_accelerations
df_meta = build_metadata("../data/raw/query.zip")
df_acc = build_accelerations("../data/raw/query.zip")

--------------------------------------------------
## Output
Generated figures are saved in:
figures/

Processed datasets are stored in:
data/processed/

--------------------------------------------------
## Requirements
Main Python libraries used in the project:
pandas
numpy
matplotlib
seaborn
scipy
jupyter
contextily
adjustText
pyarrow

--------------------------------------------------
## Author
Giuliana Paradiso
Politecnico di Torino