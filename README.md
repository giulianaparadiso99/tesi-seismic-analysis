# Seismic acceleration analysis

This repository contains the code used for the analysis of ground motion recordings in my master's thesis at Politecnico di Torino.

The project processes accelerometric data stored in .ASC files and performs statistical analysis on metadata and acceleration time series.

The goal is to explore statistical properties of seismic signals and their relationship with event and station metadata.

--------------------------------------------------
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
│   ├── 03_signal_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── io.py
│   ├── cleaning.py
│   ├── metadata.py
│   ├── signals.py
│   └── plots.py
│
├── figures/
│
├── scripts/
│   └── generate_figures.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

--------------------------------------------------

## INSTALLATION

Clone the repository:

git clone https://github.com/giulianaparadiso99/tesi-seismic-analysis.git
cd tesi-seismic-analysis

Install the required Python libraries:

pip install -r requirements.txt

--------------------------------------------------

## DATA

The dataset is not included in the repository due to size constraints.

Place the data archive in the following location:

data/raw/query.zip

The raw dataset should contain the .ASC files used in the analysis.

--------------------------------------------------

## USAGE

The analysis is performed through Jupyter notebooks.

Start Jupyter:

jupyter notebook

Then open the notebooks in the notebooks directory.

Typical workflow:

1. Load the dataset
2. Clean metadata
3. Perform statistical analysis
4. Generate figures

Example code snippet:

from src.io import build_dataframes

df_meta, df_acc = build_dataframes("../data/raw/query.zip")

--------------------------------------------------

## OUTPUT

Generated figures are saved in:

figures/

Processed datasets may be stored in:

data/processed/

--------------------------------------------------

## REQUIREMENTS

Main Python libraries used in the project:

pandas  
numpy  
matplotlib  
seaborn  
scipy  
jupyter  

--------------------------------------------------

## AUTHOR

Giuliana Paradiso  
Politecnico di Torino