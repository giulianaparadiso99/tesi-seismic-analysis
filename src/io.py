from pathlib import Path
import zipfile
import pandas as pd
import numpy as np


def _read_asc_files(zip_path):
    """Read all .ASC files from a zip archive and return raw lines per file."""
    zip_path = Path(zip_path)
    files_lines = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        asc_files = [f for f in z.namelist() if f.endswith(".ASC")]
        for fname in asc_files:
            with z.open(fname) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            files_lines[fname] = lines
    return files_lines


def build_metadata(zip_path):
    """
    Reads a zip archive containing .ASC files and returns
    a dataframe with metadata (df_meta).
    """
    files_lines = _read_asc_files(zip_path)
    meta_rows = []
    for fname, lines in files_lines.items():
        kv_lines = [l.strip() for l in lines if ":" in l]
        meta_dict = {"file": fname}
        for l in kv_lines:
            key, value = l.split(":", 1)
            meta_dict[key.strip()] = value.strip()
        meta_rows.append(meta_dict)
    return pd.DataFrame(meta_rows)


def build_accelerations(zip_path):
    """
    Reads a zip archive containing .ASC files and returns
    a dataframe with acceleration data (df_acc).
    """
    files_lines = _read_asc_files(zip_path)
    acc_chunks = []
    for fname, lines in files_lines.items():
        num_lines = [l.strip() for l in lines if ":" not in l and l.strip()]
        try:
            num_values = np.array(num_lines, dtype=float)
        except ValueError:
            continue
        acc_chunks.append(pd.DataFrame({
            "file": fname,
            "sample": np.arange(len(num_values)),
            "acceleration": num_values
        }))
    return pd.concat(acc_chunks, ignore_index=True) if acc_chunks else pd.DataFrame(columns=["file", "sample", "acceleration"])


def build_dataframes(zip_path):
    """
    Reads a zip archive containing .ASC files and returns
    two dataframes: df_meta and df_acc.
    Kept for backwards compatibility.
    """
    return build_metadata(zip_path), build_accelerations(zip_path)