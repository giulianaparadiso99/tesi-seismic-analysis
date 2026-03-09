from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

def build_dataframes(zip_path):
    """
    Reads a zip archive containing .ASC files and returns
    two dataframes:
        df_meta -> metadata
        df_acc  -> accelerations
    """
    zip_path = Path(zip_path)
    meta_rows = []
    acc_chunks = []

    with zipfile.ZipFile(zip_path, "r") as z:
        asc_files = [f for f in z.namelist() if f.endswith(".ASC")]

        for fname in asc_files:
            with z.open(fname) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()

            # ----------------
            # METADATA
            # ----------------
            kv_lines = [l.strip() for l in lines if ":" in l]
            meta_dict = {"file": fname}
            for l in kv_lines:
                key, value = l.split(":", 1)
                meta_dict[key.strip()] = value.strip()
            meta_rows.append(meta_dict)

            # ----------------
            # ACCELERATIONS
            # ----------------
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

    df_meta = pd.DataFrame(meta_rows)
    df_acc = pd.concat(acc_chunks, ignore_index=True) if acc_chunks else pd.DataFrame(columns=["file", "sample", "acceleration"])

    return df_meta, df_acc