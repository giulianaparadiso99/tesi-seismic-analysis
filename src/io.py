from pathlib import Path
import zipfile
import pandas as pd
import numpy as np


def build_dataframes(zip_path):
    """
    Legge un archivio zip contenente file .ASC e restituisce
    due dataframe:
        df_meta -> metadati
        df_acc  -> accelerazioni
    """

    zip_path = Path(zip_path)

    meta_rows = []
    acc_rows = []

    with zipfile.ZipFile(zip_path, "r") as z:

        asc_files = [f for f in z.namelist() if f.endswith(".ASC")]

        for fname in asc_files:

            with z.open(fname) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()

            # ----------------
            # METADATI
            # ----------------

            kv_lines = [l.strip() for l in lines if ":" in l]

            meta_dict = {"file": fname}

            for l in kv_lines:
                key, value = l.split(":", 1)
                meta_dict[key.strip()] = value.strip()

            meta_rows.append(meta_dict)

            # ----------------
            # ACCELERAZIONI
            # ----------------

            num_lines = [l.strip() for l in lines if ":" not in l and l.strip()]

            try:
                num_values = np.array(num_lines, dtype=float)
            except:
                continue

            for i, val in enumerate(num_values):
                acc_rows.append({
                    "file": fname,
                    "sample": i,
                    "acceleration": val
                })

    df_meta = pd.DataFrame(meta_rows)
    df_acc = pd.DataFrame(acc_rows)

    return df_meta, df_acc