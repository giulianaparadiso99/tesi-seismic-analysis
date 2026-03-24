"""
latex_export.py
---------------
Utility functions for exporting analysis results to LaTeX format.
Each function takes a pandas DataFrame and returns a string containing
the LaTeX row data for a longtable environment.

Usage in notebook:
    from src.latex_export import heavy_tail_to_latex
    latex_rows = heavy_tail_to_latex(df_heavy_tail_results)
"""


# ===============================================================================================
# ======================================= Helpers ===============================================
# ===============================================================================================

def _format_aic(val):
    """
    Format an AIC value as a LaTeX-compatible string.
    Negative values are rendered with a proper LaTeX minus sign ($-$)
    to avoid the short hyphen that plain text would produce.

    Parameters
    ----------
    val : float

    Returns
    -------
    str
    """
    if val < 0:
        return f"$-${abs(val):,.2f}"
    else:
        return f"{val:,.2f}"


def _best_fit_label(label):
    """
    Convert a best-fit model label from the DataFrame into a
    LaTeX-formatted string.

    Parameters
    ----------
    label : str
        One of 'Levy-stable', 'Student-t', 'Gaussian', 'Laplace'

    Returns
    -------
    str
    """
    mapping = {
        "Levy-stable": r"L\'evy-stable",
        "Student-t":   r"Student-$t$",
        "Gaussian":    "Gaussian",
        "Laplace":     "Laplace",
    }
    return mapping.get(label, label)

# ===============================================================================================
# ========================== Correlation differences table ======================================
# ===============================================================================================

def corr_diff_to_latex(df, output_path=None):
    """
    Generate a LaTeX table from the significant correlation differences
    DataFrame produced by the Fisher z-test analysis in Notebook 1.

    The output is a self-contained table environment (not a longtable,
    since the number of significant pairs is typically small) ready to
    be pasted directly into the Overleaf document.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the following columns:
            'Comparison' : str   — group pair label (e.g. 'Near vs Mid')
            'Variable 1' : str   — first metadata variable
            'Variable 2' : str   — second metadata variable
            'Corr. diff.': float — difference in correlation coefficients
            'p-value'    : float — p-value from the Fisher z-test
    output_path : str or None
        If provided, the LaTeX string is also saved to this path.

    Returns
    -------
    str : complete LaTeX table environment as a string
    """
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"{row['Comparison']} & {row['Variable 1']} & "
            f"{row['Variable 2']} & {row['Corr. diff.']} & "
            f"{row['p-value']} \\\\"
        )

    latex_str = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\caption{Statistically significant correlation differences "
        r"by distance group ($p < 0.05$)}" + "\n"
        r"\label{tab:corr_diff}" + "\n"
        r"\begin{tabular}{lllrr}" + "\n"
        r"\toprule" + "\n"
        r"Comparison & Variable 1 & Variable 2 & Corr.\ diff. & $p$-value \\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        + r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"Saved to: {output_path}")

    return latex_str

# ===============================================================================================
# ========================== Preprocessing quality checks table =================================
# ===============================================================================================

def preprocess_checks_to_latex(rows, output_path=None):
    """
    Generate a LaTeX table from the post-preprocessing quality check results
    produced in Notebook 2.

    Parameters
    ----------
    rows : list of list of str
        Each inner list contains five string elements corresponding to the
        table columns: Check, Single, Aggregated, Expected, Pass.
        Example row: ['Files retained', '66 / 66', '48 / 66', '--', '--']
    output_path : str or None
        If provided, the LaTeX string is also saved to this path.

    Returns
    -------
    str : complete LaTeX table environment as a string
    """
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)

    latex_str = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\begin{tabular}{lllll}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Check} & \textbf{Single} & \textbf{Aggregated} & "
        r"\textbf{Expected} & \textbf{Pass} \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        + r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Post-preprocessing quality checks for the single signal "
        r"and aggregated pipelines.}" + "\n"
        r"\label{tab:postcheck}" + "\n"
        r"\end{table}"
    )

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"Saved to: {output_path}")

    return latex_str


# ===============================================================================================
# ========================== Heavy-tail assessment table ========================================
# ===============================================================================================

def heavy_tail_to_latex(df, output_path=None):
    """
    Generate the row data for a LaTeX longtable from the heavy-tail
    assessment results DataFrame.

    The output contains only the table rows (no header, no footer),
    ready to be pasted inside the longtable environment defined in
    the appendix .tex file. A blank \\addlinespace is inserted between
    different stations for readability.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the following columns:
            station         : str  — station code
            stream          : str  — stream component (e.g. HNE, HNN, HNZ)
            aic_levy_stable : float — AIC for the Levy-stable fit
            aic_student_t   : float — AIC for the Student-t fit
            best_fit_aic    : str  — winning model label
            student_t_df    : float — Student-t degrees of freedom (nu)
            power_law_exp   : float — Hill estimator power-law exponent
    output_path : str or None
        If provided, the LaTeX string is also saved to this path.

    Returns
    -------
    str : LaTeX row data as a single string
    """
    lines = []
    current_station = None

    for _, row in df.iterrows():
        # Insert a small vertical space between station groups
        if row['station'] != current_station:
            if current_station is not None:
                lines.append(r"\addlinespace")
            current_station = row['station']

        aic_levy = _format_aic(row['aic_levy_stable'])
        aic_st   = _format_aic(row['aic_student_t'])
        best     = _best_fit_label(row['best_fit_aic'])
        nu       = f"{row['student_t_df']:.4f}"
        alpha    = f"{row['power_law_exp']:.4f}"

        line = (
            f"{row['station']} & {row['stream']} & "
            f"{aic_levy} & {aic_st} & {best} & "
            f"{nu} & {alpha} \\\\"
        )
        lines.append(line)

    latex_str = "\n".join(lines)

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"Saved to: {output_path}")

    return latex_str