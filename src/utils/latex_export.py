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
# ========================== Metadata header table ==============================================
# ===============================================================================================

def metadata_table_to_latex(df_meta, output_path=None):
    """
    Generate a LaTeX longtable describing metadata fields, their meaning,
    and whether they are constant across files.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata DataFrame (one row per record).
    output_path : str or None
        If provided, the LaTeX string is also saved to this path.

    Returns
    -------
    str : complete LaTeX longtable environment
    """ 
    descriptions = {
        'file': 'File name',
        'EVENT_NAME': 'Event name',
        'EVENT_ID': 'Unique event identifier',
        'EVENT_DATE_YYYYMMDD': 'Event date (YYYYMMDD)',
        'EVENT_TIME_HHMMSS': 'UTC event time',
        'EVENT_LATITUDE_DEGREE': 'Hypocentral latitude',
        'EVENT_LONGITUDE_DEGREE': 'Hypocentral longitude',
        'EVENT_DEPTH_KM': 'Hypocentral depth (km)',
        'HYPOCENTER_REFERENCE': 'Source of hypocentral location',
        'MAGNITUDE_W': r'Moment magnitude $M_w$',
        'MAGNITUDE_W_REFERENCE': r'Source of $M_w$ estimate',
        'MAGNITUDE_L': r'Local magnitude $M_L$',
        'MAGNITUDE_L_REFERENCE': r'Source of $M_L$ estimate',
        'FOCAL_MECHANISM': 'Focal mechanism (fault type)',
        'NETWORK': 'Network code',
        'STATION_CODE': 'Station identifier',
        'STATION_NAME': 'Station name',
        'STATION_LATITUDE_DEGREE': 'Station latitude',
        'STATION_LONGITUDE_DEGREE': 'Station longitude',
        'STATION_ELEVATION_M': 'Station elevation (m a.s.l.)',
        'LOCATION': 'Sub-location code',
        'SENSOR_DEPTH_M': 'Sensor depth below ground (m)',
        'VS30_M/S': 'Average shear-wave velocity in top 30 m',
        'SITE_CLASSIFICATION_EC8': 'EC8 site class',
        'MORPHOLOGIC_CLASSIFICATION': 'Morphological site classification',
        'EPICENTRAL_DISTANCE_KM': 'Epicentral distance (km)',
        'EARTHQUAKE_BACKAZIMUTH_DEGREE': r'Back-azimuth ($^\circ$)',
        'DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS': 'Timestamp of first sample',
        'DATE_TIME_FIRST_SAMPLE_PRECISION': 'Precision of first sample timestamp',
        'SAMPLING_INTERVAL_S': 'Sampling interval (s)',
        'NDATA': 'Number of recorded samples',
        'DURATION_S': 'Recording duration (s)',
        'STREAM': 'Channel code (orientation)',
        'UNITS': r'Data units (cm/s$^2$)',
        'INSTRUMENT': 'Instrument type code',
        'INSTRUMENT_ANALOG/DIGITAL': 'Analog or digital',
        'INSTRUMENTAL_FREQUENCY_HZ': 'Natural frequency (Hz)',
        'INSTRUMENTAL_DAMPING': 'Critical damping ratio',
        'FULL_SCALE_G': 'Full-scale range (g)',
        'N_BIT_DIGITAL_CONVERTER': 'ADC resolution (bit)',
        'PGA_CM/S^2': 'Peak Ground Acceleration',
        'TIME_PGA_S': 'Time of PGA from start (s)',
        'BASELINE_CORRECTION': 'Baseline correction type',
        'FILTER_TYPE': 'Filter type (Butterworth)',
        'FILTER_ORDER': 'Filter order',
        'LOW_CUT_FREQUENCY_HZ': 'Low-cut frequency (Hz)',
        'HIGH_CUT_FREQUENCY_HZ': 'High-cut frequency (Hz)',
        'LATE/NORMAL_TRIGGERED': 'Trigger type (NT/LT)',
        'DATABASE_VERSION': 'Database version',
        'HEADER_FORMAT': 'Header format version',
        'DATA_TYPE': 'Data type (acceleration)',
        'DATA_LICENSE': 'Data license',
        'PROCESSING': 'Processing information',
        'DATA_TIMESTAMP_YYYYMMDD_HHMMSS': 'Data timestamp',
        'DATA_LICENSE': 'Data license',
        'DATA_CITATION': 'Bibliographic citation',
        'DATA_CREATOR': 'Data creator',
        'ORIGINAL_DATA_MEDIATOR_CITATION': 'Citation for the original data mediator',
        'ORIGINAL_DATA_MEDIATOR': 'Original data mediator',
        'ORIGINAL_DATA_CREATOR_CITATION': 'Citation for the original data creator',
        'ORIGINAL_DATA_CREATOR': 'Original data creator',
        'USER1': 'Free annotation field',
        'USER2': 'Free annotation field',
        'USER3': 'Free annotation field',
        'USER4': 'Free annotation field',
        'USER5': 'Free annotation field',
    }
    # Build rows
    rows = []

    for col in df_meta.columns:
        if col == 'file':
            continue

        desc = descriptions.get(col, col)

        # Check constancy
        is_constant = df_meta[col].nunique(dropna=False) == 1
        const_str = 'Yes' if is_constant else 'No'
        field_name = col.replace('_', ' ')

        rows.append(f"{field_name} & {desc} & {const_str} \\\\")

    body = "\n".join(rows)

    # Full LaTeX longtable
    latex_str = (
        r"\begin{longtable}{" + "\n"
        r"  >{\raggedright\arraybackslash}p{4cm}"
        r"  >{\raggedright\arraybackslash}p{8cm}"
        r"  >{\centering\arraybackslash}p{2cm}" + "\n"
        r"}" + "\n"
        r"\caption{Summary of header fields, their description, and constancy across files.}" + "\n"
        r"\label{tab:metadata_fields} \\" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Field} & \textbf{Description} & \textbf{Constant} \\" + "\n"
        r"\midrule" + "\n"
        r"\endfirsthead" + "\n\n"
        r"\multicolumn{3}{c}{\tablename~\thetable{} -- continued from previous page} \\" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Field} & \textbf{Description} & \textbf{Constant} \\" + "\n"
        r"\midrule" + "\n"
        r"\endhead" + "\n\n"
        r"\midrule" + "\n"
        r"\multicolumn{3}{r}{\textit{Continued on next page}} \\" + "\n"
        r"\endfoot" + "\n\n"
        r"\bottomrule" + "\n"
        r"\endlastfoot" + "\n\n"
        + body + "\n"
        + r"\end{longtable}"
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

def onset_detection_to_latex(df_onsets_full, coda_method='rautian', 
                             output_path=None, caption=None, label=None):
    """
    Convert onset detection results to LaTeX table format.
    
    Creates a professional table showing detected P, S, and coda onset times
    with comparison to theoretical arrivals and crustal velocities.
    
    Parameters
    ----------
    df_onsets_full : pd.DataFrame
        Full onset detection results with columns:
        - STATION_CODE
        - COMPONENT (or STREAM)
        - EPICENTRAL_DISTANCE_KM
        - vp_crust, vs_crust
        - t_p_theo, t_p_detected, p_residual
        - t_s_theo, t_s_detected, s_residual
        - t_coda_rautian, t_coda_arias, t_coda_envelope
        - s_duration_rautian, s_duration_arias, s_duration_envelope
    coda_method : str, optional
        Which coda method to display: 'rautian', 'arias', or 'envelope'
        (default: 'rautian')
    output_path : str or Path, optional
        If provided, save LaTeX code to this file
    caption : str, optional
        Table caption (default: auto-generated)
    label : str, optional
        LaTeX label for cross-referencing (default: 'tab:onset_detection')
    
    Returns
    -------
    latex_str : str
        LaTeX longtable code
    
    Examples
    --------
    >>> # Use Rautian method (default)
    >>> latex = onset_detection_to_latex(
    ...     df_onsets_full,
    ...     output_path='tables/onset_detection.tex'
    ... )
    >>> 
    >>> # Use Arias method instead
    >>> latex = onset_detection_to_latex(
    ...     df_onsets_full,
    ...     coda_method='arias',
    ...     output_path='tables/onset_detection_arias.tex'
    ... )
    """
    from pathlib import Path
    
    # Validate coda_method
    valid_methods = ['rautian', 'arias', 'envelope']
    if coda_method not in valid_methods:
        raise ValueError(
            f"coda_method must be one of {valid_methods}, got '{coda_method}'"
        )
    
    # Determine component column name
    comp_col = 'COMPONENT' if 'COMPONENT' in df_onsets_full.columns else 'STREAM'
    
    # Select columns
    t_coda_col = f't_coda_{coda_method}'
    s_duration_col = f's_duration_{coda_method}'
    
    required_cols = [
        'STATION_CODE', comp_col, 'EPICENTRAL_DISTANCE_KM',
        'vp_crust', 'vs_crust',
        't_p_theo', 't_p_detected', 'p_residual',
        't_s_theo', 't_s_detected', 's_residual',
        t_coda_col, s_duration_col
    ]
    
    # Check if required columns exist
    missing = [col for col in required_cols if col not in df_onsets_full.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Prepare data
    table = df_onsets_full[required_cols].copy()
    
    # Sort by station then component
    table = table.sort_values(['STATION_CODE', comp_col]).reset_index(drop=True)
    
    # Round values
    table['EPICENTRAL_DISTANCE_KM'] = table['EPICENTRAL_DISTANCE_KM'].round(1)
    table['vp_crust'] = table['vp_crust'].round(2)
    table['vs_crust'] = table['vs_crust'].round(2)
    table['t_p_theo'] = table['t_p_theo'].round(2)
    table['t_p_detected'] = table['t_p_detected'].round(2)
    table['p_residual'] = table['p_residual'].round(2)
    table['t_s_theo'] = table['t_s_theo'].round(2)
    table['t_s_detected'] = table['t_s_detected'].round(2)
    table['s_residual'] = table['s_residual'].round(2)
    table[t_coda_col] = table[t_coda_col].round(2)
    table[s_duration_col] = table[s_duration_col].round(2)
    
    # Statistics for caption
    n_records = len(table)
    n_stations = table['STATION_CODE'].nunique()
    dist_min = table['EPICENTRAL_DISTANCE_KM'].min()
    dist_max = table['EPICENTRAL_DISTANCE_KM'].max()
    vp_median = table['vp_crust'].median()
    vs_median = table['vs_crust'].median()
    
    # Method name for caption
    method_names = {
        'rautian': 'Rautian \\& Khalturin (1978) lapse-time criterion',
        'arias': 'Arias Intensity D5-75 threshold (Lanzano et al., 2019)',
        'envelope': 'envelope decay to 30\\% of peak (Boore \\& Bommer, 2005)'
    }
    method_name = method_names[coda_method]
    
    # Default caption and label
    if caption is None:
        caption = (
            f"P and S wave onset detection results for {n_stations} stations "
            f"({n_records} three-component recordings). "
            f"Theoretical arrivals calculated using CRUST1.0 crustal velocities "
            f"(median: $v_P = {vp_median:.2f}$ km/s, $v_S = {vs_median:.2f}$ km/s). "
            f"Detected times from AR-AIC method (Leonard \\& Kennett, 1999). "
            f"Coda onset identified using {method_name}. "
            f"Residuals computed as $\\Delta t = t_{{\\text{{det}}}} - t_{{\\text{{theo}}}}$. "
            f"S-wave duration: $\\Delta_S = t_{{\\text{{coda}}}} - t_{{S,\\text{{det}}}}$."
        )
    
    if label is None:
        label = 'tab:onset_detection'
    
    # Build LaTeX longtable
    latex_lines = [
        r'\begin{longtable}{llrcccccccccc}',
        r'    \caption{' + caption + r'} \\',
        r'    \label{' + label + r'} \\',
        r'    \toprule',
        r'    \multirow{2}{*}{Station} & \multirow{2}{*}{Comp.} & \multirow{2}{*}{Dist.} & ',
        r'    \multicolumn{2}{c}{Velocity} & \multicolumn{3}{c}{P-wave} & ',
        r'    \multicolumn{3}{c}{S-wave} & \multirow{2}{*}{$t_{\text{coda}}$} & ',
        r'    \multirow{2}{*}{$\Delta_S$} \\',
        r'    \cmidrule(lr){4-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}',
        r'    & & (km) & $v_P$ & $v_S$ & $t_{\text{theo}}$ & $t_{\text{det}}$ & ',
        r'    $\Delta t$ & $t_{\text{theo}}$ & $t_{\text{det}}$ & $\Delta t$ & (s) & (s) \\',
        r'    & & & (km/s) & (km/s) & (s) & (s) & (s) & (s) & (s) & (s) & & \\',
        r'    \midrule',
        r'    \endfirsthead',
        r'',
        r'    \multicolumn{13}{c}{\tablename\ \thetable{} -- continued from previous page} \\',
        r'    \toprule',
        r'    \multirow{2}{*}{Station} & \multirow{2}{*}{Comp.} & \multirow{2}{*}{Dist.} & ',
        r'    \multicolumn{2}{c}{Velocity} & \multicolumn{3}{c}{P-wave} & ',
        r'    \multicolumn{3}{c}{S-wave} & \multirow{2}{*}{$t_{\text{coda}}$} & ',
        r'    \multirow{2}{*}{$\Delta_S$} \\',
        r'    \cmidrule(lr){4-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}',
        r'    & & (km) & $v_P$ & $v_S$ & $t_{\text{theo}}$ & $t_{\text{det}}$ & ',
        r'    $\Delta t$ & $t_{\text{theo}}$ & $t_{\text{det}}$ & $\Delta t$ & (s) & (s) \\',
        r'    & & & (km/s) & (km/s) & (s) & (s) & (s) & (s) & (s) & (s) & & \\',
        r'    \midrule',
        r'    \endhead',
        r'',
        r'    \midrule',
        r'    \multicolumn{13}{r}{\textit{Continued on next page}} \\',
        r'    \endfoot',
        r'',
        r'    \bottomrule',
        r'    \endlastfoot',
        r'',
    ]
    
    # Add data rows with visual grouping by station
    current_station = None
    
    for idx, row in table.iterrows():
        station = row['STATION_CODE']
        component = row[comp_col]
        
        # Add spacing between stations
        if station != current_station:
            if current_station is not None:
                latex_lines.append(r'    \addlinespace')
            current_station = station
        
        # Format residuals with sign
        p_res = row['p_residual']
        s_res = row['s_residual']
        p_res_str = f"+{p_res:.2f}" if p_res >= 0 else f"{p_res:.2f}"
        s_res_str = f"+{s_res:.2f}" if s_res >= 0 else f"{s_res:.2f}"
        
        line = (
            f"    {station} & {component} & "
            f"{row['EPICENTRAL_DISTANCE_KM']:.1f} & "
            f"{row['vp_crust']:.2f} & "
            f"{row['vs_crust']:.2f} & "
            f"{row['t_p_theo']:.2f} & "
            f"{row['t_p_detected']:.2f} & "
            f"{p_res_str} & "
            f"{row['t_s_theo']:.2f} & "
            f"{row['t_s_detected']:.2f} & "
            f"{s_res_str} & "
            f"{row[t_coda_col]:.2f} & "
            f"{row[s_duration_col]:.2f} \\\\"
        )
        latex_lines.append(line)
    
    # Close table
    latex_lines.append(r'\end{longtable}')
    
    latex_str = '\n'.join(latex_lines)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved: {output_path}")
        print(f"  Method used: {coda_method}")
        print(f"  Rows: {n_records} (from {n_stations} stations)")
    
    return latex_str

def coda_onset_comparison_to_latex(df_onsets_full, output_path=None):
    """
    Generate LaTeX longtable with coda onset times for all methods.
    
    Creates a table with one row per (station, component) showing
    detected coda onset times for Rautian, Arias, and Envelope methods.
    
    Parameters
    ----------
    df_onsets_full : pd.DataFrame
        Must contain columns:
        - STATION_CODE
        - COMPONENT (or STREAM)
        - t_s_detected
        - t_coda_rautian
        - t_coda_arias
        - t_coda_envelope
        - s_duration_rautian
        - s_duration_arias
        - s_duration_envelope
    output_path : str or Path, optional
        If provided, save LaTeX code to this file
    
    Returns
    -------
    str
        LaTeX longtable code
    
    Examples
    --------
    >>> latex = coda_onset_comparison_to_latex(
    ...     df_onsets_full,
    ...     output_path='tables/coda_comparison.tex'
    ... )
    """
    from pathlib import Path
    
    # Determine component column name
    comp_col = 'COMPONENT' if 'COMPONENT' in df_onsets_full.columns else 'STREAM'
    
    # Select and sort data
    table = df_onsets_full[[
        'STATION_CODE',
        comp_col,
        't_s_detected',
        't_coda_rautian',
        't_coda_arias',
        't_coda_envelope',
        's_duration_rautian',
        's_duration_arias',
        's_duration_envelope'
    ]].copy()
    
    table = table.sort_values(['STATION_CODE', comp_col]).reset_index(drop=True)
    
    # Round values
    for col in table.columns:
        if col not in ['STATION_CODE', comp_col]:
            table[col] = table[col].round(2)
    
    # Build LaTeX
    latex_lines = [
        r'\begin{longtable}{llcccccccc}',
        r'    \caption{Coda onset times detected by three methods: Rautian \& Khalturin (1978) lapse-time criterion, Arias Intensity D5-75 threshold (Lanzano et al., 2019), and envelope decay to 30\% of peak (Boore \& Bommer, 2005). S-wave duration ($\Delta_S$) computed as $t_{\text{coda}} - t_{S,\text{det}}$ for each method.} \\',
        r'    \label{tab:coda_comparison} \\',
        r'    \toprule',
        r'    \multirow{2}{*}{Station} & \multirow{2}{*}{Comp.} & \multirow{2}{*}{$t_{S,\text{det}}$} & \multicolumn{3}{c}{$t_{\text{coda}}$ (s)} & \multicolumn{3}{c}{$\Delta_S$ (s)} \\',
        r'    \cmidrule(lr){4-6} \cmidrule(lr){7-9}',
        r'    & & (s) & Rautian & Arias & Envelope & Rautian & Arias & Envelope \\',
        r'    \midrule',
        r'    \endfirsthead',
        r'',
        r'    \multicolumn{9}{c}{\tablename\ \thetable{} -- continued from previous page} \\',
        r'    \toprule',
        r'    \multirow{2}{*}{Station} & \multirow{2}{*}{Comp.} & \multirow{2}{*}{$t_{S,\text{det}}$} & \multicolumn{3}{c}{$t_{\text{coda}}$ (s)} & \multicolumn{3}{c}{$\Delta_S$ (s)} \\',
        r'    \cmidrule(lr){4-6} \cmidrule(lr){7-9}',
        r'    & & (s) & Rautian & Arias & Envelope & Rautian & Arias & Envelope \\',
        r'    \midrule',
        r'    \endhead',
        r'',
        r'    \midrule',
        r'    \multicolumn{9}{r}{\textit{Continued on next page}} \\',
        r'    \endfoot',
        r'',
        r'    \bottomrule',
        r'    \endlastfoot',
        r'',
    ]
    
    # Add data rows with visual grouping by station
    current_station = None
    
    for idx, row in table.iterrows():
        station = row['STATION_CODE']
        component = row[comp_col]
        
        # Add spacing between stations
        if station != current_station:
            if current_station is not None:
                latex_lines.append(r'    \addlinespace')
            current_station = station
        
        line = (
            f"    {station} & {component} & "
            f"{row['t_s_detected']:.2f} & "
            f"{row['t_coda_rautian']:.2f} & "
            f"{row['t_coda_arias']:.2f} & "
            f"{row['t_coda_envelope']:.2f} & "
            f"{row['s_duration_rautian']:.2f} & "
            f"{row['s_duration_arias']:.2f} & "
            f"{row['s_duration_envelope']:.2f} \\\\"
        )
        latex_lines.append(line)
    
    # Close table
    latex_lines.append(r'\end{longtable}')
    
    latex_str = '\n'.join(latex_lines)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved: {output_path}")
    
    return latex_str