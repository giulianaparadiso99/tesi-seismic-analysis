"""
signals_scaling_ensemble.py
----------------------------
Moment scaling analysis using ensemble (spatial) averaging with temporal windowing.

This module implements windowed ensemble-averaged moment scaling, where seismic
signals are separated into temporal windows (pre-arrival, P-wave, S-wave, coda)
representing different dynamical regimes. Statistical properties are obtained by
averaging over multiple stations (spatial ensemble) at a fixed reference time t₀
per window, rather than averaging over time for individual stations.

Key differences from signals_scaling.py:
    - Temporal windowing: Separates signal into 4 distinct phases
    - Adaptive t₀: Each station has its own t₀ per window (based on wave arrivals)
    - Fixed tau per window: All stations use same tau values for ensemble averaging
    - Per-window scaling: Each window has its own ζ(q) exponents

Main workflow:
    1. Detect temporal windows for all files (window_detection.py)
    2. Compute increments per window with adaptive t₀ and filtered tau
    3. Compute ensemble moments by averaging over stations
    4. Compute scaling exponents per window
    5. Compare ζ(q) across windows to identify regime changes

Theoretical background:
    Non-stationary processes require temporal windowing to separate dynamical
    regimes. Each window should exhibit stationary statistics within itself.
    
    For seismic signals:
        - Pre-arrival: ζ(q) ≈ 0 (instrumental noise, no scaling)
        - P-wave: ζ(q) > 0 with slope a (scaling regime 1)
        - S-wave: ζ(q) > 0 with slope b ≠ a (scaling regime 2)
        - Coda: ζ(q) ≈ 0 (return to noise/weak fluctuations)
    
    Different slopes demonstrate non-stationary multifractal behavior.

Usage:
    from src.window_detection import identify_windows_all_files
    from src.signals_scaling_ensemble import (
        compute_increments_all_windows,
        compute_moments_all_windows,
        compute_exponents_all_windows,
        compute_and_save_all_windowed
    )
    
    # Detect windows
    df_windows = identify_windows_all_files(df_acc)
    
    # Complete analysis
    results = compute_and_save_all_windowed(
        df_acc, df_vel, df_disp, df_windows,
        tau_values, q_values, output_dir
    )
    
    # Access results
    df_zeta_p_wave = results['displacement']['exponents']['p_wave']
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from src.visualization.plot_settings import set_plot_style
colors = set_plot_style()


# ===============================================================================================
# ==================================== Increment Computation ====================================
# ===============================================================================================

def compute_increments_ensemble_windowed(df, df_windows, tau_values, 
                                         window_name='p_wave', 
                                         column='displacement'):
    """
    Compute increments for ensemble averaging with adaptive t₀ per station.
    
    For each station, uses the detected window boundaries to set t₀ (start of window).
    Tau values are filtered to fit the shortest window across all stations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Signal data with columns ['file', 'sample', column]
    df_windows : pd.DataFrame
        Window boundaries from identify_windows_all_files()
        Must have columns: ['file', 'window_name', 'start', 'end', 'n_samples']
    tau_values : list of int
        Time lags (in samples) - will be filtered to fit shortest window
    window_name : str
        Which window to analyze: 'pre_arrival', 'p_wave', 's_wave', 'coda'
    column : str
        Column name of the process ('acceleration', 'velocity', 'displacement')
    
    Returns
    -------
    pd.DataFrame
        Columns: [file, station, stream, tau, increment, t0, window_length]
        One row per (file, tau) combination
        
    Notes
    -----
    - t₀ is adaptive: each station has its own t₀ based on actual wave arrivals
    - tau values are uniform: all stations use the same filtered tau_values
    - tau filtering ensures t₀ + tau stays within window for all stations
    
    Examples
    --------
    >>> df_windows = identify_windows_all_files(df_acc)
    >>> df_inc = compute_increments_ensemble_windowed(
    ...     df_disp, df_windows, tau_values, window_name='p_wave'
    ... )
    >>> # df_inc contains increments from all stations for P-wave window
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    
    inc_rows = []
    
    # =========================================================================
    # STEP 1: Find minimum window length across all files for THIS window
    # =========================================================================
    
    window_lengths = []
    valid_files = []
    
    for file in df['file'].unique():
        window_info = df_windows[
            (df_windows['file'] == file) & 
            (df_windows['window_name'] == window_name)
        ]
        
        if len(window_info) == 0:
            continue
        
        window_length = window_info.iloc[0]['n_samples']
        
        if window_length < 100:
            continue
        
        window_lengths.append(window_length)
        valid_files.append(file)
    
    if len(window_lengths) == 0:
        raise ValueError(f"No valid windows found for '{window_name}'")
    
    min_window_length = min(window_lengths)
    max_window_length = max(window_lengths)
    mean_window_length = np.mean(window_lengths)
    
    print(f"\n  Window '{window_name}' statistics:")
    print(f"    Valid files: {len(valid_files)}")
    print(f"    Length range: [{min_window_length}, {max_window_length}] samples")
    print(f"    Length range: [{min_window_length/200:.1f}, {max_window_length/200:.1f}] s")
    print(f"    Mean length: {mean_window_length:.0f} samples ({mean_window_length/200:.1f} s)")
    
    # =========================================================================
    # STEP 2: Filter tau_values to fit shortest window
    # =========================================================================
    
    tau_max = min_window_length - 1
    tau_values_valid = [t for t in tau_values if t < tau_max]
    
    if len(tau_values_valid) == 0:
        raise ValueError(f"No valid tau values for window '{window_name}' "
                        f"(min_window_length={min_window_length})")
    
    n_removed = len(tau_values) - len(tau_values_valid)
    if n_removed > 0:
        print(f"    Filtered tau: removed {n_removed}/{len(tau_values)} values (> {tau_max})")
        print(f"    Valid tau range: [{min(tau_values_valid)}, {max(tau_values_valid)}] samples")
        print(f"    Valid tau range: [{min(tau_values_valid)/200:.3f}, {max(tau_values_valid)/200:.3f}] s")
    else:
        print(f"    All {len(tau_values)} tau values valid (< {tau_max})")
    
    # =========================================================================
    # STEP 3: Compute increments with adaptive t₀ and uniform tau
    # =========================================================================
    
    for file in valid_files:
        signal = df[df['file'] == file][column].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        
        window_info = df_windows[
            (df_windows['file'] == file) & 
            (df_windows['window_name'] == window_name)
        ].iloc[0]
        
        t0 = window_info['start']
        window_end = window_info['end']
        window_length = window_info['n_samples']
        
        for tau in tau_values_valid:
            if t0 + tau >= window_end:
                print(f"    Warning: tau={tau} exceeds window for {file}")
                continue
            
            increment = signal[t0 + tau] - signal[t0]
            
            inc_rows.append({
                'file': file,
                'station': station,
                'stream': stream,
                'tau': tau,
                'increment': increment,
                't0': t0,
                'window_length': window_length
            })
    
    df_result = pd.DataFrame(inc_rows)
    
    print(f"\n    Increments computed:")
    print(f"      Files: {df_result['file'].nunique()}")
    print(f"      Tau values: {df_result['tau'].nunique()}")
    print(f"      Total increments: {len(df_result):,}")
    print(f"      Increments per tau: {len(df_result) // df_result['tau'].nunique()}")
    
    return df_result


def compute_increments_all_windows(df, df_windows, tau_values, column='displacement'):
    """
    Compute increments for ALL temporal windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Signal data
    df_windows : pd.DataFrame
        Window boundaries for all files
    tau_values : list of int
        Time lags (will be filtered per window)
    column : str
        Signal column name
    
    Returns
    -------
    dict
        {window_name: df_increments}
        
    Examples
    --------
    >>> increments = compute_increments_all_windows(df_disp, df_windows, tau_values)
    >>> df_inc_p = increments['p_wave']
    >>> df_inc_s = increments['s_wave']
    """
    window_names = ['pre_arrival', 'p_wave', 's_wave', 'coda']
    
    increments_dict = {}
    
    for window_name in window_names:
        print(f"\n{'='*70}")
        print(f"Computing increments for: {window_name}")
        print(f"{'='*70}")
        
        try:
            df_inc = compute_increments_ensemble_windowed(
                df, df_windows, tau_values, 
                window_name=window_name, 
                column=column
            )
            increments_dict[window_name] = df_inc
        
        except ValueError as e:
            print(f"  Error: {e}")
            print(f"  Skipping window '{window_name}'")
            increments_dict[window_name] = pd.DataFrame()
    
    return increments_dict


# ===============================================================================================
# ==================================== Moment Computation =======================================
# ===============================================================================================

def compute_moments_from_increments_ensemble(df_increments, q_values):
    """
    Compute q-th order moments from increments using ensemble averaging.
    
    M_q(τ) = ⟨|Δx(τ, t₀)|^q⟩_{stations}
    
    For each (tau, q), average over all stations instead of over all t0.
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        Increments from compute_increments_ensemble_windowed()
        Must have columns: ['file', 'tau', 'increment']
    q_values : list of float
        Moment orders to compute
    
    Returns
    -------
    pd.DataFrame
        Columns: [q, tau, moment, n_stations]
        
    Examples
    --------
    >>> df_inc = compute_increments_ensemble_windowed(df, df_windows, tau_values)
    >>> df_mom = compute_moments_from_increments_ensemble(df_inc, q_values)
    """
    if len(df_increments) == 0:
        return pd.DataFrame(columns=['q', 'tau', 'moment', 'n_stations'])
    
    rows = []
    
    for tau, group in df_increments.groupby('tau'):
        increments = group['increment'].values
        n_stations = len(increments)
        
        for q in q_values:
            moment = np.mean(np.abs(increments) ** q)
            
            rows.append({
                'q': q,
                'tau': tau,
                'moment': moment,
                'n_stations': n_stations
            })
    
    df_result = pd.DataFrame(rows)
    
    if len(df_result) > 0:
        print(f"\n  Moment computation summary (ensemble):")
        print(f"    q values: {df_result['q'].nunique()}")
        print(f"    Tau values: {df_result['tau'].nunique()}")
        print(f"    Stations per tau: {df_result['n_stations'].iloc[0]}")
        print(f"    Total moment values: {len(df_result)}")
    
    return df_result


def compute_moments_all_windows(increments_dict, q_values):
    """
    Compute moments for all windows.
    
    Parameters
    ----------
    increments_dict : dict
        {window_name: df_increments}
    q_values : list of float
        Moment orders
    
    Returns
    -------
    dict
        {window_name: df_moments}
        
    Examples
    --------
    >>> increments = compute_increments_all_windows(df_disp, df_windows, tau_values)
    >>> moments = compute_moments_all_windows(increments, q_values)
    >>> df_mom_p = moments['p_wave']
    """
    moments_dict = {}
    
    for window_name, df_inc in increments_dict.items():
        print(f"\n{'='*70}")
        print(f"Computing moments for: {window_name}")
        print(f"{'='*70}")
        
        df_mom = compute_moments_from_increments_ensemble(df_inc, q_values)
        
        if len(df_mom) > 0:
            df_mom['window'] = window_name
        
        moments_dict[window_name] = df_mom
    
    return moments_dict


# ===============================================================================================
# ==================================== Scaling Exponents ========================================
# ===============================================================================================

def compute_scaling_exponents(df_moments, tau_min=None, tau_max=None):
    """
    Compute scaling exponents ζ(q) from moments.
    
    Fits: M_q(τ) ~ τ^ζ(q)
    In log-log: log(M_q) = ζ(q) * log(τ) + const
    
    Parameters
    ----------
    df_moments : pd.DataFrame
        Moments with columns ['q', 'tau', 'moment']
    tau_min : int, optional
        Minimum tau for fitting range
    tau_max : int, optional
        Maximum tau for fitting range
    
    Returns
    -------
    pd.DataFrame
        Columns: [q, zeta, zeta_err, r_squared, n_points]
        
    Examples
    --------
    >>> df_mom = compute_moments_from_increments_ensemble(df_inc, q_values)
    >>> df_zeta = compute_scaling_exponents(df_mom)
    """
    if len(df_moments) == 0:
        return pd.DataFrame(columns=['q', 'zeta', 'zeta_err', 'r_squared', 'n_points'])
    
    df_fit = df_moments.copy()
    
    if tau_min is not None:
        df_fit = df_fit[df_fit['tau'] >= tau_min]
    if tau_max is not None:
        df_fit = df_fit[df_fit['tau'] <= tau_max]
    
    rows = []
    
    for q, group in df_fit.groupby('q'):
        log_tau = np.log10(group['tau'].values)
        log_moment = np.log10(group['moment'].values)
        
        if len(log_tau) < 3:
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_moment)
        
        rows.append({
            'q': q,
            'zeta': slope,
            'zeta_err': std_err,
            'r_squared': r_value**2,
            'n_points': len(log_tau)
        })
    
    df_result = pd.DataFrame(rows)
    
    if len(df_result) > 0:
        print(f"\n  Scaling exponents computed:")
        print(f"    q values: {len(df_result)}")
        print(f"    Mean R²: {df_result['r_squared'].mean():.4f}")
        print(f"    Min R²: {df_result['r_squared'].min():.4f}")
    
    return df_result


def compute_exponents_all_windows(moments_dict, tau_min=None, tau_max=None):
    """
    Compute scaling exponents for all windows.
    
    Parameters
    ----------
    moments_dict : dict
        {window_name: df_moments}
    tau_min : int, optional
        Minimum tau for fitting range
    tau_max : int, optional
        Maximum tau for fitting range
    
    Returns
    -------
    dict
        {window_name: df_exponents}
        
    Examples
    --------
    >>> moments = compute_moments_all_windows(increments, q_values)
    >>> exponents = compute_exponents_all_windows(moments)
    >>> df_zeta_p = exponents['p_wave']
    """
    exponents_dict = {}
    
    for window_name, df_mom in moments_dict.items():
        print(f"\n{'='*70}")
        print(f"Computing exponents for: {window_name}")
        print(f"{'='*70}")
        
        df_zeta = compute_scaling_exponents(df_mom, tau_min=tau_min, tau_max=tau_max)
        
        if len(df_zeta) > 0:
            df_zeta['window'] = window_name
        
        exponents_dict[window_name] = df_zeta
    
    return exponents_dict


# ===============================================================================================
# ==================================== Save/Load Functions ======================================
# ===============================================================================================

def save_windowed_results(increments_dict, moments_dict, exponents_dict, 
                          base_dir, process_name):
    """
    Save windowed analysis results to disk.
    
    Parameters
    ----------
    increments_dict : dict
        {window_name: df_increments}
    moments_dict : dict
        {window_name: df_moments}
    exponents_dict : dict
        {window_name: df_exponents}
    base_dir : str or Path
        Base directory for saving
    process_name : str
        Process name ('acceleration', 'velocity', 'displacement')
    
    Returns
    -------
    None
    """
    base_dir = Path(base_dir)
    process_dir = base_dir / process_name
    process_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {process_name} results to {process_dir}")
    
    for window_name in increments_dict.keys():
        if len(increments_dict[window_name]) > 0:
            filepath = process_dir / f'increments_{window_name}.parquet'
            increments_dict[window_name].to_parquet(filepath)
            print(f"  Saved: {filepath.name}")
        
        if len(moments_dict[window_name]) > 0:
            filepath = process_dir / f'moments_{window_name}.parquet'
            moments_dict[window_name].to_parquet(filepath)
            print(f"  Saved: {filepath.name}")
        
        if len(exponents_dict[window_name]) > 0:
            filepath = process_dir / f'exponents_{window_name}.parquet'
            exponents_dict[window_name].to_parquet(filepath)
            print(f"  Saved: {filepath.name}")


def load_windowed_results(base_dir, process_name):
    """
    Load windowed analysis results from disk.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory containing results
    process_name : str
        Process name ('acceleration', 'velocity', 'displacement')
    
    Returns
    -------
    dict
        {
            'increments': {window_name: df_increments},
            'moments': {window_name: df_moments},
            'exponents': {window_name: df_exponents}
        }
    """
    base_dir = Path(base_dir)
    process_dir = base_dir / process_name
    
    windows = ['pre_arrival', 'p_wave', 's_wave', 'coda']
    
    results = {
        'increments': {},
        'moments': {},
        'exponents': {}
    }
    
    for window in windows:
        inc_path = process_dir / f'increments_{window}.parquet'
        if inc_path.exists():
            results['increments'][window] = pd.read_parquet(inc_path)
        
        mom_path = process_dir / f'moments_{window}.parquet'
        if mom_path.exists():
            results['moments'][window] = pd.read_parquet(mom_path)
        
        exp_path = process_dir / f'exponents_{window}.parquet'
        if exp_path.exists():
            results['exponents'][window] = pd.read_parquet(exp_path)
    
    print(f"\nLoaded {process_name} results from {process_dir}")
    
    return results


# ===============================================================================================
# ==================================== Complete Pipeline ========================================
# ===============================================================================================

def compute_and_save_all_windowed(df_acc, df_vel, df_disp, df_windows,
                                  tau_values, q_values, output_dir,
                                  tau_min=None, tau_max=None):
    """
    Complete windowed ensemble analysis for all processes.
    
    Computes increments, moments, and scaling exponents for all temporal windows
    and all three processes (acceleration, velocity, displacement).
    
    Parameters
    ----------
    df_acc : pd.DataFrame
        Preprocessed acceleration data
    df_vel : pd.DataFrame
        Velocity data (from integration)
    df_disp : pd.DataFrame
        Displacement data (from integration)
    df_windows : pd.DataFrame
        Window boundaries from identify_windows_all_files()
    tau_values : list of int
        Time lags (will be filtered per window)
    q_values : list of float
        Moment orders
    output_dir : str or Path
        Base directory for saving results
    tau_min : int, optional
        Minimum tau for exponent fitting
    tau_max : int, optional
        Maximum tau for exponent fitting
    
    Returns
    -------
    dict
        Nested dictionary:
        {
            'acceleration': {
                'increments': {window: df},
                'moments': {window: df},
                'exponents': {window: df}
            },
            'velocity': {...},
            'displacement': {...}
        }
        
    Examples
    --------
    >>> from src.window_detection import identify_windows_all_files
    >>> 
    >>> # Detect windows
    >>> df_windows = identify_windows_all_files(df_acc)
    >>> 
    >>> # Complete analysis
    >>> results = compute_and_save_all_windowed(
    ...     df_acc, df_vel, df_disp, df_windows,
    ...     tau_values, q_values,
    ...     output_dir='../data/processed/04b_ensemble'
    ... )
    >>> 
    >>> # Access results
    >>> df_zeta_p_disp = results['displacement']['exponents']['p_wave']
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processes = {
        'acceleration': df_acc,
        'velocity': df_vel,
        'displacement': df_disp
    }
    
    results = {}
    
    for process_name, df in processes.items():
        print(f"\n{'#'*80}")
        print(f"# Processing: {process_name.upper()}")
        print(f"{'#'*80}")
        
        increments = compute_increments_all_windows(df, df_windows, tau_values, 
                                                    column=process_name)
        
        moments = compute_moments_all_windows(increments, q_values)
        
        exponents = compute_exponents_all_windows(moments, tau_min=tau_min, 
                                                 tau_max=tau_max)
        
        save_windowed_results(increments, moments, exponents, 
                            output_dir, process_name)
        
        results[process_name] = {
            'increments': increments,
            'moments': moments,
            'exponents': exponents
        }
    
    print(f"\n{'#'*80}")
    print(f"# Analysis complete!")
    print(f"# Results saved to: {output_dir}")
    print(f"{'#'*80}")
    
    return results


# ===============================================================================================
# ==================================== Validation ===============================================
# ===============================================================================================

def validate_moments_ensemble(df_moments, expected_n_stations=None):
    """
    Validate ensemble-averaged moments.
    
    Parameters
    ----------
    df_moments : pd.DataFrame
        Moments from compute_moments_from_increments_ensemble()
        Must have columns: ['q', 'tau', 'moment', 'n_stations']
    expected_n_stations : int, optional
        Expected number of stations in ensemble
    
    Returns
    -------
    bool
        True if validation passes
    """
    if len(df_moments) == 0:
        print("Warning: Empty moments DataFrame, skipping validation")
        return True
    
    print("Validating ensemble moments...")
    
    n_stations_unique = df_moments['n_stations'].unique()
    print(f"  Ensemble size(s): {n_stations_unique}")
    
    if expected_n_stations is not None:
        n_stations = n_stations_unique[0]
        assert n_stations == expected_n_stations, \
            f"Expected {expected_n_stations} stations, got {n_stations}"
        print(f"  Ensemble size matches expected: {n_stations} stations")
    
    assert df_moments['moment'].isna().sum() == 0, "NaN found in moments"
    print(f"  No NaN in moments")
    
    assert np.isinf(df_moments['moment']).sum() == 0, "Inf found in moments"
    print(f"  No Inf in moments")
    
    assert (df_moments['moment'] > 0).all(), "Non-positive moments found"
    print(f"  All moments positive")
    
    for tau in df_moments['tau'].unique()[:5]:
        df_tau = df_moments[df_moments['tau'] == tau].sort_values('q')
        moments = df_tau['moment'].values
        if len(moments) > 1:
            assert np.all(np.diff(moments) > 0), \
                f"Moments not monotonic in q at tau={tau}"
    print(f"  Moments monotonic in q")
    
    print("All validation checks passed!\n")
    return True


def analyze_increments_ensemble(df_increments, output_dir=None):
    """
    Analyze ensemble increment statistics.
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        Increments from compute_increments_ensemble_windowed()
    output_dir : str or Path, optional
        Directory to save diagnostic plots
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per tau across ensemble
    """
    if len(df_increments) == 0:
        return pd.DataFrame()
    
    summary_rows = []
    
    for tau, group in df_increments.groupby('tau'):
        increments = group['increment'].values
        
        summary_rows.append({
            'tau': tau,
            'n_stations': len(increments),
            'mean': np.mean(increments),
            'std': np.std(increments),
            'median': np.median(increments),
            'min': np.min(increments),
            'max': np.max(increments),
            'skewness': stats.skew(increments),
            'kurtosis': stats.kurtosis(increments),
        })
    
    df_summary = pd.DataFrame(summary_rows)
    
    print("Ensemble increment statistics summary:")
    print(f"  Tau range: {df_summary['tau'].min()} - {df_summary['tau'].max()}")
    print(f"  Ensemble size: {df_summary['n_stations'].iloc[0]} stations")
    print(f"  Mean across all tau: {df_summary['mean'].mean():.6f}")
    print(f"  Std range: [{df_summary['std'].min():.6f}, {df_summary['std'].max():.6f}]")
    print()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].semilogx(df_summary['tau'], df_summary['mean'], 'o-')
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('τ (samples)')
        axes[0, 0].set_ylabel('Mean increment')
        axes[0, 0].set_title('Ensemble Mean vs τ')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].loglog(df_summary['tau'], df_summary['std'], 'o-')
        axes[0, 1].set_xlabel('τ (samples)')
        axes[0, 1].set_ylabel('Std increment')
        axes[0, 1].set_title('Ensemble Std vs τ')
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].semilogx(df_summary['tau'], df_summary['skewness'], 'o-')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('τ (samples)')
        axes[1, 0].set_ylabel('Skewness')
        axes[1, 0].set_title('Ensemble Skewness vs τ')
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].semilogx(df_summary['tau'], df_summary['kurtosis'], 'o-')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5, label='Gaussian')
        axes[1, 1].set_xlabel('τ (samples)')
        axes[1, 1].set_ylabel('Excess kurtosis')
        axes[1, 1].set_title('Ensemble Kurtosis vs τ')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'increment_diagnostics_ensemble.pdf')
        plt.close()
        
        print(f"Saved diagnostics to {output_dir / 'increment_diagnostics_ensemble.pdf'}")
    
    return df_summary
