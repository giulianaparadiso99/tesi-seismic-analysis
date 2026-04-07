"""
Moment scaling analysis using ensemble (spatial) averaging.

This module implements ensemble-averaged moment scaling, where statistical
properties are obtained by averaging over multiple stations (spatial ensemble)
at a fixed reference time t₀, rather than averaging over time for individual
stations.

This approach allows testing the ergodic hypothesis by comparing time-averaged
(signals_scaling.py) and ensemble-averaged scaling exponents.

Main differences from signals_scaling.py:
    - compute_increments_ensemble: Uses fixed t₀ (default: 0) instead of looping
    - compute_moments_from_increments_ensemble: Averages over stations, not over t₀
    - validate_moments_ensemble: Checks ensemble size consistency
    - analyze_increments_ensemble: Statistics across spatial ensemble

Theoretical background:
    Ergodic hypothesis states that time-averaged and ensemble-averaged
    statistical properties should coincide for ergodic processes:
    
    ⟨f⟩_time = ⟨f⟩_ensemble
    
    For moment scaling: ζ(q)_time ≈ ζ(q)_ensemble

Usage:
    from src.signals_scaling_ensemble import (
        compute_increments_ensemble,
        compute_moments_from_increments_ensemble
    )
    
    # Ensemble averaging with t₀=0
    df_inc = compute_increments_ensemble(df, tau_values, t0=0)
    df_mom = compute_moments_from_increments_ensemble(df_inc, q_values)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ==================================== Increment Computation ====================================
# ===============================================================================================

def compute_increments_ensemble(df, tau_values, t0=0, column='displacement'):
    
    """
    Compute increments Δx(τ, t₀) = x(t₀ + τ) - x(t₀) for ensemble averaging.
    
    For each station, compute ONE increment at fixed t0 for each tau.
    This is used for spatial (ensemble) averaging instead of temporal averaging.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns ['file', column]
    tau_values : list of int
        Time lags (in samples)
    t0 : int, default=0
        Fixed reference time (same for all stations)
        Default is 0 (start of recording)
    column : str
        Column name of the process ('acceleration', 'velocity', 'displacement')
    
    Returns
    -------
    pd.DataFrame
        Columns: [file, station, stream, tau, increment]
        
    Notes
    -----
    Unlike compute_increments() which loops over all t0 values, this function
    uses a single fixed t0 for all stations, enabling ensemble (spatial) averaging.
    
    Examples
    --------
    >>> df = integrate_to_displacement(df_acc_event)
    >>> df_inc = compute_increments_ensemble(df, tau_values, t0=0, column='displacement')
    >>> print(df_inc.head())
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    
    # Find minimum signal length to determine valid tau range
    min_length = df.groupby('file')['sample'].max().min() + 1
    tau_values_valid = [t for t in tau_values if t0 + t < min_length]
    
    if len(tau_values_valid) < len(tau_values):
        print(f"Warning: Reduced tau range from {len(tau_values)} to {len(tau_values_valid)} "
              f"values due to signal length constraint (min_length={min_length})")
    
    inc_rows = []
    
    for file in df['file'].unique():
        signal = df[df['file'] == file][column].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        
        N = len(signal)
        
        for tau in tau_values_valid:
            # Check if t0 + tau is within signal bounds
            if t0 + tau >= N:
                continue
            
            # Single increment at fixed t0
            increment = signal[t0 + tau] - signal[t0]
            
            inc_rows.append({
                'file': file,
                'station': station,
                'stream': stream,
                'tau': tau,
                'increment': increment
            })
    
    df_result = pd.DataFrame(inc_rows)
    
    print(f"\nIncrement computation summary (ensemble, t0={t0}):")
    print(f"  Files processed: {df_result['file'].nunique()}")
    print(f"  Tau values: {df_result['tau'].nunique()}")
    print(f"  Total increments: {len(df_result)}")
    print(f"  Increments per tau: {len(df_result) // df_result['tau'].nunique()}")
    
    return df_result


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
        Increments from compute_increments_ensemble()
        Must have columns: ['file', 'station', 'stream', 'tau', 'increment']
    q_values : list of float
        Moment orders to compute
    
    Returns
    -------
    pd.DataFrame
        Columns: [q, tau, moment, n_stations]
        
    Notes
    -----
    Unlike compute_moments_from_increments() which averages over t0 for each file,
    this function averages over all stations at each tau, producing a single
    moment value per (tau, q) pair for the entire ensemble.
    
    Examples
    --------
    >>> df_inc = compute_increments_ensemble(df, tau_values, t0=0, column='displacement')
    >>> df_mom = compute_moments_from_increments_ensemble(df_inc, q_values)
    """
    rows = []
    
    # Group by tau only (ensemble averaging over all stations)
    for tau, group in df_increments.groupby('tau'):
        increments = group['increment'].values
        n_stations = len(increments)
        
        for q in q_values:
            # Ensemble average: mean over all stations at this tau
            moment = np.mean(np.abs(increments) ** q)
            
            rows.append({
                'q': q,
                'tau': tau,
                'moment': moment,
                'n_stations': n_stations
            })
    
    df_result = pd.DataFrame(rows)
    
    print(f"\nMoment computation summary (ensemble):")
    print(f"  q values: {df_result['q'].nunique()}")
    print(f"  Tau values: {df_result['tau'].nunique()}")
    print(f"  Stations per tau: {df_result['n_stations'].iloc[0]}")
    print(f"  Total moment values: {len(df_result)}")
    
    return df_result

# ===============================================================================================
# ==================================== Validation ===============================================
# ===============================================================================================

def validate_moments_ensemble(df_moments, expected_n_stations=48):
    """
    Validate ensemble-averaged moments.
    
    Parameters
    ----------
    df_moments : pd.DataFrame
        Moments from compute_moments_from_increments_ensemble()
        Must have columns: ['q', 'tau', 'moment', 'n_stations']
    expected_n_stations : int
        Expected number of stations in ensemble
    
    Returns
    -------
    bool
        True if validation passes
    
    Raises
    ------
    AssertionError
        If validation fails
    """
    print("Validating ensemble moments...")
    
    # Check 1: Consistent ensemble size
    n_stations_unique = df_moments['n_stations'].unique()
    assert len(n_stations_unique) == 1, \
        f"Inconsistent ensemble size: {n_stations_unique}"
    
    n_stations = n_stations_unique[0]
    assert n_stations == expected_n_stations, \
        f"Expected {expected_n_stations} stations, got {n_stations}"
    
    print(f"Ensemble size: {n_stations} stations (consistent)")
    
    # Check 2: No NaN
    assert df_moments['moment'].isna().sum() == 0, \
        "NaN found in moments"
    print(f"No NaN in moments")
    
    # Check 3: No Inf
    assert np.isinf(df_moments['moment']).sum() == 0, \
        "Inf found in moments"
    print(f"No Inf in moments")
    
    # Check 4: Moments positive
    assert (df_moments['moment'] > 0).all(), \
        "Non-positive moments found"
    print(f"All moments positive")
    
    # Check 5: Monotonicity in q (for fixed tau)
    for tau in df_moments['tau'].unique()[:5]:  # Check first 5 tau
        df_tau = df_moments[df_moments['tau'] == tau].sort_values('q')
        moments = df_tau['moment'].values
        assert np.all(np.diff(moments) > 0), \
            f"Moments not monotonic in q at tau={tau}"
    print(f"Moments monotonic in q")
    
    print("All validation checks passed! ✓\n")
    return True

# ===============================================================================================
# ==================================== Increment Analysis =======================================
# ===============================================================================================

def analyze_increments_ensemble(df_increments, output_dir=None):
    """
    Analyze ensemble increment statistics.
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        Increments from compute_increments_ensemble()
    output_dir : str, optional
        Directory to save diagnostic plots
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per tau across ensemble
    """
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
    
    # Print summary
    print("Ensemble increment statistics summary:")
    print(f"  Tau range: {df_summary['tau'].min()} - {df_summary['tau'].max()}")
    print(f"  Ensemble size: {df_summary['n_stations'].iloc[0]} stations")
    print(f"  Mean across all tau: {df_summary['mean'].mean():.6f}")
    print(f"  Std range: [{df_summary['std'].min():.6f}, {df_summary['std'].max():.6f}]")
    print()
    
    # Optional: plot diagnostics
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean vs tau
        axes[0, 0].semilogx(df_summary['tau'], df_summary['mean'], 'o-')
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('τ (samples)')
        axes[0, 0].set_ylabel('Mean increment')
        axes[0, 0].set_title('Ensemble Mean vs τ')
        axes[0, 0].grid(alpha=0.3)
        
        # Std vs tau
        axes[0, 1].loglog(df_summary['tau'], df_summary['std'], 'o-')
        axes[0, 1].set_xlabel('τ (samples)')
        axes[0, 1].set_ylabel('Std increment')
        axes[0, 1].set_title('Ensemble Std vs τ')
        axes[0, 1].grid(alpha=0.3)
        
        # Skewness vs tau
        axes[1, 0].semilogx(df_summary['tau'], df_summary['skewness'], 'o-')
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('τ (samples)')
        axes[1, 0].set_ylabel('Skewness')
        axes[1, 0].set_title('Ensemble Skewness vs τ')
        axes[1, 0].grid(alpha=0.3)
        
        # Kurtosis vs tau
        axes[1, 1].semilogx(df_summary['tau'], df_summary['kurtosis'], 'o-')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5, label='Gaussian')
        axes[1, 1].set_xlabel('τ (samples)')
        axes[1, 1].set_ylabel('Excess kurtosis')
        axes[1, 1].set_title('Ensemble Kurtosis vs τ')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'increment_diagnostics_ensemble.pdf'))
        plt.close()
        
        print(f"Saved diagnostics to {output_dir}/increment_diagnostics_ensemble.pdf")
    
    return df_summary