"""
Ensemble-averaged moment scaling analysis for seismic signals.

This module implements spatial ensemble averaging across multiple stations to
compute moment scaling exponents ζ(q) for different seismic phases (pre-event,
P-wave, S-wave, coda). Each phase is analyzed separately with a fixed reference
time t₀ at the window start and varying time lag τ.

Theoretical framework:
    For a stochastic process x(t), the q-th order moment of increments scales as:
    
        M_q(τ) = ⟨|x(t₀+τ) - x(t₀)|^q⟩ ~ τ^ζ(q)
    
    where:
    - τ is the time lag (increment duration)
    - q is the moment order
    - ζ(q) is the scaling exponent
    - ⟨·⟩ denotes ensemble average (across stations)
    
    Normal diffusion: ζ(q) = q/2 (linear in q)
    Anomalous diffusion: ζ(q) ≠ q/2
    Strong anomalous diffusion: ζ(q) piecewise-linear with breakpoint

Methodology:
    1. For each seismic window (pre_event, p_wave, s_wave, coda):
       - Fix t₀ at window start (different absolute time for each station)
       - Use common τ vector (limited by shortest window)
    2. Compute increments: Δx(τ) = x(t₀+τ) - x(t₀) for each station
    3. Compute moments: M_q(τ) = |Δx(τ)|^q for each station
    4. Spatial ensemble: ⟨M_q(τ)⟩ = mean over all stations
    5. Extract scaling: fit log⟨M_q⟩ vs log(τ) → slope = ζ(q)
    6. Compare ζ(q) across windows to identify dynamical regime changes

Expected behavior:
    - Pre-event: ζ(q) ≈ 0 (no scaling, instrumental noise)
    - P-wave: ζ(q) > 0 with slope α₁
    - S-wave: ζ(q) > 0 with slope α₂ ≠ α₁
    - Coda: ζ(q) → 0 (return to background fluctuations)

Usage:
    from window_segmentation import segment_all_signals
    from signals_scaling_ensemble import (
        analyze_all_windows,
        save_results_parquet,
        plot_scaling_curves,
        plot_scaling_exponents
    )
    
    # Segment signals into windows
    windowed_signals = segment_all_signals(signals_dict, df_onsets)
    
    # Analyze all windows
    results = analyze_all_windows(
        windowed_signals,
        tau_min=0.01,
        n_tau=50,
        q_values=np.array([0.5, 0.75, ..., 5.0]),
        sampling_rate=200.0
    )
    
    # Save results
    save_results_parquet(results, output_dir='../data/processed/ensemble_spatial')
    
    # Plot
    plot_scaling_curves(results, output_dir='../figures')
    plot_scaling_exponents(results, output_dir='../figures')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


def prepare_window_data(
    windowed_signals: Dict,
    window_name: str,
    exclude_components: Optional[List[str]] = None 
) -> Tuple[List[np.ndarray], List[np.ndarray], float, int]:
    """
    Extract signal and time arrays for a specific seismic window across all stations.
    
    Parameters
    ----------
    windowed_signals : dict
        Nested dictionary from segment_all_signals():
        {station: {component: {window_name: {'signal': array, 'time': array, ...}}}}
    window_name : str
        Window to extract: 'pre_event', 'p_wave', 's_wave', 'coda'
    exclude_components : list of str, optional
        Component codes to exclude (e.g., ['HNZ', 'HGZ'] for vertical)
        If None, includes all components
        
    Returns
    -------
    signals_list : list of np.ndarray
        Signal arrays for this window (one per station-component)
    times_list : list of np.ndarray
        Time arrays for this window
    tau_max_seconds : float
        Maximum usable tau (duration of shortest window)
    n_signals : int
        Number of signals in ensemble
        
    Raises
    ------
    ValueError
        If window_name not found or no valid signals available
        
    Notes
    -----
    Stations with the same code but different components are treated as
    independent ensemble members (as instructed by advisor).
    """
    if exclude_components is None:
        exclude_components = []

    signals_list = []
    times_list = []
    durations = []
    
    for station in windowed_signals:
        for component in windowed_signals[station]:
            if component in exclude_components:
                continue
            if window_name not in windowed_signals[station][component]:
                continue
            
            window_data = windowed_signals[station][component][window_name]
            signal = window_data['signal']
            time = window_data['time']
            duration = window_data['duration']
            
            if len(signal) < 2:
                continue
            
            signals_list.append(signal)
            times_list.append(time)
            durations.append(duration)
    
    if len(signals_list) == 0:
        if exclude_components:
            raise ValueError(
                f"No valid signals found for window '{window_name}' "
                f"after excluding components: {exclude_components}"
            )
        else:
            raise ValueError(f"No valid signals found for window '{window_name}'")
    tau_max_seconds = min(durations)
    n_signals = len(signals_list)
    
    return signals_list, times_list, tau_max_seconds, n_signals


def compute_moments_single_signal(
    signal: np.ndarray,
    tau_indices: np.ndarray,
    q_values: np.ndarray,
    t0_index: int = 0
) -> np.ndarray:
    """
    Compute moments M_q(tau) for a single signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series (acceleration, velocity, or displacement)
    tau_indices : np.ndarray
        Array of time lag indices (in samples)
    q_values : np.ndarray
        Array of moment orders
    t0_index : int, optional
        Starting index for increments (default: 0 = window start)
        
    Returns
    -------
    moments : np.ndarray
        Shape (n_tau, n_q) containing M_q(tau) = |signal[t0+tau] - signal[t0]|^q
        
    Notes
    -----
    Increments are computed as point differences:
        Δx(τ) = x(t₀ + τ) - x(t₀)
    
    Moments are defined as:
        M_q(τ) = |Δx(τ)|^q
    
    For ensemble averaging, this function is called once per signal, then results
    are averaged across the ensemble.
    """
    n_tau = len(tau_indices)
    n_q = len(q_values)
    moments = np.zeros((n_tau, n_q))
    
    x_t0 = signal[t0_index]
    
    for i, tau_idx in enumerate(tau_indices):
        endpoint_idx = t0_index + tau_idx
        
        if endpoint_idx >= len(signal):
            moments[i, :] = np.nan
            continue
        
        increment = signal[endpoint_idx] - x_t0
        abs_increment = np.abs(increment)
        
        for j, q in enumerate(q_values):
            moments[i, j] = abs_increment ** q
    
    return moments


def compute_spatial_ensemble(
    windowed_signals: Dict,
    window_name: str,
    tau_min: float = 0.01,
    n_tau: Optional[int] = None,
    q_values: np.ndarray = None,
    sampling_rate: float = 200.0,
    exclude_components: Optional[List[str]] = None 
) -> Dict:
    """
    Compute spatial ensemble-averaged moments for a single seismic window.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals()
    window_name : str
        Window to analyze: 'pre_event', 'p_wave', 's_wave', 'coda'
    tau_min : float, optional
        Minimum time lag in seconds (default: 0.01s)
    n_tau : int, optional
        Number of tau values. If None, computed automatically from tau range
    q_values : np.ndarray, optional
        Moment orders to compute. If None, uses default range [0.5, ..., 5.0]
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
     exclude_components : list of str, optional
        Component codes to exclude from ensemble
        
    Returns
    -------
    results : dict
        {
            'tau': np.ndarray (n_tau,) - time lags in seconds
            'tau_indices': np.ndarray (n_tau,) - time lags in samples
            'q': np.ndarray (n_q,) - moment orders
            'moments_mean': np.ndarray (n_tau, n_q) - ensemble-averaged moments
            'moments_std': np.ndarray (n_tau, n_q) - std across ensemble
            'moments_individual': list of np.ndarray - individual moments per signal
            'n_signals': int - number of signals in ensemble
            'tau_max': float - maximum tau (seconds)
            'window_name': str
        }
        
    Notes
    -----
    Workflow:
    1. Extract all signals for this window across stations
    2. Find tau_max from shortest window duration
    3. Generate logarithmic tau vector from tau_min to tau_max
    4. Compute moments for each signal individually
    5. Average moments across all signals (spatial ensemble)
    
    The number of tau points is automatically adjusted based on the dynamic range:
        n_tau = max(30, int(log10(tau_max/tau_min) * 20))
    This ensures ~20 points per decade in log space.
    """
    if q_values is None:
        q_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                            2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0])
    
    signals_list, times_list, tau_max_seconds, n_signals = prepare_window_data(
        windowed_signals, window_name, exclude_components=exclude_components
    )
    
    if tau_max_seconds <= tau_min:
        raise ValueError(
            f"Window '{window_name}' too short: tau_max={tau_max_seconds:.3f}s <= tau_min={tau_min:.3f}s"
        )
    
    if n_tau is None:
        n_decades = np.log10(tau_max_seconds / tau_min)
        n_tau = max(30, int(n_decades * 20))
    
    tau_values_seconds = np.logspace(np.log10(tau_min), np.log10(tau_max_seconds), n_tau)
    tau_indices = np.round(tau_values_seconds * sampling_rate).astype(int)
    tau_indices = np.unique(tau_indices)
    tau_values_seconds = tau_indices / sampling_rate
    n_tau = len(tau_indices)
    
    moments_individual = []
    
    for signal in signals_list:
        moments = compute_moments_single_signal(signal, tau_indices, q_values, t0_index=0)
        moments_individual.append(moments)
    
    moments_stack = np.stack(moments_individual, axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        moments_mean = np.nanmean(moments_stack, axis=0)
        moments_std = np.nanstd(moments_stack, axis=0)
    
    results = {
        'tau': tau_values_seconds,
        'tau_indices': tau_indices,
        'q': q_values,
        'moments_mean': moments_mean,
        'moments_std': moments_std,
        'moments_individual': moments_individual,
        'n_signals': n_signals,
        'tau_max': tau_max_seconds,
        'window_name': window_name
    }
    
    return results


def extract_scaling_exponents(
    tau: np.ndarray,
    moments_mean: np.ndarray,
    q_values: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None,
    threshold: float = 1e-15
) -> Dict:
    """
    Extract scaling exponents ζ(q) from ensemble-averaged moments.
    
    For each moment order q, performs linear fit in log-log space:
        log(M_q) = ζ(q) * log(τ) + intercept
    
    Parameters
    ----------
    tau : np.ndarray
        Time lags in seconds (n_tau,)
    moments_mean : np.ndarray
        Ensemble-averaged moments (n_tau, n_q)
    q_values : np.ndarray
        Moment orders (n_q,)
    fit_range : tuple of float, optional
        (tau_min, tau_max) to restrict fit range. If None, uses all tau.
    threshold : float, optional
        Minimum moment value to include in fit (default: 1e-15)
        Values below threshold are excluded to avoid log(0)
        
    Returns
    -------
    results : dict
        {
            'zeta': np.ndarray (n_q,) - scaling exponents
            'zeta_err': np.ndarray (n_q,) - standard errors from fit
            'intercepts': np.ndarray (n_q,) - y-intercepts
            'r_squared': np.ndarray (n_q,) - R² goodness of fit
            'n_points': np.ndarray (n_q,) - number of points used in each fit
        }
        
    Notes
    -----
    Points are excluded from fit if:
    - moment_mean < threshold (to avoid log of very small/zero values)
    - tau outside fit_range (if specified)
    - moment is NaN or Inf
    
    If fewer than 3 valid points remain for a given q, that exponent is set to NaN.
    """
    n_q = len(q_values)
    
    zeta = np.zeros(n_q)
    zeta_err = np.zeros(n_q)
    intercepts = np.zeros(n_q)
    r_squared = np.zeros(n_q)
    n_points = np.zeros(n_q, dtype=int)
    
    for i, q in enumerate(q_values):
        moments_q = moments_mean[:, i]
        
        valid_mask = (moments_q > threshold) & np.isfinite(moments_q)
        
        if fit_range is not None:
            tau_min_fit, tau_max_fit = fit_range
            valid_mask &= (tau >= tau_min_fit) & (tau <= tau_max_fit)
        
        n_valid = valid_mask.sum()
        n_points[i] = n_valid
        
        if n_valid < 3:
            zeta[i] = np.nan
            zeta_err[i] = np.nan
            intercepts[i] = np.nan
            r_squared[i] = np.nan
            continue
        
        log_tau_valid = np.log10(tau[valid_mask])
        log_M_valid = np.log10(moments_q[valid_mask])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_tau_valid, log_M_valid
        )
        
        zeta[i] = slope
        zeta_err[i] = std_err
        intercepts[i] = intercept
        r_squared[i] = r_value ** 2
    
    results = {
        'zeta': zeta,
        'zeta_err': zeta_err,
        'intercepts': intercepts,
        'r_squared': r_squared,
        'n_points': n_points
    }
    
    return results


def analyze_all_windows(
    windowed_signals: Dict,
    tau_min: float = 0.01,
    n_tau: Optional[int] = None,
    q_values: np.ndarray = None,
    sampling_rate: float = 200.0,
    fit_range: Optional[Tuple[float, float]] = None,
    exclude_components: Optional[List[str]] = None
) -> Dict:
    """
    Analyze all four seismic windows with spatial ensemble averaging.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals()
    tau_min : float, optional
        Minimum time lag in seconds (default: 0.01s, fixed for all windows)
    n_tau : int, optional
        Number of tau values per window. If None, computed automatically.
    q_values : np.ndarray, optional
        Moment orders. If None, uses [0.5, 0.75, ..., 5.0]
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
    fit_range : tuple of float, optional
        (tau_min_fit, tau_max_fit) for scaling exponent extraction
    exclude_components : list of str, optional
        Component codes to exclude from ensemble
        
    Returns
    -------
    results : dict
        {
            'pre_event': {
                'ensemble': {...},  # from compute_spatial_ensemble()
                'scaling': {...}    # from extract_scaling_exponents()
            },
            'p_wave': {...},
            's_wave': {...},
            'coda': {...}
        }
        
    Notes
    -----
    Each window may have different tau_max (based on shortest duration),
    but tau_min is fixed across all windows for consistency.
    
    The function prints summary statistics for each window including:
    - Number of signals in ensemble
    - Tau range (seconds)
    - Number of tau points
    - Mean ζ(q) values
    """
    if q_values is None:
        q_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                            2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0])
    
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    results = {}
    
    print("="*80)
    print("ENSEMBLE SPATIAL SCALING ANALYSIS")
    print("="*80)
    print(f"tau_min: {tau_min:.3f} s (fixed for all windows)")
    print(f"q_values: {len(q_values)} values from {q_values.min():.2f} to {q_values.max():.2f}")
    print(f"sampling_rate: {sampling_rate:.1f} Hz")
    if fit_range is not None:
        print(f"fit_range: [{fit_range[0]:.3f}, {fit_range[1]:.3f}] s")
    print("="*80)
    
    for window_name in windows:
        print(f"\nProcessing window: {window_name.upper()}")
        print("-"*80)
        
        try:
            ensemble_results = compute_spatial_ensemble(
                windowed_signals=windowed_signals,
                window_name=window_name,
                tau_min=tau_min,
                n_tau=n_tau,
                q_values=q_values,
                sampling_rate=sampling_rate,
                exclude_components=exclude_components
            )
            
            scaling_results = extract_scaling_exponents(
                tau=ensemble_results['tau'],
                moments_mean=ensemble_results['moments_mean'],
                q_values=ensemble_results['q'],
                fit_range=fit_range
            )
            
            results[window_name] = {
                'ensemble': ensemble_results,
                'scaling': scaling_results
            }

            if exclude_components:
                n_excluded = len(exclude_components)
                print(f"  Excluded {n_excluded} component type(s): {exclude_components}")
            
            if exclude_components:
                n_excluded = len(exclude_components)
                print(f"  Excluded {n_excluded} component type(s): {exclude_components}")

            tau = ensemble_results['tau']
            n_signals = ensemble_results['n_signals']
            zeta = scaling_results['zeta']
            r_squared = scaling_results['r_squared']

            print(f"Ensemble size: {n_signals} signals")
            print(f"Tau range: [{tau.min():.4f}, {tau.max():.4f}] s")
            print(f"Number of tau points: {len(tau)}")
            print(f"Mean ζ(q): {np.nanmean(zeta):.4f} ± {np.nanstd(zeta):.4f}")
            print(f"Mean R²: {np.nanmean(r_squared):.4f}")
            print(f"ζ(q=1): {zeta[np.argmin(np.abs(q_values - 1.0))]:.4f}")
            print(f"ζ(q=2): {zeta[np.argmin(np.abs(q_values - 2.0))]:.4f}")
            
        except Exception as e:
            print(f"Error processing {window_name}: {e}")
            results[window_name] = None
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results


def save_results_parquet(
    results: Dict,
    output_dir: str = '../data/processed/ensemble_spatial'
) -> None:
    """
    Save ensemble scaling results in parquet format.
    
    Creates two types of files:
    1. Summary file: scaling exponents for all windows
    2. Moments files: detailed moment data per window
    
    Parameters
    ----------
    results : dict
        Output from analyze_all_windows()
    output_dir : str or Path
        Directory to save parquet files
        
    Output Files
    ------------
    ensemble_spatial_summary.parquet:
        Columns: window, q, zeta, zeta_err, r_squared, intercept, n_points,
                 n_signals, tau_min, tau_max, n_tau
                 
    ensemble_spatial_moments_{window}.parquet (one per window):
        Columns: tau, q, moment_mean, moment_std, n_signals
        
    Notes
    -----
    Uses long format (one row per tau-q combination) for easy filtering and plotting.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for window_name, window_results in results.items():
        if window_results is None:
            continue
        
        ensemble = window_results['ensemble']
        scaling = window_results['scaling']
        
        tau = ensemble['tau']
        q_values = ensemble['q']
        n_signals = ensemble['n_signals']
        tau_min = tau.min()
        tau_max = tau.max()
        n_tau = len(tau)
        
        moments_rows = []
        
        for i, tau_val in enumerate(tau):
            for j, q_val in enumerate(q_values):
                moments_rows.append({
                    'tau': tau_val,
                    'q': q_val,
                    'moment_mean': ensemble['moments_mean'][i, j],
                    'moment_std': ensemble['moments_std'][i, j],
                    'n_signals': n_signals
                })
        
        df_moments = pd.DataFrame(moments_rows)
        moments_file = output_dir / f'ensemble_spatial_moments_{window_name}.parquet'
        df_moments.to_parquet(moments_file, index=False)
        print(f"Saved: {moments_file}")
        
        for j, q_val in enumerate(q_values):
            summary_rows.append({
                'window': window_name,
                'q': q_val,
                'zeta': scaling['zeta'][j],
                'zeta_err': scaling['zeta_err'][j],
                'r_squared': scaling['r_squared'][j],
                'intercept': scaling['intercepts'][j],
                'n_points': scaling['n_points'][j],
                'n_signals': n_signals,
                'tau_min': tau_min,
                'tau_max': tau_max,
                'n_tau': n_tau
            })
    
    df_summary = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'ensemble_spatial_summary.parquet'
    df_summary.to_parquet(summary_file, index=False)
    print(f"Saved: {summary_file}")
    
    print(f"\nAll results saved to: {output_dir}")