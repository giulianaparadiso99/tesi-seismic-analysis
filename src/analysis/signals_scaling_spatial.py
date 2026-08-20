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
import warnings
from scipy import stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src import derive_threshold_run_config
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def prepare_window_data(
    windowed_signals: Dict,
    window_name: str,
    signal_field: str = 'signal',
    sampling_rate: float = 200.0,
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
    signal_field : str, optional
        Key to extract signal from window_data dict (default: 'signal')
        Use 'signal' for raw data or custom field name if available
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
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
            signal = window_data[signal_field]
            time = window_data['time']
            duration = window_data['duration_samples']
            
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
    min_signal_length = min([len(s) for s in signals_list])
    dt = 1.0 / sampling_rate
    tau_max_seconds = min_signal_length * dt
    n_signals = len(signals_list)
    
    # Debug validation
    for i, (signal, duration) in enumerate(zip(signals_list, durations)):
        if len(signal) != duration:
            print(f"Warning: Signal {i} length mismatch: len={len(signal)}, duration_samples={duration}")

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
    signal_field: str = 'signal',
    tau_min: float = 0.01,
    n_tau: Optional[int] = None,
    tau_max_fraction: Optional[float] = None,
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
    tau_max_fraction : float or None, optional
        Fraction of shortest window duration to use for tau_max (default: None)
        - None (default): use full window duration (tau_max = 100%)
        - 0.5: use first 50% of window (recommended for scaling analysis)
        - 0.3: use first 30% (conservative)
        Values < 1.0 avoid finite-size effects in scaling analysis
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
            'tau_samples': np.ndarray (n_tau,) - time lags in sample indices
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
    CRITICAL IMPLEMENTATION DETAIL:
    Tau values are generated directly in sample space to preserve logarithmic
    distribution. The workflow is:
    1. Define tau_min and tau_max in samples
    2. Generate logarithmic tau array in samples using np.logspace
    3. Apply np.unique to remove duplicate sample indices
    4. Convert to seconds ONLY for output (single conversion, no rounding loss)
    
    This approach avoids the problem of:
        seconds → samples → unique → seconds
    which breaks logarithmic distribution due to rounding artifacts.
    
    Workflow:
    1. Extract all signals for this window across stations
    2. Find tau_max from shortest window duration
    3. Generate logarithmic tau vector from tau_min to tau_max (in SAMPLES)
    4. Compute moments for each signal individually using sample indices
    5. Average moments across all signals (spatial ensemble)
    
    The number of tau points is automatically adjusted based on the dynamic range:
        n_tau = max(30, int(log10(tau_max/tau_min) * 20))
    This ensures ~20 points per decade in log space.
    """
    if q_values is None:
        q_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                            2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0])
    
    # Extract signals and get tau_max in seconds
    signals_list, times_list, tau_max_seconds, n_signals = prepare_window_data(
        windowed_signals, window_name, signal_field=signal_field, 
        sampling_rate=sampling_rate, exclude_components=exclude_components
    )

    # Apply tau_max_fraction if specified
    if tau_max_fraction is not None:
        if not (0 < tau_max_fraction <= 1):
            raise ValueError("tau_max_fraction must be in (0, 1]")
        tau_max_seconds *= tau_max_fraction
    
    # Validate tau range
    if tau_max_seconds <= tau_min:
        raise ValueError(
            f"Window '{window_name}' too short: tau_max={tau_max_seconds:.3f}s <= tau_min={tau_min:.3f}s"
        )
    
    # Generate tau in SAMPLES, not seconds =====
    # Convert tau_min and tau_max to samples
    tau_min_samples = max(1, int(np.round(tau_min * sampling_rate)))
    tau_max_samples = int(np.floor(tau_max_seconds * sampling_rate))
    
    # Validate sample range
    if tau_max_samples <= tau_min_samples:
        raise ValueError(
            f"Insufficient sample range: tau_max_samples={tau_max_samples} <= "
            f"tau_min_samples={tau_min_samples}"
        )
    
    # Determine number of tau points (if not specified)
    if n_tau is None:
        n_decades = np.log10(tau_max_samples / tau_min_samples)
        n_tau = max(30, int(n_decades * 20))
    
    # Generate tau directly in sample space
    tau_samples = np.unique(np.round(
        np.logspace(np.log10(tau_min_samples), np.log10(tau_max_samples), n_tau)
    ).astype(int))
    
    # Convert to seconds ONLY for output
    tau_seconds = tau_samples / sampling_rate
    
    # Update n_tau after unique() operation
    n_tau = len(tau_samples)
    
    # ===== Compute moments for each signal using sample indices =====
    moments_individual = []
    for signal in signals_list:
        moments = compute_moments_single_signal(
            signal, tau_samples, q_values, t0_index=0
        )
        moments_individual.append(moments)
    
    # Stack and compute ensemble statistics
    moments_stack = np.stack(moments_individual, axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        moments_mean = np.nanmean(moments_stack, axis=0)
        moments_std = np.nanstd(moments_stack, axis=0)
    
    results = {
        'tau': tau_seconds,              # For output/plotting (seconds)
        'tau_samples': tau_samples,      # Primary representation (samples)
        'q': q_values,
        'moments_mean': moments_mean,
        'moments_std': moments_std,
        'moments_individual': moments_individual,
        'n_signals': n_signals,
        'tau_max': tau_max_seconds,      # For reference (seconds)
        'window_name': window_name
    }
    
    return results


def extract_scaling_exponents(
    tau: np.ndarray,
    moments_mean: np.ndarray,
    q_values: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None,
    threshold: float = 1e-300
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
        Minimum moment value to include in fit (default: 1e-300)
        Values below threshold are excluded to avoid log(0).
        Not intended as a physical lower bound
        on increment amplitudes: for signals with small absolute scale
        (e.g. displacement), |Δx|^q for large q can legitimately be
        much smaller than typical "small number" thresholds like 1e-15.
        
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
    signal_field: str = 'signal',
    tau_min: float = 0.01,
    n_tau: Optional[int] = None,
    tau_max_fraction: Optional[float] = None,
    q_values: np.ndarray = None,
    sampling_rate: float = 200.0,
    fit_range: Optional[Tuple[float, float]] = None,
    exclude_components: Optional[List[str]] = None,
    verbose: bool = True
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
    tau_max_fraction : float or None, optional
        Fraction of shortest window duration to use for tau_max (default: None)
        - None (default): use full window duration for each phase
        - 0.5: use first 50% (recommended to avoid finite-size effects)
        - 0.3: use first 30% (conservative)
        Applied independently to each window based on its shortest duration
    q_values : np.ndarray, optional
        Moment orders. If None, uses [0.5, 0.75, ..., 5.0]
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
    fit_range : tuple of float, optional
        (tau_min_fit, tau_max_fit) for scaling exponent extraction
    exclude_components : list of str, optional
        Component codes to exclude from ensemble
    verbose : bool, optional
        If True, print detailed progress
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
    
    if verbose:
        print("="*80)
        print("ENSEMBLE SPATIAL SCALING ANALYSIS")
        print("="*80)
        print(f"tau_min: {tau_min:.3f} s (fixed for all windows)")
    if verbose:
        if tau_max_fraction is None:
            print("tau_max_fraction: None (use full window duration)")
        else:
            print(f"tau_max_fraction: {tau_max_fraction:.1%} of shortest window duration")
    if verbose:
        print(f"q_values: {len(q_values)} values from {q_values.min():.2f} to {q_values.max():.2f}")
        print(f"sampling_rate: {sampling_rate:.1f} Hz")
    if fit_range is not None and verbose:
        print(f"fit_range: [{fit_range[0]:.3f}, {fit_range[1]:.3f}] s")
    if verbose:
        print("="*80)
    
    for window_name in windows:
        if verbose:
            print(f"\nProcessing window: {window_name.upper()}")
            print("-"*80)
        
        try:
            ensemble_results = compute_spatial_ensemble(
                windowed_signals=windowed_signals,
                window_name=window_name,
                signal_field=signal_field,
                tau_min=tau_min,
                tau_max_fraction=tau_max_fraction,
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
                if verbose:
                    print(f"  Excluded {n_excluded} component type(s): {exclude_components}")
            

            tau = ensemble_results['tau']
            n_signals = ensemble_results['n_signals']
            zeta = scaling_results['zeta']
            r_squared = scaling_results['r_squared']

            if verbose:
                idx_q1 = np.argmin(np.abs(q_values - 1.0))
                idx_q2 = np.argmin(np.abs(q_values - 2.0))

                print(f"Ensemble size: {n_signals} signals")
                print(f"Tau range: [{tau.min():.4f}, {tau.max():.4f}] s")
                print(f"Number of tau points: {len(tau)}")
                print(f"Mean ζ(q): {np.nanmean(zeta):.4f} ± {np.nanstd(zeta):.4f}")
                print(f"Mean R²: {np.nanmean(r_squared):.4f}")
                print(f"ζ(q=1): {zeta[idx_q1]:.4f}, R²(q=1): {r_squared[idx_q1]:.4f}")
                print(f"ζ(q=2): {zeta[idx_q2]:.4f}, R²(q=2): {r_squared[idx_q2]:.4f}")
            
        except ValueError as e:
            if verbose:
                print(f"Error processing {window_name}: {e}")
            results[window_name] = None
    if verbose:
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

    Returns
    -------
    None
        Files are saved to disk; function returns nothing
        
    Output Files
    ------------
    ensemble_spatial_summary.parquet:
        Columns: window, q, zeta, zeta_err, r_squared, intercept, n_points,
                 n_signals, tau_min, tau_max, n_tau
                 
    ensemble_spatial_moments_{window}.parquet (one per window):
        Columns: tau, tau_samples, q, moment_mean, moment_std, n_signals
        
    Notes
    -----
    Uses long format (one row per tau-q combination) for easy filtering and plotting.
    Both tau (seconds) and tau_samples (sample indices) are saved for dual representation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for window_name, window_results in results.items():
        if window_results is None:
            continue
        
        ensemble = window_results['ensemble']
        scaling = window_results['scaling']
        
        tau = ensemble['tau']  # seconds
        tau_samples = ensemble.get('tau_samples', None)  # samples (if available)
        q_values = ensemble['q']
        n_signals = ensemble['n_signals']
        tau_min = tau.min()
        tau_max = tau.max()
        n_tau = len(tau)
        
        # ===== MOMENTS FILE (detailed data) =====
        moments_rows = []
        
        for i, tau_val in enumerate(tau):
            tau_samp = tau_samples[i] if tau_samples is not None else None
            
            for j, q_val in enumerate(q_values):
                row = {
                    'tau': tau_val,
                    'q': q_val,
                    'moment_mean': ensemble['moments_mean'][i, j],
                    'moment_std': ensemble['moments_std'][i, j],
                    'n_signals': n_signals
                }
                
                # Add tau_samples if available
                if tau_samp is not None:
                    row['tau_samples'] = tau_samp
                
                moments_rows.append(row)
        
        df_moments = pd.DataFrame(moments_rows)
        moments_file = output_dir / f'ensemble_spatial_moments_{window_name}.parquet'
        df_moments.to_parquet(moments_file, index=False)
        print(f"Saved: {moments_file}")
        
        # ===== SUMMARY FILE (scaling exponents) =====
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

def analyze_single_signal(
    signal: np.ndarray,
    tau_min: float = 0.01,
    tau_max_fraction: Optional[float] = None,
    n_tau: Optional[int] = None,
    q_values: np.ndarray = None,
    sampling_rate: float = 200.0
) -> Dict:
    """
    Analyze moment scaling for a single signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series (acceleration, velocity, or displacement)
    tau_min : float, optional
        Minimum time lag in seconds (default: 0.01s)
    tau_max_fraction : float, optional
        Maximum tau as fraction of signal duration (default: None)
    n_tau : int, optional
        Number of tau values. If None, computed automatically
    q_values : np.ndarray, optional
        Moment orders. If None, uses [0.5, 0.75, ..., 5.0]
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200.0)
        
    Returns
    -------
    results : dict
        {
            'tau': array of time lags (seconds),
            'tau_samples': array of time lags (sample indices),
            'q': array of moment orders,
            'moments': moments M_q(tau), shape (n_tau, n_q),
            'zeta': scaling exponents,
            'zeta_err': standard errors,
            'intercepts': fit intercepts,
            'r_squared': R² values,
            'n_points': number of points in each fit
        }
        
    Notes
    -----
    Tau values are generated directly in sample space to preserve logarithmic
    distribution, then converted to seconds for output. This avoids rounding
    artifacts from seconds → samples → seconds conversion.
    """
    if q_values is None:
        q_values = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
                            2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0])
    
    # Calculate signal duration
    dt = 1.0 / sampling_rate
    duration = len(signal) * dt
    
    # Determine tau_max
    if tau_max_fraction is None:
        tau_max_seconds = duration
    else:
        tau_max_seconds = tau_max_fraction * duration
    
    if tau_max_seconds <= tau_min:
        raise ValueError(
            f"Signal too short: tau_max={tau_max_seconds:.3f}s <= tau_min={tau_min:.3f}s"
        )
    
    # Convert tau_min and tau_max to samples
    tau_min_samples = max(1, int(np.round(tau_min * sampling_rate)))
    tau_max_samples = int(np.floor(tau_max_seconds * sampling_rate))
    
    # Validate sample range
    if tau_max_samples <= tau_min_samples:
        raise ValueError(
            f"Insufficient sample range: tau_max_samples={tau_max_samples} <= "
            f"tau_min_samples={tau_min_samples}"
        )
    
    # Determine number of tau points (if not specified)
    if n_tau is None:
        n_decades = np.log10(tau_max_samples / tau_min_samples)
        n_tau = max(30, int(n_decades * 20))
    
    # Generate tau directly in sample space
    tau_samples = np.unique(np.round(
        np.logspace(np.log10(tau_min_samples), np.log10(tau_max_samples), n_tau)
    ).astype(int))
    
    # Convert to seconds ONLY for output
    tau_seconds = tau_samples / sampling_rate
    
    # ===== Compute moments using sample indices =====
    moments = compute_moments_single_signal(signal, tau_samples, q_values, t0_index=0)
    
    # ===== Fit scaling exponents for each q =====
    zeta = np.zeros(len(q_values))
    zeta_err = np.zeros(len(q_values))
    intercepts = np.zeros(len(q_values))
    r_squared = np.zeros(len(q_values))
    n_points = np.zeros(len(q_values), dtype=int)
    
    for i, q in enumerate(q_values):
        M_q = moments[:, i]
        valid = (M_q > 0) & np.isfinite(M_q)
        
        if valid.sum() < 2:
            zeta[i] = np.nan
            zeta_err[i] = np.nan
            intercepts[i] = np.nan
            r_squared[i] = np.nan
            n_points[i] = 0
            continue
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(tau_seconds[valid]),
            np.log10(M_q[valid])
        )
        
        zeta[i] = slope
        zeta_err[i] = std_err
        intercepts[i] = intercept
        r_squared[i] = r_value ** 2
        n_points[i] = valid.sum()
    
    results = {
        'tau': tau_seconds,        # For output/plotting (seconds)
        'tau_samples': tau_samples, # Primary representation (samples)
        'q': q_values,
        'moments': moments,
        'zeta': zeta,
        'zeta_err': zeta_err,
        'intercepts': intercepts,
        'r_squared': r_squared,
        'n_points': n_points
    }
    
    return results

def compute_tau_subinterval_sensitivity(
    tau: np.ndarray,
    moments_mean: np.ndarray,
    q_values: np.ndarray,
    threshold: float = 1e-300,
) -> pd.DataFrame:
    """
    Assess the sensitivity of scaling exponents zeta(q) to the choice
    of tau sub-interval used in the log-log regression, by splitting
    the available tau range in half (in log10 space) and re-fitting
    zeta(q) independently on each half.

    Parameters
    ----------
    tau : np.ndarray
        Time lags in seconds, as used in the baseline fit.
    moments_mean : np.ndarray
        Ensemble-averaged moments (n_tau, n_q), as used in the
        baseline fit.
    q_values : np.ndarray
        Moment orders (n_q,).
    threshold : float, optional
        Minimum moment value to include in fit, passed through to
        extract_scaling_exponents() (default: 1e-300).

    Returns
    -------
    pd.DataFrame
        One row per q, with columns: q, zeta_baseline, zeta_err_baseline,
        zeta_lower_half, zeta_err_lower_half, zeta_upper_half,
        zeta_err_upper_half, z_score_lower, z_score_upper, where
        z_score_X = |zeta_X - zeta_baseline| /
        sqrt(zeta_err_X^2 + zeta_err_baseline^2).

    Notes
    -----
    The split point is the midpoint of the tau range in log10 space,
    not in linear seconds, consistent with the log-uniform spacing of
    tau values used elsewhere in the pipeline.
    """
    tau_min, tau_max = tau.min(), tau.max()
    log_tau_split = (np.log10(tau_min) + np.log10(tau_max)) / 2
    tau_split = 10 ** log_tau_split

    baseline = extract_scaling_exponents(
        tau, moments_mean, q_values, fit_range=None, threshold=threshold
    )
    lower_half = extract_scaling_exponents(
        tau, moments_mean, q_values,
        fit_range=(tau_min, tau_split), threshold=threshold,
    )
    upper_half = extract_scaling_exponents(
        tau, moments_mean, q_values,
        fit_range=(tau_split, tau_max), threshold=threshold,
    )

    def _z_score(zeta_a, err_a, zeta_b, err_b):
        return np.abs(zeta_a - zeta_b) / np.sqrt(err_a**2 + err_b**2)

    return pd.DataFrame({
        'q': q_values,
        'zeta_baseline': baseline['zeta'],
        'zeta_err_baseline': baseline['zeta_err'],
        'n_points_baseline': baseline['n_points'],
        'zeta_lower_half': lower_half['zeta'],
        'zeta_err_lower_half': lower_half['zeta_err'],
        'n_points_lower_half': lower_half['n_points'],
        'zeta_upper_half': upper_half['zeta'],
        'zeta_err_upper_half': upper_half['zeta_err'],
        'n_points_upper_half': upper_half['n_points'],
        'z_score_lower': _z_score(
            lower_half['zeta'], lower_half['zeta_err'],
            baseline['zeta'], baseline['zeta_err'],
        ),
        'z_score_upper': _z_score(
            upper_half['zeta'], upper_half['zeta_err'],
            baseline['zeta'], baseline['zeta_err'],
        ),
    })

def sign_test_tau_subinterval(df_sensitivity: pd.DataFrame) -> Dict:
    """
    Test whether zeta(q) estimated on the upper tau sub-interval is
    consistently larger (or smaller) than on the lower sub-interval,
    across all moment orders q, using a two-sided binomial sign test.

    A consistent sign across many values of q is unlikely to arise
    from independent fit-to-fit noise, and indicates a systematic
    departure from a single power-law scaling within the analysed
    window, rather than sampling fluctuation.

    Parameters
    ----------
    df_sensitivity : pd.DataFrame
        Output of compute_tau_subinterval_sensitivity(), must contain
        columns 'zeta_lower_half' and 'zeta_upper_half'.

    Returns
    -------
    dict
        {
            'n_total': int - number of q values with valid estimates
                on both sub-intervals
            'n_upper_greater': int - number of q where
                zeta_upper_half > zeta_lower_half
            'n_lower_greater': int - number of q where
                zeta_lower_half > zeta_upper_half
            'p_value': float - two-sided binomial test p-value under
                the null hypothesis that each sign is equally likely
                by chance (p=0.5)
        }

    Notes
    -----
    The binomial test assumes independence across the q values
    tested. Adjacent q values are estimated from largely overlapping
    tau data and are therefore correlated, so the resulting p-value
    should be read as indicative of a consistent directional trend,
    not as a fully independent statistical test.
    """
    valid = (
        df_sensitivity['zeta_lower_half'].notna()
        & df_sensitivity['zeta_upper_half'].notna()
    )
    zeta_lower = df_sensitivity.loc[valid, 'zeta_lower_half'].to_numpy()
    zeta_upper = df_sensitivity.loc[valid, 'zeta_upper_half'].to_numpy()

    n_total = int(valid.sum())
    n_upper_greater = int((zeta_upper > zeta_lower).sum())
    n_lower_greater = int((zeta_lower > zeta_upper).sum())
    n_consistent = max(n_upper_greater, n_lower_greater)

    p_value = stats.binomtest(
        n_consistent, n_total, p=0.5, alternative='greater'
    ).pvalue

    return {
        'n_total': n_total,
        'n_upper_greater': n_upper_greater,
        'n_lower_greater': n_lower_greater,
        'p_value': p_value,
    }

def summarize_sign_test_by_signal(
    sign_test_by_signal: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Assemble a summary table of the tau sub-interval sign test results
    across signal types.

    Parameters
    ----------
    sign_test_by_signal : dict
        Dictionary mapping signal type to the output of
        sign_test_tau_subinterval(): {'acceleration': result_acc,
        'velocity': result_vel, 'displacement': result_disp}.

    Returns
    -------
    pd.DataFrame
        One row per signal type, with columns: signal_type, n_total,
        n_upper_greater, n_lower_greater, p_value.
    """
    signal_titles = {
        'acceleration': 'Acceleration',
        'velocity': 'Velocity',
        'displacement': 'Displacement',
    }
    rows = []
    for signal_type, result in sign_test_by_signal.items():
        rows.append({
            'signal_type': signal_titles.get(signal_type, signal_type),
            'n_total': result['n_total'],
            'n_upper_greater': result['n_upper_greater'],
            'n_lower_greater': result['n_lower_greater'],
            'p_value': result['p_value'],
        })
    return pd.DataFrame(rows)

def load_threshold_sensitivity_results(
    event_id: str,
    picker: str,
    config: str,
    threshold_tags: Dict[str, str],
    methods_by_run: Dict[str, List[str]],
) -> Dict[str, Dict[str, Dict]]:
    """
    Load moment scaling results for a set of coda threshold sensitivity
    runs, reusing load_scaling_results_by_signal for each (run, method)
    combination.

    Parameters
    ----------
    event_id : str
        Event identifier (e.g. 'IT-2009-0009').
    picker : str
        Picking method label (e.g. 'ar_pick').
    config : str
        Filter configuration label (e.g. 'no_filter').
    threshold_tags : dict
        Mapping from a human-readable run label to the THRESHOLD_TAG
        used when saving that run, e.g.
        {'baseline': '', 'thresh_env_020': '_env020', ...}.
    methods_by_run : dict
        Mapping from the same run labels to the list of coda methods
        saved under that run, e.g. {'baseline': ['rautian', 'arias',
        'envelope', 'median'], 'thresh_env_020': ['envelope', 'median']}.

    Returns
    -------
    dict
        {run_label: {coda_method: results_by_signal}}, where
        results_by_signal is the output of load_scaling_results_by_signal
        (keyed by signal type: 'acceleration', 'velocity', 'displacement').
    """
    results_by_threshold: Dict[str, Dict[str, Dict]] = {}

    for run_label, tag in threshold_tags.items():
        results_by_threshold[run_label] = {}
        for method in methods_by_run[run_label]:
            results_by_threshold[run_label][method] = load_scaling_results_by_signal(
                event_id=event_id,
                picker=picker,
                config=config,
                coda_method=method,
                threshold_tag=tag,
            )

    return results_by_threshold

def get_scaling_results_path(event_id: str, signal_type: str,
                              picker: str, config: str,
                              coda_method: str,
                              threshold_tag: str = '') -> Path:
    return (
        PROJECT_ROOT / 'data' / 'processed' / event_id
        / '04a_moment_scaling_spatial' / picker / config
        / f'{signal_type}{threshold_tag}' / coda_method
    )

def load_scaling_results_by_signal(event_id: str, picker: str,
                                    config: str, coda_method: str,
                                    threshold_tag: str = '',
                                    windows: tuple = ('p_wave', 's_wave', 'coda')
                                    ) -> Dict[str, Dict]:
    signal_types = ('acceleration', 'velocity', 'displacement')
    results_by_signal = {}
    for signal_type in signal_types:
        base_path = get_scaling_results_path(
            event_id, signal_type, picker, config, coda_method, threshold_tag
        )
        summary_path = base_path / 'ensemble_spatial_summary.parquet'
        if not summary_path.exists():
            results_by_signal[signal_type] = None
            continue
        df_summary = pd.read_parquet(summary_path)
        results = {}
        for window_name in windows:
            moments_path = base_path / f'ensemble_spatial_moments_{window_name}.parquet'
            if not moments_path.exists():
                results[window_name] = None
                continue
            df_moments = pd.read_parquet(moments_path)
            df_win = df_summary[df_summary['window'] == window_name]
            if df_win.empty:
                results[window_name] = None
                continue
            q_values = df_win['q'].values
            tau_values = df_moments['tau'].unique()
            tau_values.sort()
            moments_mean = np.zeros((len(tau_values), len(q_values)))
            for j, q_val in enumerate(q_values):
                df_q = df_moments[np.isclose(df_moments['q'], q_val)]
                for i, tau_val in enumerate(tau_values):
                    row = df_q[np.isclose(df_q['tau'], tau_val)]
                    if not row.empty:
                        moments_mean[i, j] = row['moment_mean'].values[0]
            results[window_name] = {
                'ensemble': {
                    'tau': tau_values,
                    'q': q_values,
                    'moments_mean': moments_mean,
                    'n_signals': df_win['n_signals'].values[0],
                },
                'scaling': {
                    'zeta': df_win['zeta'].values,
                    'zeta_err': df_win['zeta_err'].values,
                    'r_squared': df_win['r_squared'].values,
                    'intercepts': df_win['intercept'].values,
                    'n_points': df_win['n_points'].values,
                },
            }
        results_by_signal[signal_type] = results
    return results_by_signal

def compute_pointwise_zscore(
    zeta_baseline: np.ndarray,
    zeta_baseline_err: np.ndarray,
    zeta_alt: np.ndarray,
    zeta_alt_err: np.ndarray,
) -> np.ndarray:
    """
    Compute the pointwise z-score between two sets of scaling exponents.

    Implements Eq. (z_score_sensitivity): the absolute difference between
    two zeta(q) estimates, normalised by their combined standard error.

    Parameters
    ----------
    zeta_baseline : np.ndarray
        Baseline scaling exponents, one value per q.
    zeta_baseline_err : np.ndarray
        Standard errors of the baseline exponents, aligned with
        zeta_baseline.
    zeta_alt : np.ndarray
        Alternative-configuration scaling exponents, aligned with
        zeta_baseline by q.
    zeta_alt_err : np.ndarray
        Standard errors of the alternative-configuration exponents,
        aligned with zeta_alt.

    Returns
    -------
    np.ndarray
        Pointwise z-score for each q. NaN where either input is NaN.
    """
    combined_std = np.sqrt(zeta_baseline_err**2 + zeta_alt_err**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        z_score = np.abs(zeta_alt - zeta_baseline) / combined_std
    return z_score


def _load_summary_window(
    event_id: str,
    signal_type: str,
    picker: str,
    config: str,
    coda_method: str,
    window_name: str,
    threshold_tag: str = '',
) -> pd.DataFrame:
    """
    Load the zeta(q) summary for a single method, window, and threshold tag.

    Reuses get_scaling_results_path() for path construction, reading only
    the summary file (not the moments files), since sensitivity comparison
    operates on already-fitted exponents.

    Parameters
    ----------
    event_id : str
        Event identifier (e.g. 'IT-2009-0009').
    signal_type : str
        Signal type, e.g. 'acceleration'.
    picker : str
        Picking method label (e.g. 'ar_pick').
    config : str
        Filter configuration label (e.g. 'no_filter').
    coda_method : str
        Coda onset method: 'rautian', 'arias', 'envelope', or 'median'.
    window_name : str
        Analysis window to select, e.g. 's_wave' or 'coda'.
    threshold_tag : str, optional
        Threshold tag as returned by derive_threshold_run_config. Empty
        string selects the baseline run (default: '').

    Returns
    -------
    pd.DataFrame
        Summary rows for the requested window, indexed by q, with
        columns 'zeta' and 'zeta_err'.

    Raises
    ------
    FileNotFoundError
        If the expected summary file does not exist.
    """
    base_path = get_scaling_results_path(
        event_id, signal_type, picker, config, coda_method, threshold_tag
    )
    summary_path = base_path / 'ensemble_spatial_summary.parquet'
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df_summary = pd.read_parquet(summary_path)
    df_window = df_summary.loc[df_summary['window'] == window_name, ['q', 'zeta', 'zeta_err']]
    return df_window.set_index('q')


def compute_coda_threshold_sensitivity(
    event_id: str,
    signal_type: str,
    picker: str,
    config: str,
    threshold_configs: List[Dict[str, float]],
    windows: Tuple[str, ...] = ('s_wave', 'coda'),
    reference_q_values: Tuple[float, float] = (1.0, 2.0),
) -> pd.DataFrame:
    """
    Compute z-score sensitivity of moment scaling exponents to coda thresholds.

    For each alternative threshold configuration, compares the resulting
    zeta(q) spectrum against the baseline for every coda method affected
    by that configuration, over the requested windows.

    Parameters
    ----------
    event_id : str
        Event identifier (e.g. 'IT-2009-0009').
    signal_type : str
        Signal type to analyse, e.g. 'acceleration'.
    picker : str
        Picking method label (e.g. 'ar_pick').
    config : str
        Filter configuration label (e.g. 'no_filter').
    threshold_configs : list of dict
        Each dict has keys 'threshold_coda_onset' and 'threshold_coda_end',
        passed directly to derive_threshold_run_config. Exactly one of the
        two must differ from its baseline value per configuration.
    windows : tuple of str, optional
        Windows to evaluate (default: ('s_wave', 'coda')).
    reference_q_values : tuple of float, optional
        The two q values reported individually alongside z_max
        (default: (1.0, 2.0)).

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: threshold_type, threshold_value,
        method, window, z1, z2, z_max.

    Notes
    -----
    For each (configuration, method, window) combination, q values with
    an undefined zeta in either the baseline or the alternative fit are
    excluded from the z-score computation; the count of excluded values
    is printed, not included in the returned table.
    """
    q_low, q_high = reference_q_values
    records = []

    for run_config in threshold_configs:
        threshold_tag, affected_methods = derive_threshold_run_config(**run_config)
        if threshold_tag == '':
            raise ValueError(
                f"Configuration {run_config} matches the baseline; "
                "only alternative configurations should be passed."
            )

        baseline_onset = run_config.get('baseline_coda_onset', 0.30)
        onset_changed = not np.isclose(
            run_config['threshold_coda_onset'], baseline_onset
        )
        threshold_type = 'onset' if onset_changed else 'end'
        threshold_value = (
            run_config['threshold_coda_onset'] if onset_changed
            else run_config['threshold_coda_end']
        )

        for coda_method in affected_methods:
            for window_name in windows:
                baseline_summary = _load_summary_window(
                    event_id, signal_type, picker, config, coda_method,
                    window_name, threshold_tag='',
                )
                alt_summary = _load_summary_window(
                    event_id, signal_type, picker, config, coda_method,
                    window_name, threshold_tag=threshold_tag,
                )

                merged = baseline_summary.join(
                    alt_summary, how='inner', lsuffix='_base', rsuffix='_alt'
                )

                n_total = len(merged)
                valid = merged.dropna(subset=['zeta_base', 'zeta_alt'])
                n_excluded = n_total - len(valid)
                if n_excluded > 0:
                    print(
                        f"[{signal_type} | {threshold_type}={threshold_value} | "
                        f"{coda_method} | {window_name}] excluded "
                        f"{n_excluded}/{n_total} q values with undefined zeta "
                        "in baseline or alternative fit"
                    )

                z_score = compute_pointwise_zscore(
                    valid['zeta_base'].to_numpy(),
                    valid['zeta_err_base'].to_numpy(),
                    valid['zeta_alt'].to_numpy(),
                    valid['zeta_err_alt'].to_numpy(),
                )
                valid = valid.assign(z_score=z_score)

                z1 = valid['z_score'].get(q_low, np.nan)
                z2 = valid['z_score'].get(q_high, np.nan)
                z_max = valid['z_score'].max() if len(valid) > 0 else np.nan

                records.append({
                    'threshold_type': threshold_type,
                    'threshold_value': threshold_value,
                    'method': coda_method,
                    'window': window_name,
                    'z1': z1,
                    'z2': z2,
                    'z_max': z_max,
                })

    return pd.DataFrame.from_records(records)