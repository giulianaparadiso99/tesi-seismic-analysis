import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
from src.visualization.plot_settings import set_plot_style
colors = set_plot_style()

"""
signals_scaling_temporal.py
---------------------------
Moment scaling analysis using temporal ensemble averaging.

Computes moments M_q(τ) by averaging over multiple temporal offsets t₀
within each seismic window, rather than averaging across spatial stations.

Key difference from spatial ensemble (signals_scaling_ensemble.py):
- Spatial: ⟨...⟩ over different stations/components
- Temporal: ⟨...⟩ over different starting times t₀ within same window

Constraint: t₀ + τ must stay within window boundaries, so the number
of valid offsets decreases as τ increases.
"""


def compute_temporal_ensemble_moments(
    signal: np.ndarray,
    tau_indices: np.ndarray,
    q_values: np.ndarray,
    n_t0_offsets: int = 100,
    save_increments: bool = False
) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Compute moments M_q(τ) using temporal ensemble averaging.
    
    For each τ, computes increments starting from multiple time offsets:
        Δa(τ, t₀) = a(t₀ + τ) - a(t₀)
    
    Then averages moments over valid t₀ offsets:
        M_q(τ) = ⟨|Δa(τ, t₀)|^q⟩_{t₀}
    
    Constraint: t₀ + τ must stay within signal length, so number of valid
    offsets decreases as τ increases.
    
    Parameters
    ----------
    signal : np.ndarray
        Time series (acceleration, velocity, or displacement)
    tau_indices : np.ndarray
        Array of time lag indices (in samples)
    q_values : np.ndarray
        Array of moment orders
    n_t0_offsets : int, optional
        Maximum number of temporal offsets to use (default: 100)
    save_increments : bool, optional
        If True, also return increments for each τ (default: False)
    
    Returns
    -------
    moments : np.ndarray
        Shape (n_tau, n_q) containing averaged moments
    increments_list : List[np.ndarray] or None
        If save_increments=True: list of length n_tau, where
        increments_list[i] = array of valid increments for tau_indices[i]
        If save_increments=False: None
    
    Notes
    -----
    Number of valid offsets per τ:
        n_valid(τ) = min(n_t0_offsets, len(signal) - τ)
    
    For large τ, fewer offsets are available.
    """
    n_tau = len(tau_indices)
    n_q = len(q_values)
    
    moments = np.zeros((n_tau, n_q))
    increments_list = [] if save_increments else None
    
    for i, tau_idx in enumerate(tau_indices):
        # Maximum t0 such that t0 + tau_idx < len(signal)
        t0_max = len(signal) - tau_idx
        
        # Number of valid offsets
        n_offsets_valid = min(n_t0_offsets, t0_max)
        
        if n_offsets_valid < 1:
            warnings.warn(
                f"tau_idx={tau_idx} too large for signal length={len(signal)}, "
                f"no valid offsets available"
            )
            moments[i, :] = np.nan
            if save_increments:
                increments_list.append(np.array([]))
            continue
        
        # Compute increments for all valid t0 offsets
        increments = np.zeros(n_offsets_valid)
        for t0_offset in range(n_offsets_valid):
            increments[t0_offset] = signal[t0_offset + tau_idx] - signal[t0_offset]
        
        # Save increments if requested
        if save_increments:
            increments_list.append(increments.copy())
        
        # Compute moments: average over t0 offsets
        abs_increments = np.abs(increments)
        
        for j, q in enumerate(q_values):
            moments[i, j] = np.mean(abs_increments ** q)
    
    return moments, increments_list


def compute_ensemble_single_window_temporal(
    signals_dict: Dict,
    stations: List[str],
    components: List[str],
    window_name: str,
    tau_indices: np.ndarray,
    q_values: np.ndarray,
    n_t0_offsets: int = 100,
    save_increments: bool = False
) -> Dict:
    """
    Compute temporal ensemble moments for a single window across all stations/components.
    
    Parameters
    ----------
    signals_dict : dict
        Nested dict: signals_dict[station][component][window_name]
        Each contains 'signal', 'time', etc.
    stations : list of str
        Station codes to include
    components : list of str
        Component codes to include
    window_name : str
        Window name: 'pre_event', 'p_wave', 's_wave', or 'coda'
    tau_indices : np.ndarray
        Time lag indices
    q_values : np.ndarray
        Moment orders
    n_t0_offsets : int, optional
        Number of temporal offsets (default: 100)
    save_increments : bool, optional
        Whether to save increments (default: False)
    
    Returns
    -------
    ensemble_dict : dict
        Contains:
        - 'moments_all': list of moment arrays (one per signal)
        - 'moments_mean': mean over signals, shape (n_tau, n_q)
        - 'moments_std': std over signals
        - 'n_offsets_per_tau': array of valid offsets per tau (averaged)
        - 'increments_all': list of lists (if save_increments=True)
        - 'n_signals': number of signals processed
    """
    moments_all = []
    increments_all = [] if save_increments else None
    n_offsets_all = []
    
    for station in stations:
        if station not in signals_dict:
            continue
        
        for component in components:
            if component not in signals_dict[station]:
                continue
            
            if window_name not in signals_dict[station][component]:
                continue
            
            window_data = signals_dict[station][component][window_name]
            signal = window_data['signal']
            
            if len(signal) < 10:
                warnings.warn(
                    f"Skipping {station}_{component}_{window_name}: "
                    f"insufficient samples ({len(signal)})"
                )
                continue
            
            # Compute moments for this signal
            moments, increments_list = compute_temporal_ensemble_moments(
                signal=signal,
                tau_indices=tau_indices,
                q_values=q_values,
                n_t0_offsets=n_t0_offsets,
                save_increments=save_increments
            )
            
            moments_all.append(moments)
            
            if save_increments:
                increments_all.append(increments_list)
            
            # Track number of valid offsets per tau
            n_offsets = np.array([
                len(inc) if inc is not None and len(inc) > 0 else 0
                for inc in (increments_list if increments_list else [np.array([]) for _ in tau_indices])
            ])
            n_offsets_all.append(n_offsets)
    
    if len(moments_all) == 0:
        warnings.warn(f"No valid signals for window '{window_name}'")
        return None
    
    # Convert to arrays
    moments_all = np.array(moments_all)  # shape: (n_signals, n_tau, n_q)
    n_offsets_all = np.array(n_offsets_all)  # shape: (n_signals, n_tau)
    
    # Ensemble statistics: average over signals
    moments_mean = np.nanmean(moments_all, axis=0)  # shape: (n_tau, n_q)
    moments_std = np.nanstd(moments_all, axis=0)
    n_offsets_mean = np.mean(n_offsets_all, axis=0)  # Average offsets per tau
    
    ensemble_dict = {
        'moments_all': moments_all,
        'moments_mean': moments_mean,
        'moments_std': moments_std,
        'n_offsets_per_tau': n_offsets_mean,
        'n_signals': len(moments_all)
    }
    
    if save_increments:
        ensemble_dict['increments_all'] = increments_all
    
    return ensemble_dict


def compute_scaling_exponents_temporal(
    tau: np.ndarray,
    moments_mean: np.ndarray,
    q_values: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Compute scaling exponents ζ(q) from moments M_q(τ).
    
    Fits: log(M_q) = ζ(q) × log(τ) + intercept
    
    Parameters
    ----------
    tau : np.ndarray
        Time lags (in seconds)
    moments_mean : np.ndarray
        Mean moments, shape (n_tau, n_q)
    q_values : np.ndarray
        Moment orders
    fit_range : tuple of float, optional
        (tau_min, tau_max) for fitting. If None, uses all points.
    
    Returns
    -------
    scaling_dict : dict
        Contains:
        - 'zeta': scaling exponents, shape (n_q,)
        - 'zeta_err': standard errors
        - 'intercepts': fit intercepts
        - 'r_squared': R² values
        - 'fit_range': actual range used
    """
    n_q = len(q_values)
    
    zeta = np.zeros(n_q)
    zeta_err = np.zeros(n_q)
    intercepts = np.zeros(n_q)
    r_squared = np.zeros(n_q)
    
    # Determine fit range
    if fit_range is not None:
        tau_min, tau_max = fit_range
        mask = (tau >= tau_min) & (tau <= tau_max)
    else:
        mask = np.ones(len(tau), dtype=bool)
    
    log_tau = np.log10(tau[mask])
    
    for i in range(n_q):
        M_q = moments_mean[mask, i]
        
        # Filter out invalid values
        valid = (M_q > 0) & np.isfinite(M_q) & np.isfinite(log_tau)
        
        if valid.sum() < 3:
            warnings.warn(f"Insufficient valid points for q={q_values[i]:.2f}")
            zeta[i] = np.nan
            zeta_err[i] = np.nan
            intercepts[i] = np.nan
            r_squared[i] = np.nan
            continue
        
        log_M_q = np.log10(M_q[valid])
        log_tau_valid = log_tau[valid]
        
        # Linear fit
        coeffs = np.polyfit(log_tau_valid, log_M_q, 1)
        slope, intercept = coeffs
        
        # Compute R²
        y_fit = slope * log_tau_valid + intercept
        residuals = log_M_q - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_M_q - np.mean(log_M_q))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Standard error
        n_points = len(log_tau_valid)
        se = np.sqrt(ss_res / (n_points - 2)) if n_points > 2 else np.nan
        x_var = np.sum((log_tau_valid - np.mean(log_tau_valid))**2)
        slope_err = se / np.sqrt(x_var) if x_var > 0 else np.nan
        
        zeta[i] = slope
        zeta_err[i] = slope_err
        intercepts[i] = intercept
        r_squared[i] = r2
    
    return {
        'zeta': zeta,
        'zeta_err': zeta_err,
        'intercepts': intercepts,
        'r_squared': r_squared,
        'fit_range': fit_range
    }


def analyze_all_windows_temporal(
    windowed_signals: Dict,
    tau_min: float = 0.01,
    n_tau: Optional[int] = None,
    q_values: Optional[np.ndarray] = None,
    sampling_rate: float = 200.0,
    n_t0_offsets: int = 100,
    save_increments: bool = False,
    fit_range: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Analyze moment scaling with temporal ensemble for all windows.
    
    Parameters
    ----------
    windowed_signals : dict
        Nested dict: windowed_signals[station][component][window_name]
    tau_min : float, optional
        Minimum time lag in seconds (default: 0.01)
    n_tau : int, optional
        Number of tau values. If None, auto-determined.
    q_values : np.ndarray, optional
        Moment orders. If None, uses default range.
    sampling_rate : float, optional
        Sampling rate in Hz (default: 200)
    n_t0_offsets : int, optional
        Number of temporal offsets (default: 100)
    save_increments : bool, optional
        Whether to save increments (default: False)
    fit_range : tuple of float, optional
        (tau_min, tau_max) for scaling exponent fits
    
    Returns
    -------
    results : dict
        Nested dict with structure:
        results[window_name] = {
            'ensemble': {...},
            'scaling': {...},
            'increments': {...}  # if save_increments=True
        }
    """
    dt = 1.0 / sampling_rate
    
    # Default q values
    if q_values is None:
        q_values = np.linspace(0.5, 5.0, 19)
    
    # Get all stations and components
    stations = list(windowed_signals.keys())
    components_set = set()
    for station in stations:
        components_set.update(windowed_signals[station].keys())
    components = sorted(list(components_set))
    
    # Determine tau values from longest window
    if n_tau is None:
        max_duration = 0
        for station in stations:
            for component in components:
                if component not in windowed_signals[station]:
                    continue
                for window_name in ['pre_event', 'p_wave', 's_wave', 'coda']:
                    if window_name not in windowed_signals[station][component]:
                        continue
                    duration = windowed_signals[station][component][window_name]['duration']
                    max_duration = max(max_duration, duration)
        
        tau_max = max_duration / 2.0
        n_tau = int(np.log10(tau_max / tau_min) * 10) + 1
    else:
        tau_max = None
        for station in stations:
            for component in components:
                if component not in windowed_signals[station]:
                    continue
                for window_name in ['pre_event', 'p_wave', 's_wave', 'coda']:
                    if window_name not in windowed_signals[station][component]:
                        continue
                    duration = windowed_signals[station][component][window_name]['duration']
                    if tau_max is None or duration / 2.0 > tau_max:
                        tau_max = duration / 2.0
    
    tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)
    tau_indices = (tau_values / dt).astype(int)
    
    # Analyze each window
    windows = ['pre_event', 'p_wave', 's_wave', 'coda']
    results = {}
    
    for window_name in windows:
        print(f"\nProcessing window: {window_name}")
        
        ensemble = compute_ensemble_single_window_temporal(
            signals_dict=windowed_signals,
            stations=stations,
            components=components,
            window_name=window_name,
            tau_indices=tau_indices,
            q_values=q_values,
            n_t0_offsets=n_t0_offsets,
            save_increments=save_increments
        )
        
        if ensemble is None:
            print(f"  No valid data for {window_name}, skipping")
            results[window_name] = None
            continue
        
        print(f"  Processed {ensemble['n_signals']} signals")
        print(f"  Mean offsets per tau: {ensemble['n_offsets_per_tau'].mean():.1f}")
        
        # Compute scaling exponents
        scaling = compute_scaling_exponents_temporal(
            tau=tau_values,
            moments_mean=ensemble['moments_mean'],
            q_values=q_values,
            fit_range=fit_range
        )
        
        # Store results
        results[window_name] = {
            'ensemble': {
                'tau': tau_values,
                'q': q_values,
                'moments_mean': ensemble['moments_mean'],
                'moments_std': ensemble['moments_std'],
                'n_offsets_per_tau': ensemble['n_offsets_per_tau'],
                'n_signals': ensemble['n_signals']
            },
            'scaling': scaling
        }
        
        if save_increments:
            results[window_name]['increments'] = {
                'increments_list': ensemble['increments_all'],
                'tau_values': tau_values
            }
    
    return results