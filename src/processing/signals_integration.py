"""
signals_integration.py
----------------------
Integration of seismic acceleration signals to velocity and displacement.

This module provides integration functions with support for both custom
implementations and ObsPy's optimized methods.

Main functions:
    DataFrame-based (full signals):
    - integrate_to_velocity: Single integration (acceleration → velocity)
    - integrate_to_displacement: Double integration (acceleration → displacement)
    
    Window-based (segmented signals):
    - integrate_windowed_signals_to_velocity: Integrate windowed signals to velocity
    - integrate_windowed_signals_to_displacement: Integrate windowed signals to displacement
    - validate_windowed_integration: Validate windowed integration results

Integration Methods:
    - 'trapz': Trapezoidal rule (scipy.integrate.cumulative_trapezoid)
    - 'obspy': ObsPy's integration (uses cumtrapz, optimized for seismic data)
    - 'euler': Simple Euler integration (less accurate, not recommended)

Notes:
    - Baseline correction MUST be applied before integration to avoid drift
    - Physical units preserved: acceleration (cm/s²) → velocity (cm/s) → displacement (cm)
    - ObsPy method is recommended when available (more robust for seismic data)
    - Windowed functions support initial condition propagation between windows

Usage:
    # Full signals (DataFrame)
    from src.signals_integration import integrate_to_velocity, integrate_to_displacement
    
    df_vel = integrate_to_velocity(df_acc, method='obspy')
    df_disp = integrate_to_displacement(df_acc, method='obspy')
    
    # Windowed signals (nested dict)
    from src.signals_integration import (
        integrate_windowed_signals_to_velocity,
        integrate_windowed_signals_to_displacement,
        validate_windowed_integration
    )
    
    windowed_vel = integrate_windowed_signals_to_velocity(
        windowed_signals, method='obspy', propagate_initial_conditions=True
    )
    
    windowed_disp = integrate_windowed_signals_to_displacement(
        windowed_signals, method='obspy', propagate_initial_conditions=True
    )
    
    stats = validate_windowed_integration(windowed_vel, check_field='velocity')
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

# ObsPy import (optional, with fallback)
try:
    from obspy import Trace
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False


# ===============================================================================================
# ==================================== Process Integration ======================================
# ===============================================================================================

def integrate_to_velocity(df_acc, dt=0.005, normalized=False, method='obspy'):
    """
    Integrate acceleration to obtain velocity: v(t) = ∫a(t)dt
    
    Supports both custom scipy implementation and ObsPy's optimized integration.
    
    Parameters
    ----------
    df_acc : pd.DataFrame
        Acceleration data with columns ['file', 'sample', 'acceleration']
        IMPORTANT: Baseline correction must be already applied
    dt : float, default=0.005
        Sampling interval in seconds (200 Hz → dt = 1/200 = 0.005s)
    normalized : bool, default=False
        If True, use 'acceleration_normalized' column
        If False, use 'acceleration' column (preserves physical units)
    method : str, default='trapz'
        Integration method:
        - 'trapz': Scipy trapezoidal rule (fast, accurate)
        - 'obspy': ObsPy integration (recommended for seismic data if available)
        - 'euler': Simple Euler method (less accurate, not recommended)
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'velocity' column (in cm/s)
        
    Examples
    --------
    >>> # Standard usage (preserves physical units)
    >>> df_vel = integrate_to_velocity(df_acc, method='trapz')
    >>> 
    >>> # With ObsPy (if available)
    >>> df_vel = integrate_to_velocity(df_acc, method='obspy')
    
    Notes
    -----
    CRITICAL: Baseline correction (mean removal) must be applied BEFORE integration
    to prevent drift. Use cleaning_signals.preprocess_signals() with 
    baseline_correction=True.
    
    Physical units:
        Input: acceleration in cm/s²
        Output: velocity in cm/s
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    
    if col not in df_acc.columns:
        raise ValueError(f"Column '{col}' not found in dataframe. "
                        f"Available columns: {df_acc.columns.tolist()}")
    
    df = df_acc.copy()
    
    if method == 'obspy' and OBSPY_AVAILABLE:
        # ObsPy integration (more robust for seismic data)
        def integrate_obspy(signal_array):
            trace = Trace(data=signal_array)
            trace.stats.delta = dt  # Sampling interval
            trace.integrate(method='cumtrapz')
            return trace.data
        
        df['velocity'] = df.groupby('file')[col].transform(integrate_obspy)
        
    elif method == 'obspy' and not OBSPY_AVAILABLE:
        # Fallback to trapz if ObsPy requested but not available
        print("Warning: ObsPy not available, falling back to 'trapz' method")
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: cumulative_trapezoid(a, dx=dt, initial=0)
        )
        
    elif method == 'trapz':
        # Scipy trapezoidal integration
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: cumulative_trapezoid(a, dx=dt, initial=0)
        )
        
    elif method == 'euler':
        # Simple Euler integration (less accurate)
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: np.cumsum(a) * dt
        )
        
    else:
        raise ValueError(f"method must be 'trapz', 'obspy', or 'euler', got '{method}'")
    
    return df


def integrate_to_displacement(df_acc, dt=0.005, normalized=False, method='obspy'):
    """
    Integrate acceleration twice to obtain displacement: x(t) = ∫∫a(t)dt²
    
    Performs double integration: acceleration → velocity → displacement.
    Supports both custom scipy implementation and ObsPy's optimized integration.
    
    Parameters
    ----------
    df_acc : pd.DataFrame
        Acceleration data with columns ['file', 'sample', 'acceleration']
        IMPORTANT: Baseline correction must be already applied
    dt : float, default=0.005
        Sampling interval in seconds (200 Hz)
    normalized : bool, default=False
        If True, use 'acceleration_normalized' column
        If False, use 'acceleration' column (preserves physical units)
    method : str, default='trapz'
        Integration method:
        - 'trapz': Scipy trapezoidal rule (fast, accurate)
        - 'obspy': ObsPy integration (recommended for seismic data if available)
        - 'euler': Simple Euler method (less accurate, not recommended)
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'velocity' and 'displacement' columns
        
    Examples
    --------
    >>> # Standard usage (preserves physical units)
    >>> df_disp = integrate_to_displacement(df_acc, method='trapz')
    >>> 
    >>> # With ObsPy (if available)
    >>> df_disp = integrate_to_displacement(df_acc, method='obspy')
    >>> 
    >>> # Check units
    >>> print("Acceleration range:", df_disp['acceleration'].min(), df_disp['acceleration'].max(), "cm/s²")
    >>> print("Velocity range:", df_disp['velocity'].min(), df_disp['velocity'].max(), "cm/s")
    >>> print("Displacement range:", df_disp['displacement'].min(), df_disp['displacement'].max(), "cm")
    
    Notes
    -----
    CRITICAL: Baseline correction (mean removal) must be applied BEFORE integration
    to prevent drift. A non-zero mean in acceleration causes linear drift in 
    velocity and quadratic drift in displacement.
    
    Physical units:
        Input: acceleration in cm/s²
        Output: velocity in cm/s, displacement in cm
        
    The function performs two sequential integrations:
        1. a(t) → v(t) = ∫a(t)dt
        2. v(t) → x(t) = ∫v(t)dt
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    
    if col not in df_acc.columns:
        raise ValueError(f"Column '{col}' not found in dataframe. "
                        f"Available columns: {df_acc.columns.tolist()}")
    
    df = df_acc.copy()
    
    if method == 'obspy' and OBSPY_AVAILABLE:
        # ObsPy double integration
        def integrate_double_obspy(signal_array):
            trace = Trace(data=signal_array)
            trace.stats.delta = dt
            
            # First integration: a → v
            trace.integrate(method='cumtrapz')
            velocity = trace.data.copy()
            
            # Second integration: v → x
            trace.integrate(method='cumtrapz')
            displacement = trace.data
            
            return pd.Series({'velocity': velocity, 'displacement': displacement})
        
        # Apply and unpack
        result = df.groupby('file')[col].apply(
            lambda group: pd.DataFrame({
                'velocity': integrate_double_obspy(group.values)['velocity'],
                'displacement': integrate_double_obspy(group.values)['displacement']
            })
        ).reset_index(level=0, drop=True)
        
        df['velocity'] = result['velocity'].values
        df['displacement'] = result['displacement'].values
        
    elif method == 'obspy' and not OBSPY_AVAILABLE:
        # Fallback to trapz if ObsPy requested but not available
        print("Warning: ObsPy not available, falling back to 'trapz' method")
        
        # First integration: a → v
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: cumulative_trapezoid(a, dx=dt, initial=0)
        )
        
        # Second integration: v → x
        df['displacement'] = df.groupby('file')['velocity'].transform(
            lambda v: cumulative_trapezoid(v, dx=dt, initial=0)
        )
        
    elif method == 'trapz':
        # Scipy trapezoidal integration (double)
        
        # First integration: a → v
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: cumulative_trapezoid(a, dx=dt, initial=0)
        )
        
        # Second integration: v → x
        df['displacement'] = df.groupby('file')['velocity'].transform(
            lambda v: cumulative_trapezoid(v, dx=dt, initial=0)
        )
        
    elif method == 'euler':
        # Simple Euler integration (double)
        
        # First integration: a → v
        df['velocity'] = df.groupby('file')[col].transform(
            lambda a: np.cumsum(a) * dt
        )
        
        # Second integration: v → x
        df['displacement'] = df.groupby('file')['velocity'].transform(
            lambda v: np.cumsum(v) * dt
        )
        
    else:
        raise ValueError(f"method must be 'trapz', 'obspy', or 'euler', got '{method}'")
    
    return df

# ===============================================================================================
# ==================================== Validation ===============================================
# ===============================================================================================

def validate_integration(df_integrated, process='velocity'):
    """
    Validate integration results with quality checks.
    
    Parameters
    ----------
    df_integrated : pd.DataFrame
        Dataframe with integrated signals
    process : str
        Which process to validate: 'velocity' or 'displacement'
    
    Returns
    -------
    bool
        True if validation passes
    
    Raises
    ------
    AssertionError
        If validation fails
    """
    print(f"Validating {process} integration...")
    
    # Check column exists
    assert process in df_integrated.columns, f"Column '{process}' not found"
    print(f"Column '{process}' exists")
    
    # Check no NaN
    n_nan = df_integrated[process].isna().sum()
    assert n_nan == 0, f"Found {n_nan} NaN values in {process}"
    print(f"  ✓ No NaN values")
    
    # Check no Inf
    n_inf = np.isinf(df_integrated[process]).sum()
    assert n_inf == 0, f"Found {n_inf} Inf values in {process}"
    print(f"  ✓ No Inf values")
    
    # Check reasonable range (for displacement/velocity)
    if process == 'velocity':
        max_vel = df_integrated[process].abs().max()
        assert max_vel < 1000, f"Velocity too large: {max_vel:.2f} cm/s (check baseline correction)"
        print(f"Velocity range reasonable: max = {max_vel:.2f} cm/s")
    
    if process == 'displacement':
        max_disp = df_integrated[process].abs().max()
        assert max_disp < 10000, f"Displacement too large: {max_disp:.2f} cm (check baseline correction)"
        print(f"Displacement range reasonable: max = {max_disp:.2f} cm")
    
    # Check baseline (mean should be ~0 if properly preprocessed)
    mean_val = df_integrated.groupby('file')[process].mean().abs().max()
    if mean_val > 1.0:
        print(f"  ⚠ Warning: Large mean detected ({mean_val:.2f}), baseline may not be properly corrected")
    else:
        print(f"Baseline check passed (max mean: {mean_val:.4f})")
    
    print(f"Validation passed!\n")
    return True

# ===============================================================================================
# ========================== Windowed Signals Integration ======================================
# ===============================================================================================

def integrate_windowed_signals_to_velocity(
    windowed_signals,
    dt=0.005,
    method='obspy',
    apply_baseline=False,
    propagate_initial_conditions=True
):
    """
    Integrate windowed acceleration signals to velocity.
    
    Integrates each temporal window (pre_event, p_wave, s_wave, coda)
    independently and optionally propagates final conditions between windows.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals():
        {station: {component: {window_name: {'signal': array, 'time': array, ...}}}}
    dt : float, default=0.005
        Sampling interval in seconds (200 Hz)
    method : str, default='obspy'
        Integration method:
        - 'trapz': Scipy trapezoidal rule
        - 'obspy': ObsPy integration (recommended)
        - 'euler': Simple Euler method
    apply_baseline : bool, default=False
        If True, apply baseline correction (mean removal) to each window
        before integration. Use only if baseline was not already corrected
        on the full signal before segmentation.
    propagate_initial_conditions : bool, default=True
        If True, propagate final velocity from one window as initial
        condition for the next window (maintains continuity).
        Order: pre_event → p_wave → s_wave → coda
        If False, each window starts from v₀=0 independently.
        
    Returns
    -------
    dict
        Same nested structure with added 'velocity' array in each window
        
    Examples
    --------
    >>> # Standard usage (recommended)
    >>> windowed_vel = integrate_windowed_signals_to_velocity(
    ...     windowed_signals, method='obspy', propagate_initial_conditions=True
    ... )
    >>> 
    >>> # Access velocity for a specific window
    >>> vel = windowed_vel['AMT']['HGE']['s_wave']['velocity']
    >>> time = windowed_vel['AMT']['HGE']['s_wave']['time']
    
    Notes
    -----
    Baseline correction should already be applied to the full signal before
    segmentation. Set apply_baseline=True only if you need to correct each
    window independently (e.g., for comparison purposes).
    
    When propagate_initial_conditions=True, missing windows are skipped
    and the last valid final condition is propagated forward.
    """
    if method == 'obspy' and not OBSPY_AVAILABLE:
        print("Warning: ObsPy not available, falling back to 'trapz' method")
        method = 'trapz'
    
    integrated = {}
    window_order = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    for station in windowed_signals:
        integrated[station] = {}
        
        for component in windowed_signals[station]:
            integrated[station][component] = {}
            
            v_final = 0.0
            
            for window_name in window_order:
                if window_name not in windowed_signals[station][component]:
                    print(f"Warning: {station}-{component} missing window '{window_name}', "
                          f"propagating last valid condition")
                    continue
                
                window = windowed_signals[station][component][window_name].copy()
                signal = window['signal']
                
                if len(signal) == 0:
                    print(f"Warning: {station}-{component}-{window_name} is empty, skipping")
                    integrated[station][component][window_name] = window
                    continue
                
                if apply_baseline:
                    signal = signal - np.mean(signal)
                
                v0 = v_final if propagate_initial_conditions else 0.0
                
                if method == 'obspy':
                    trace = Trace(data=signal)
                    trace.stats.delta = dt
                    trace.integrate(method='cumtrapz')
                    velocity = trace.data + v0
                    
                elif method == 'trapz':
                    velocity = cumulative_trapezoid(signal, dx=dt, initial=0) + v0
                    
                elif method == 'euler':
                    velocity = np.cumsum(signal) * dt + v0
                    
                else:
                    raise ValueError(f"method must be 'trapz', 'obspy', or 'euler', got '{method}'")
                
                window['velocity'] = velocity
                integrated[station][component][window_name] = window
                
                v_final = velocity[-1]
    
    return integrated


def integrate_windowed_signals_to_displacement(
    windowed_signals,
    dt=0.005,
    method='obspy',
    apply_baseline=False,
    propagate_initial_conditions=True
):
    """
    Integrate windowed acceleration signals to displacement.
    
    Performs double integration (acceleration → velocity → displacement)
    for each temporal window, with optional propagation of initial conditions.
    
    Parameters
    ----------
    windowed_signals : dict
        Output from segment_all_signals():
        {station: {component: {window_name: {'signal': array, 'time': array, ...}}}}
    dt : float, default=0.005
        Sampling interval in seconds (200 Hz)
    method : str, default='obspy'
        Integration method:
        - 'trapz': Scipy trapezoidal rule
        - 'obspy': ObsPy integration (recommended)
        - 'euler': Simple Euler method
    apply_baseline : bool, default=False
        If True, apply baseline correction (mean removal) to each window
        before integration. Use only if baseline was not already corrected
        on the full signal before segmentation.
    propagate_initial_conditions : bool, default=True
        If True, propagate final velocity and displacement from one window
        as initial conditions for the next window (maintains continuity).
        Order: pre_event → p_wave → s_wave → coda
        If False, each window starts from v₀=0, x₀=0 independently.
        
    Returns
    -------
    dict
        Same nested structure with added 'velocity' and 'displacement' arrays
        in each window
        
    Examples
    --------
    >>> # Standard usage (recommended)
    >>> windowed_disp = integrate_windowed_signals_to_displacement(
    ...     windowed_signals, method='obspy', propagate_initial_conditions=True
    ... )
    >>> 
    >>> # Access displacement for a specific window
    >>> disp = windowed_disp['AMT']['HGE']['coda']['displacement']
    >>> vel = windowed_disp['AMT']['HGE']['coda']['velocity']
    >>> time = windowed_disp['AMT']['HGE']['coda']['time']
    
    Notes
    -----
    Baseline correction should already be applied to the full signal before
    segmentation. Set apply_baseline=True only if you need to correct each
    window independently.
    
    When propagate_initial_conditions=True, missing windows are skipped
    and the last valid final conditions (v_final, x_final) are propagated.
    
    Physical units:
        Input: acceleration in cm/s²
        Output: velocity in cm/s, displacement in cm
    """
    if method == 'obspy' and not OBSPY_AVAILABLE:
        print("Warning: ObsPy not available, falling back to 'trapz' method")
        method = 'trapz'
    
    integrated = {}
    window_order = ['pre_event', 'p_wave', 's_wave', 'coda']
    
    for station in windowed_signals:
        integrated[station] = {}
        
        for component in windowed_signals[station]:
            integrated[station][component] = {}
            
            v_final = 0.0
            x_final = 0.0
            
            for window_name in window_order:
                if window_name not in windowed_signals[station][component]:
                    print(f"Warning: {station}-{component} missing window '{window_name}', "
                          f"propagating last valid condition")
                    continue
                
                window = windowed_signals[station][component][window_name].copy()
                signal = window['signal']
                
                if len(signal) == 0:
                    print(f"Warning: {station}-{component}-{window_name} is empty, skipping")
                    integrated[station][component][window_name] = window
                    continue
                
                if apply_baseline:
                    signal = signal - np.mean(signal)
                
                v0 = v_final if propagate_initial_conditions else 0.0
                x0 = x_final if propagate_initial_conditions else 0.0
                
                if method == 'obspy':
                    trace = Trace(data=signal)
                    trace.stats.delta = dt
                    
                    trace.integrate(method='cumtrapz')
                    velocity = trace.data + v0
                    
                    trace.integrate(method='cumtrapz')
                    displacement = trace.data + x0
                    
                elif method == 'trapz':
                    velocity = cumulative_trapezoid(signal, dx=dt, initial=0) + v0
                    displacement = cumulative_trapezoid(velocity, dx=dt, initial=0) + x0
                    
                elif method == 'euler':
                    velocity = np.cumsum(signal) * dt + v0
                    displacement = np.cumsum(velocity) * dt + x0
                    
                else:
                    raise ValueError(f"method must be 'trapz', 'obspy', or 'euler', got '{method}'")
                
                window['velocity'] = velocity
                window['displacement'] = displacement
                integrated[station][component][window_name] = window
                
                v_final = velocity[-1]
                x_final = displacement[-1]
    
    return integrated


def validate_windowed_integration(windowed_integrated, check_field='velocity'):
    """
    Validate integration results for windowed signals.
    
    Checks for NaN, Inf, and physically reasonable ranges across all
    windows in the nested structure.
    
    Parameters
    ----------
    windowed_integrated : dict
        Output from integrate_windowed_signals_to_velocity() or
        integrate_windowed_signals_to_displacement()
    check_field : str, default='velocity'
        Which field to validate: 'velocity' or 'displacement'
        
    Returns
    -------
    dict
        Validation statistics with keys:
        - 'n_windows': total number of windows checked
        - 'n_nan': number of windows with NaN values
        - 'n_inf': number of windows with Inf values
        - 'max_value': maximum absolute value across all windows
        - 'mean_of_means': average of window means (baseline check)
        - 'passed': True if validation passed
        
    Examples
    --------
    >>> stats = validate_windowed_integration(windowed_vel, check_field='velocity')
    >>> print(f"Validation passed: {stats['passed']}")
    >>> 
    >>> stats_disp = validate_windowed_integration(windowed_disp, check_field='displacement')
    """
    print(f"Validating windowed {check_field} integration...")
    
    n_windows = 0
    n_nan = 0
    n_inf = 0
    max_value = 0.0
    all_means = []
    
    for station in windowed_integrated:
        for component in windowed_integrated[station]:
            for window_name in windowed_integrated[station][component]:
                window = windowed_integrated[station][component][window_name]
                
                if check_field not in window:
                    print(f"Warning: {station}-{component}-{window_name} "
                          f"missing '{check_field}' field, skipping")
                    continue
                
                data = window[check_field]
                n_windows += 1
                
                if np.isnan(data).any():
                    n_nan += 1
                    print(f"  ⚠ NaN found in {station}-{component}-{window_name}")
                
                if np.isinf(data).any():
                    n_inf += 1
                    print(f"  ⚠ Inf found in {station}-{component}-{window_name}")
                
                max_value = max(max_value, np.abs(data).max())
                all_means.append(np.mean(data))
    
    mean_of_means = np.mean(np.abs(all_means)) if all_means else 0.0
    
    print(f"\nValidation Summary:")
    print(f"  Total windows checked: {n_windows}")
    print(f"  Windows with NaN: {n_nan}")
    print(f"  Windows with Inf: {n_inf}")
    print(f"  Max absolute value: {max_value:.2f}")
    print(f"  Mean of window means: {mean_of_means:.4f}")
    
    passed = True
    
    if n_nan > 0:
        print(f"Validation FAILED: Found NaN values")
        passed = False
    else:
        print(f"No NaN values")
    
    if n_inf > 0:
        print(f"Validation FAILED: Found Inf values")
        passed = False
    else:
        print(f"No Inf values")
    
    if check_field == 'velocity' and max_value > 1000:
        print(f"Warning: Very large velocity ({max_value:.2f} cm/s), check baseline")
    elif check_field == 'displacement' and max_value > 10000:
        print(f"Warning: Very large displacement ({max_value:.2f} cm), check baseline")
    else:
        print(f"{check_field.capitalize()} range reasonable")
    
    if mean_of_means > 1.0:
        print(f"Warning: Large mean detected, baseline may not be properly corrected")
    else:
        print(f"Baseline check passed")
    
    if passed:
        print(f"\nValidation PASSED\n")
    else:
        print(f"\nValidation FAILED\n")
    
    return {
        'n_windows': n_windows,
        'n_nan': n_nan,
        'n_inf': n_inf,
        'max_value': max_value,
        'mean_of_means': mean_of_means,
        'passed': passed
    }