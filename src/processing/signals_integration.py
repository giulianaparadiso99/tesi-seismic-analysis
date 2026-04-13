"""
signals_integration.py
----------------------
Integration of seismic acceleration signals to velocity and displacement.

This module provides integration functions with support for both custom
implementations and ObsPy's optimized methods.

Main functions:
    - integrate_to_velocity: Single integration (acceleration → velocity)
    - integrate_to_displacement: Double integration (acceleration → displacement)

Integration Methods:
    - 'trapz': Trapezoidal rule (scipy.integrate.cumulative_trapezoid)
    - 'obspy': ObsPy's integration (uses cumtrapz, optimized for seismic data)

Notes:
    - Baseline correction MUST be applied before integration to avoid drift
    - Physical units preserved: acceleration (cm/s²) → velocity (cm/s) → displacement (cm)
    - ObsPy method is recommended when available (more robust for seismic data)

Usage:
    from src.signals_integration import integrate_to_velocity, integrate_to_displacement
    
    # Velocity
    df_vel = integrate_to_velocity(df_acc, method='obspy')
    
    # Displacement
    df_disp = integrate_to_displacement(df_acc, method='obspy')
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

def integrate_to_velocity(df_acc, dt=0.005, normalized=False, method='trapz'):
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


def integrate_to_displacement(df_acc, dt=0.005, normalized=False, method='trapz'):
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