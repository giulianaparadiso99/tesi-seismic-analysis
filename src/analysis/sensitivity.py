"""
Utility functions for sensitivity analysis of phase picking errors on moment scaling.

This module provides:
- Perturbation functions for different error scenarios
- Metrics computation for comparing original vs perturbed results
- Result aggregation and summary statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import pickle


# ==============================================================================
# PERTURBATION FUNCTIONS
# ==============================================================================

def perturb_picks_gaussian(
    df_picks: pd.DataFrame,
    noise_std: float,
    sampling_rate: float = 200.0,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Perturb P and S picks with Gaussian noise.
    
    Parameters
    ----------
    df_picks : pd.DataFrame
        DataFrame with columns 't_p_detected_seconds', 't_s_detected_seconds'
    noise_std : float
        Standard deviation of Gaussian noise (seconds)
    sampling_rate : float
        Sampling rate for converting to samples
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    df_perturbed : pd.DataFrame
        DataFrame with perturbed picks, maintaining physical constraints
    """
    
    df_perturbed = df_picks.copy()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_stations = len(df_perturbed)
    
    # Generate noise
    noise_p = np.random.normal(0, noise_std, n_stations)
    noise_s = np.random.normal(0, noise_std, n_stations)
    
    # Apply noise
    df_perturbed['t_p_detected_seconds'] = df_picks['t_p_detected_seconds'] + noise_p
    df_perturbed['t_s_detected_seconds'] = df_picks['t_s_detected_seconds'] + noise_s
    
    # Apply physical constraints
    df_perturbed = apply_physical_constraints(df_perturbed, sampling_rate)
    
    # Update sample indices
    df_perturbed['t_p_detected_samples'] = (
        df_perturbed['t_p_detected_seconds'] * sampling_rate
    ).astype(int)
    df_perturbed['t_s_detected_samples'] = (
        df_perturbed['t_s_detected_seconds'] * sampling_rate
    ).astype(int)
    
    return df_perturbed


def perturb_picks_bias(
    df_picks: pd.DataFrame,
    bias_seconds: float,
    sampling_rate: float = 200.0
) -> pd.DataFrame:
    """
    Perturb P and S picks with systematic bias.
    
    Parameters
    ----------
    df_picks : pd.DataFrame
        DataFrame with picks
    bias_seconds : float
        Systematic bias to add (seconds). Positive = delay, negative = advance
    sampling_rate : float
        Sampling rate
        
    Returns
    -------
    df_perturbed : pd.DataFrame
        DataFrame with biased picks
    """
    
    df_perturbed = df_picks.copy()
    
    # Apply bias
    df_perturbed['t_p_detected_seconds'] = df_picks['t_p_detected_seconds'] + bias_seconds
    df_perturbed['t_s_detected_seconds'] = df_picks['t_s_detected_seconds'] + bias_seconds
    
    # Apply constraints
    df_perturbed = apply_physical_constraints(df_perturbed, sampling_rate)
    
    # Update samples
    df_perturbed['t_p_detected_samples'] = (
        df_perturbed['t_p_detected_seconds'] * sampling_rate
    ).astype(int)
    df_perturbed['t_s_detected_samples'] = (
        df_perturbed['t_s_detected_seconds'] * sampling_rate
    ).astype(int)
    
    return df_perturbed


def apply_physical_constraints(
    df: pd.DataFrame,
    sampling_rate: float
) -> pd.DataFrame:
    """
    Apply physical constraints to perturbed picks.
    
    Constraints:
    - t_p ≥ 0
    - t_s > t_p (S must come after P)
    - Both within signal duration (if duration available)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with perturbed picks
    sampling_rate : float
        Sampling rate
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with constrained picks
    """
    
    df = df.copy()
    
    # Constraint 1: t_p ≥ 0
    df['t_p_detected_seconds'] = df['t_p_detected_seconds'].clip(lower=0)
    
    # Constraint 2: t_s > t_p (add minimum gap of 1 sample)
    min_gap = 1.0 / sampling_rate
    mask_violation = df['t_s_detected_seconds'] <= df['t_p_detected_seconds']
    df.loc[mask_violation, 't_s_detected_seconds'] = (
        df.loc[mask_violation, 't_p_detected_seconds'] + min_gap
    )
    
    # Constraint 3: within signal duration (if available)
    if 'original_duration_s' in df.columns:
        max_time = df['original_duration_s'] - min_gap
        df['t_p_detected_seconds'] = df['t_p_detected_seconds'].clip(upper=max_time)
        df['t_s_detected_seconds'] = df['t_s_detected_seconds'].clip(upper=max_time)
    
    return df


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

def compute_sensitivity_metrics(
    zeta_original: np.ndarray,
    zeta_perturbed: np.ndarray,
    q_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute metrics comparing original vs perturbed scaling exponents.
    
    Parameters
    ----------
    zeta_original : np.ndarray
        Original scaling exponents ζ(q)
    zeta_perturbed : np.ndarray
        Perturbed scaling exponents ζ(q)
    q_values : np.ndarray
        Moment orders
        
    Returns
    -------
    metrics : dict
        Dictionary with RMSE, MAE, correlation, max_deviation
    """
    
    # Remove NaN values
    mask = ~(np.isnan(zeta_original) | np.isnan(zeta_perturbed))
    zeta_orig_clean = zeta_original[mask]
    zeta_pert_clean = zeta_perturbed[mask]
    
    if len(zeta_orig_clean) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'correlation': np.nan,
            'max_deviation': np.nan,
            'n_valid': 0
        }
    
    # Compute metrics
    diff = zeta_orig_clean - zeta_pert_clean
    
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    
    # Correlation (handle case where one is constant)
    if np.std(zeta_orig_clean) > 0 and np.std(zeta_pert_clean) > 0:
        correlation = np.corrcoef(zeta_orig_clean, zeta_pert_clean)[0, 1]
    else:
        correlation = np.nan
    
    max_deviation = np.max(np.abs(diff))
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'max_deviation': float(max_deviation),
        'n_valid': int(len(zeta_orig_clean))
    }


def compute_monte_carlo_statistics(
    zeta_runs: List[np.ndarray],
    q_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute statistics from Monte Carlo runs.
    
    Parameters
    ----------
    zeta_runs : list of np.ndarray
        List of ζ(q) arrays from multiple runs
    q_values : np.ndarray
        Moment orders
        
    Returns
    -------
    stats : dict
        Dictionary with mean, std, percentiles of ζ(q)
    """
    
    # Stack all runs
    zeta_stack = np.stack(zeta_runs, axis=0)  # Shape: (n_runs, n_q)
    
    # Compute statistics along runs axis
    return {
        'mean': np.nanmean(zeta_stack, axis=0),
        'std': np.nanstd(zeta_stack, axis=0),
        'p05': np.nanpercentile(zeta_stack, 5, axis=0),
        'p95': np.nanpercentile(zeta_stack, 95, axis=0),
        'median': np.nanmedian(zeta_stack, axis=0)
    }


# ==============================================================================
# RESULT AGGREGATION
# ==============================================================================

def create_sensitivity_summary(
    results: Dict,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create summary DataFrame from sensitivity results.
    
    Parameters
    ----------
    results : dict
        Nested dictionary with sensitivity results
    save_path : Path, optional
        If provided, save CSV to this path
        
    Returns
    -------
    df_summary : pd.DataFrame
        Summary table with columns:
        [data_type, coda_method, scenario, window, rmse, mae, correlation, max_deviation]
    """
    
    rows = []
    
    for data_type, data_results in results.items():
        for coda_method, method_results in data_results.items():
            for scenario, scenario_results in method_results.items():
                for window, metrics in scenario_results.items():
                    if isinstance(metrics, dict) and 'rmse' in metrics:
                        rows.append({
                            'data_type': data_type,
                            'coda_method': coda_method,
                            'scenario': scenario,
                            'window': window,
                            **metrics
                        })
    
    df_summary = pd.DataFrame(rows)
    
    if save_path is not None:
        df_summary.to_csv(save_path, index=False)
    
    return df_summary


def save_intermediate_results(
    results: Dict,
    save_path: Path,
    overwrite: bool = True
) -> None:
    """
    Save intermediate results during long runs.
    
    Parameters
    ----------
    results : dict
        Results dictionary to save
    save_path : Path
        Path to save pickle file
    overwrite : bool
        If True, overwrite existing file
    """
    
    if not overwrite and save_path.exists():
        return
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)