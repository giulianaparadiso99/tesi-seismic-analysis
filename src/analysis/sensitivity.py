"""
Utility functions for sensitivity analysis of phase picking errors on moment scaling.

This module provides:
- Perturbation functions for different error scenarios
- Metrics computation for comparing original vs perturbed results
- Result aggregation and summary statistics
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Tuple, List, Optional, Callable
from pathlib import Path
from collections import defaultdict


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
# WRAPPER FUNCTION
# ==============================================================================

def run_sensitivity_analysis(
    data_type: str,
    signals_dict: Dict,
    df_full: pd.DataFrame,
    baseline_results: Dict,
    coda_methods: list,
    perturbation_scenarios: Dict,
    segment_function: Callable,
    analyze_function: Callable,
    config: Dict,
    output_dir: Path,
    verbose: bool = False
) -> Dict:
    """
    Run complete sensitivity analysis for ONE data type.
    
    Loops over:
    - coda_methods (rautian, arias, envelope, median)
    - perturbation scenarios (noise_small, noise_medium, etc.)
    - Monte Carlo runs
    
    Parameters
    ----------
    data_type : str
        'acceleration', 'velocity', or 'displacement'
    signals_dict : dict
        Signal dictionary for this data type
    df_full : pd.DataFrame
        Full dataframe with picks
    baseline_results : dict
        Baseline moment scaling results {coda_method: df_summary}
    coda_methods : list
        List of coda methods to analyze
    perturbation_scenarios : dict
        {scenario_name: {'type': 'gaussian'|'bias', 'std'|'bias': float}}
    segment_function : callable
        Function to segment signals (segment_all_signals)
    analyze_function : callable
        Function to analyze windows (analyze_all_windows)
    config : dict
        Configuration with TAU_MIN, Q_VALUES, SAMPLING_RATE, etc.
    output_dir : Path
        Directory to save intermediate results
    verbose : bool
        If True, print detailed progress
        
    Returns
    -------
    results : dict
        {coda_method: {scenario: {window: metrics}}}
    """
    
    
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results structure
    results = {method: defaultdict(dict) for method in coda_methods}
    
    # Progress tracking
    total_combinations = len(coda_methods) * (len(perturbation_scenarios) + 1)
    current = 0
    
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS: {data_type.upper()}")
    print(f"{'='*80}")
    print(f"Coda methods: {len(coda_methods)}")
    print(f"Scenarios: {len(perturbation_scenarios)} + Monte Carlo ({config['N_MONTE_CARLO']} runs)")
    print(f"Total combinations: {total_combinations}")
    print(f"{'='*80}\n")
    
    # Loop over coda methods
    for coda_method in coda_methods:
        
        print(f"\n[{data_type.upper()}] Processing: {coda_method}")
        print(f"{'-'*80}")
        
        # Extract baseline zeta
        baseline_zeta = {}
        for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
            try:
                baseline_zeta[window] = baseline_results[coda_method][
                    baseline_results[coda_method]['window'] == window
                ]['zeta'].values[0]
            except:
                baseline_zeta[window] = None
        
        # Perturbation scenarios
        scenario_summary = []
        
        for scenario_name, scenario_params in perturbation_scenarios.items():
            
            current += 1
            
            if verbose:
                print(f"  [{current}/{total_combinations}] {scenario_name}...", end=' ')
            
            # Perturb picks
            if scenario_params['type'] == 'gaussian':
                df_perturbed = perturb_picks_gaussian(
                    df_full,
                    noise_std=scenario_params['std'],
                    sampling_rate=config['SAMPLING_RATE'],
                    random_state=42
                )
            elif scenario_params['type'] == 'bias':
                df_perturbed = perturb_picks_bias(
                    df_full,
                    bias_seconds=scenario_params['bias'],
                    sampling_rate=config['SAMPLING_RATE']
                )
            else:
                raise ValueError(f"Unknown perturbation type: {scenario_params['type']}")
            
            # Regenerate windows
            try:
                windowed_perturbed = segment_function(
                    signals_dict,
                    df_perturbed,
                    coda_method=coda_method
                )
            except Exception as e:
                if verbose:
                    print(f"FAILED (windowing)")
                logger.warning(f"{data_type}/{coda_method}/{scenario_name}: windowing failed - {e}")
                continue
            
            # Compute moment scaling
            try:
                results_perturbed = analyze_function(
                    windowed_perturbed,
                    signal_field=config['SIGNAL_COLUMN'],
                    tau_min=config['TAU_MIN'],
                    n_tau=None,
                    q_values=config['Q_VALUES'],
                    sampling_rate=config['SAMPLING_RATE'],
                    fit_range=None
                )
            except Exception as e:
                if verbose:
                    print(f"FAILED (analysis)")
                logger.warning(f"{data_type}/{coda_method}/{scenario_name}: analysis failed - {e}")
                continue
            
            # Compute metrics for each window
            window_metrics = {}
            for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
                
                if baseline_zeta[window] is None:
                    continue
                
                if window not in results_perturbed or results_perturbed[window] is None:
                    continue
                
                zeta_perturbed = results_perturbed[window]['scaling']['zeta']
                if isinstance(baseline_zeta, pd.DataFrame):
                    # Se baseline_zeta è un DataFrame
                    zeta_baseline = baseline_zeta[baseline_zeta['window'] == window]['zeta'].values
                else:
                    # Se è già un dict corretto
                    zeta_baseline = baseline_zeta[window]
                
                metrics = compute_sensitivity_metrics(
                    zeta_baseline,
                    zeta_perturbed,
                    config['Q_VALUES']
                )
                
                window_metrics[window] = metrics
                
                scenario_summary.append({
                    'coda_method': coda_method,
                    'scenario': scenario_name,
                    'window': window,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'correlation': metrics['correlation'],
                    'max_deviation': metrics['max_deviation']
                })
            
            results[coda_method][scenario_name] = window_metrics
            
            if verbose:
                print(f"OK ({len(window_metrics)} windows)")
            
            # Save intermediate
            intermediate_file = output_dir / f'{data_type}_{coda_method}_{scenario_name}.pkl'
            save_intermediate_results(
                {scenario_name: window_metrics},
                intermediate_file
            )
        
        # Monte Carlo analysis
        current += 1
        print(f"  [{current}/{total_combinations}] Monte Carlo ({config['N_MONTE_CARLO']} runs)...", end=' ')
        
        mc_zeta = {window: [] for window in ['pre_event', 'p_wave', 's_wave', 'coda']}
        n_successful = 0
        
        for mc_run in range(config['N_MONTE_CARLO']):
            
            # Perturb
            df_mc = perturb_picks_gaussian(
                df_full,
                noise_std=config['MONTE_CARLO_STD'],
                sampling_rate=config['SAMPLING_RATE'],
                random_state=42 + mc_run
            )
            
            # Regenerate windows
            try:
                windowed_mc = segment_function(
                    signals_dict,
                    df_mc,
                    coda_method=coda_method
                )
            except Exception as e:
                continue
            
            # Analyze
            try:
                results_mc = analyze_function(
                    windowed_mc,
                    signal_field=config['SIGNAL_COLUMN'],
                    tau_min=config['TAU_MIN'],
                    n_tau=None,
                    q_values=config['Q_VALUES'],
                    sampling_rate=config['SAMPLING_RATE'],
                    fit_range=None
                )
            except Exception as e:
                continue
            
            # Store zeta
            for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
                if window in results_mc and results_mc[window] is not None:
                    mc_zeta[window].append(results_mc[window]['scaling']['zeta'])
            
            n_successful += 1
        
        # Compute MC statistics
        mc_results = {}
        for window in ['pre_event', 'p_wave', 's_wave', 'coda']:
            
            if len(mc_zeta[window]) == 0:
                continue
            
            mc_stats = compute_monte_carlo_statistics(mc_zeta[window], config['Q_VALUES'])
            
            if baseline_zeta[window] is not None:
                metrics = compute_sensitivity_metrics(
                    baseline_zeta[window],
                    mc_stats['mean'],
                    config['Q_VALUES']
                )
                
                mc_results[window] = {
                    'statistics': mc_stats,
                    'metrics': metrics,
                    'n_successful_runs': len(mc_zeta[window])
                }
                
                scenario_summary.append({
                    'coda_method': coda_method,
                    'scenario': 'monte_carlo',
                    'window': window,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'correlation': metrics['correlation'],
                    'max_deviation': metrics['max_deviation']
                })
        
        results[coda_method]['monte_carlo'] = mc_results
        
        print(f"OK ({n_successful}/{config['N_MONTE_CARLO']} successful)")
        
        # Save intermediate for this method
        method_file = output_dir / f'{data_type}_{coda_method}_complete.pkl'
        save_intermediate_results(results[coda_method], method_file)
        
        # Print compact summary for this method
        if len(scenario_summary) > 0:
            df_method = pd.DataFrame(scenario_summary)
            df_method_pivot = df_method.pivot_table(
                index=['scenario', 'window'],
                values='rmse',
                aggfunc='mean'
            )
            print(f"\n  Summary for {coda_method}:")
            print(df_method_pivot.to_string())
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {data_type.upper()}")
    print(f"{'='*80}\n")
    
    return dict(results)


def create_summary(
    results: Dict,
    data_type: str,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create compact summary table for one data type.
    
    Parameters
    ----------
    results : dict
        Results from run_sensitivity_for_datatype
    data_type : str
        Data type name
    save_path : Path, optional
        Where to save CSV
        
    Returns
    -------
    df_summary : pd.DataFrame
        Compact summary with one row per (coda_method, scenario, window)
    """
    
    rows = []
    
    for coda_method, method_results in results.items():
        for scenario, scenario_results in method_results.items():
            
            # Handle regular scenarios
            if isinstance(scenario_results, dict) and 'pre_event' in scenario_results:
                for window, metrics in scenario_results.items():
                    if isinstance(metrics, dict) and 'rmse' in metrics:
                        rows.append({
                            'data_type': data_type,
                            'coda_method': coda_method,
                            'scenario': scenario,
                            'window': window,
                            **metrics
                        })
            
            # Handle Monte Carlo (nested structure)
            elif isinstance(scenario_results, dict):
                for window, mc_data in scenario_results.items():
                    if isinstance(mc_data, dict) and 'metrics' in mc_data:
                        rows.append({
                            'data_type': data_type,
                            'coda_method': coda_method,
                            'scenario': scenario,
                            'window': window,
                            **mc_data['metrics'],
                            'n_mc_runs': mc_data.get('n_successful_runs', 0)
                        })
    
    df_summary = pd.DataFrame(rows)
    
    if save_path is not None:
        df_summary.to_csv(save_path, index=False)
    
    return df_summary


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