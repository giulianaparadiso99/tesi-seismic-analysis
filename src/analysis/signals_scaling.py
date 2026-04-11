import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from src.visualization.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ==================================== Increment Computation ====================================
# ===============================================================================================

def compute_increments(df, tau_values, column='displacement'):
    """
    Compute increments Δx(τ, t₀) = x(t₀ + τ) - x(t₀) for all files and time lags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with columns ['file', column]
    tau_values : list of int
        Time lags (in samples)
    column : str
        Column name of the process ('acceleration', 'velocity', 'displacement')
    
    Returns
    -------
    pd.DataFrame
        Columns: [file, station, stream, tau, t0, increment]
    
    Examples
    --------
    >>> df = integrate_to_displacement(df_acc_event)
    >>> df_inc = compute_increments(df, tau_values, column='displacement')
    >>> print(df_inc.head())
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    
    inc_rows = []
    
    for file in df['file'].unique():
        signal = df[df['file'] == file][column].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        N = len(signal)
        
        for tau in tau_values:
            if (N - tau) < 10:
                continue
            
            # Increments: x(t0 + tau) - x(t0)
            increments = signal[tau:] - signal[:-tau]
            
            # Save with t0 index
            for t0, inc in enumerate(increments):
                inc_rows.append({
                    'file': file,
                    'station': station,
                    'stream': stream,
                    'tau': tau,
                    't0': t0,
                    'increment': inc
                })
    
    return pd.DataFrame(inc_rows)


# ===============================================================================================
# ==================================== Moment Computation =======================================
# ===============================================================================================

def compute_moments_from_increments(df_increments, q_values):
    """
    Compute q-th order moments from pre-computed increments.
    
    M_q(τ) = ⟨|Δx(τ, t₀)|^q⟩_{t₀}
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        Increments from compute_increments()
        Must have columns: ['file', 'station', 'stream', 'tau', 'increment']
    q_values : list of float
        Moment orders to compute
    
    Returns
    -------
    pd.DataFrame
        Columns: [file, station, stream, q, tau, moment]
    
    Examples
    --------
    >>> df_inc = compute_increments(df, tau_values, column='displacement')
    >>> df_mom = compute_moments_from_increments(df_inc, q_values)
    """
    rows = []
    
    # Group by file and tau
    for (file, station, stream, tau), group in df_increments.groupby(['file', 'station', 'stream', 'tau']):
        increments = group['increment'].values
        
        for q in q_values:
            moment = np.mean(np.abs(increments) ** q)
            rows.append({
                'file': file,
                'station': station,
                'stream': stream,
                'q': q,
                'tau': tau,
                'moment': moment
            })
    
    return pd.DataFrame(rows)


# ===============================================================================================
# ==================================== Unified Wrapper ==========================================
# ===============================================================================================

def compute_moment_scaling(df_acc, q_values, tau_values, 
                          process='displacement',
                          normalized=False,
                          dt=0.005, 
                          save_increments=False):
    """
    Compute q-th order moments of increments for a given process.
    
    Wrapper function that:
    1. Ensures the process column exists (computes if needed)
    2. Computes increments
    3. Computes moments from increments
    
    For more control, use compute_increments() and compute_moments_from_increments()
    separately.
    
    Parameters
    ----------
    df_acc : pd.DataFrame
        Acceleration data (may already contain velocity/displacement)
    q_values : list of float
        Moment orders
    tau_values : list of int
        Time lags (in samples)
    process : str
        'acceleration', 'velocity', or 'displacement'
    normalized : bool
        Use normalized acceleration (only for acceleration process)
    dt : float
        Sampling interval (only used if velocity/displacement need to be computed)
    save_increments : bool
        If True, also return increments dataframe
    
    Returns
    -------
    df_moments : pd.DataFrame
        Columns: [file, station, stream, q, tau, moment]
    df_increments : pd.DataFrame (optional)
        Columns: [file, station, stream, tau, t0, increment]
        Only returned if save_increments=True
    
    Examples
    --------
    >>> # Simple usage (auto-compute everything)
    >>> df_moments = compute_moment_scaling(df_acc, q_values, tau_values,
    ...                                     process='displacement')
    
    >>> # With increments
    >>> df_moments, df_increments = compute_moment_scaling(
    ...     df_acc, q_values, tau_values, 
    ...     process='displacement',
    ...     save_increments=True
    ... )
    
    >>> # Manual control (more efficient for multiple analyses)
    >>> df = integrate_to_displacement(df_acc)
    >>> df_inc = compute_increments(df, tau_values, column='displacement')
    >>> df_mom = compute_moments_from_increments(df_inc, q_values)
    """
    valid_processes = ['acceleration', 'velocity', 'displacement']
    if process not in valid_processes:
        raise ValueError(f"process must be one of {valid_processes}, got '{process}'")
    
    # Ensure process column exists
    if process == 'acceleration':
        col = 'acceleration_normalized' if normalized else 'acceleration'
        df = df_acc.copy()
        
    elif process == 'velocity':
        if 'velocity' in df_acc.columns:
            df = df_acc.copy()
        else:
            df = integrate_to_velocity(df_acc, dt=dt, normalized=normalized)
        col = 'velocity'
    
    elif process == 'displacement':
        if 'displacement' in df_acc.columns:
            df = df_acc.copy()
        else:
            df = integrate_to_displacement(df_acc, dt=dt, normalized=normalized)
        col = 'displacement'
    
    # Compute increments
    df_increments = compute_increments(df, tau_values, column=col)
    
    # Compute moments
    df_moments = compute_moments_from_increments(df_increments, q_values)
    
    if save_increments:
        return df_moments, df_increments
    return df_moments

# ===============================================================================================
# ==================================== Moment Validation ========================================
# ===============================================================================================

def validate_moments(df_moments, process_name='displacement'):
    """
    Quick validation and sanity checks for moment data.
    
    Checks:
    - Data quality (NaN, Inf, negative values)
    - Range analysis
    - Basic statistics
    
    This is a lightweight diagnostic function. For full analysis use:
    - compute_scaling_exponents() for zeta(q) estimation
    - test_scaling_linearity() for linearity check
    - fit_piecewise_scaling() for piecewise fits
    
    Parameters
    ----------
    df_moments : pd.DataFrame
        Moments from compute_moments_from_increments()
        Columns: ['file', 'station', 'stream', 'q', 'tau', 'moment']
    process_name : str
        Name for labeling
    
    Returns
    -------
    dict
        Summary statistics and quality flags
    """
    print(f"\n{'='*80}")
    print(f"MOMENT VALIDATION - {process_name.upper()}")
    print('='*80)
    
    # Dataset info
    print(f"\nDataset Info:")
    print(f"  Total moment values: {len(df_moments):,}")
    print(f"  Number of files: {df_moments['file'].nunique()}")
    print(f"  q range: [{df_moments['q'].min()}, {df_moments['q'].max()}]")
    print(f"  tau range: [{df_moments['tau'].min()}, {df_moments['tau'].max()}]")
    
    # Quality checks
    print(f"\nData Quality Checks:")
    
    n_nan = df_moments['moment'].isna().sum()
    n_inf = np.isinf(df_moments['moment']).sum()
    n_negative = (df_moments['moment'] < 0).sum()
    n_zero = (df_moments['moment'] == 0).sum()
    
    quality_ok = True
    
    if n_nan > 0:
        print(f"  ERROR: {n_nan} NaN values found")
        quality_ok = False
    else:
        print(f"  OK: No NaN values")
    
    if n_inf > 0:
        print(f"  ERROR: {n_inf} Inf values found")
        quality_ok = False
    else:
        print(f"  OK: No Inf values")
    
    if n_negative > 0:
        print(f"  ERROR: {n_negative} negative moments (impossible!)")
        quality_ok = False
    else:
        print(f"  OK: All moments positive")
    
    if n_zero > 0:
        print(f"  WARNING: {n_zero} zero moments")
    
    # Range analysis by q
    print(f"\nRange Analysis by q:")
    print(f"  {'q':>5} | {'Min':>12} | {'Max':>12} | {'Median':>12} | {'CV':>8}")
    print(f"  {'-'*5}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*10}")
    
    q_stats = {}
    for q in sorted(df_moments['q'].unique()):
        moments_q = df_moments[df_moments['q'] == q]['moment']
        
        q_stats[q] = {
            'min': moments_q.min(),
            'max': moments_q.max(),
            'median': moments_q.median(),
            'mean': moments_q.mean(),
            'std': moments_q.std(),
            'cv': moments_q.std() / moments_q.mean() if moments_q.mean() > 0 else np.nan
        }
        
        print(f"  {q:5.2f} | {q_stats[q]['min']:12.6e} | {q_stats[q]['max']:12.6e} | "
              f"{q_stats[q]['median']:12.6e} | {q_stats[q]['cv']:8.3f}")
    
    # Variability across files
    print(f"\nVariability Across Files:")
    
    # For q=2 (example), compute coefficient of variation across files for each tau
    df_q2 = df_moments[df_moments['q'] == 2.0]
    cv_per_tau = []
    for tau in df_q2['tau'].unique():
        moments_tau = df_q2[df_q2['tau'] == tau]['moment']
        if len(moments_tau) > 1:
            cv = moments_tau.std() / moments_tau.mean()
            cv_per_tau.append(cv)
    
    if cv_per_tau:
        avg_cv = np.mean(cv_per_tau)
        print(f"  Average CV across files (q=2): {avg_cv:.3f}")
        if avg_cv < 0.5:
            print(f"  Low variability between stations")
        elif avg_cv < 1.0:
            print(f"  Moderate variability between stations")
        else:
            print(f"  High variability between stations")
    
    # Summary
    print(f"\n{'='*80}")
    if quality_ok:
        print("VALIDATION PASSED: Data quality is good")
    else:
        print("VALIDATION FAILED: Issues detected, review above")
    print('='*80)
    
    return {
        'quality_ok': quality_ok,
        'n_nan': n_nan,
        'n_inf': n_inf,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'q_stats': q_stats
    }

# ===============================================================================================
# ==================================== Processes analysis =======================================
# ===============================================================================================

def analyze_processes(df, processes=['velocity', 'displacement'], 
                     output_dir=None, save_plots=True):
    """
    Exploratory analysis of integrated processes (velocity and displacement).
    
    Analyzes:
    - Range and distribution statistics
    - Zero-crossing analysis
    - Temporal evolution
    - Cross-file comparison
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: ['file', 'sample', 'acceleration', 'velocity', 'displacement']
    processes : list
        Which processes to analyze
    output_dir : Path
        Where to save plots
    save_plots : bool
        Whether to save plots
    
    Returns
    -------
    dict
        Summary statistics for each process
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    results = {}
    
    for process in processes:
        if process not in df.columns:
            print(f"{process} not found in dataframe")
            continue
        
        print(f"\n{'='*80}")
        print(f"ANALYZING {process.upper()}")
        print('='*80)
        
        # Overall statistics
        stats = {
            'mean': df[process].mean(),
            'std': df[process].std(),
            'min': df[process].min(),
            'max': df[process].max(),
            'median': df[process].median(),
            'q25': df[process].quantile(0.25),
            'q75': df[process].quantile(0.75),
            'n_zeros': (df[process] == 0).sum(),
            'n_positive': (df[process] > 0).sum(),
            'n_negative': (df[process] < 0).sum()
        }
        
        print(f"\nOverall Statistics:")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  Mean: {stats['mean']:.6f} (should be ≈0 after integration)")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Median: {stats['median']:.6f}")
        print(f"  Q25-Q75: [{stats['q25']:.6f}, {stats['q75']:.6f}]")
        print(f"  Positive/Negative: {stats['n_positive']}/{stats['n_negative']}")
        
        # Per-file statistics
        per_file = df.groupby('file')[process].agg(['mean', 'std', 'min', 'max'])
        
        print(f"\nPer-File Statistics:")
        print(f"  Mean range across files: [{per_file['mean'].min():.6f}, {per_file['mean'].max():.6f}]")
        print(f"  Std range across files: [{per_file['std'].min():.6f}, {per_file['std'].max():.6f}]")
        print(f"  Global min/max: [{per_file['min'].min():.6f}, {per_file['max'].max():.6f}]")
        
        # Sanity checks
        print(f"\nSanity Checks:")
        if abs(stats['mean']) > 1e-3:
            print(f"Mean far from zero: {stats['mean']:.6f} (possible drift!)")
        else:
            print(f"Mean close to zero: {stats['mean']:.6e}")
        
        if stats['n_positive'] == 0 or stats['n_negative'] == 0:
            print(f" Only {'positive' if stats['n_positive'] > 0 else 'negative'} values (suspicious!)")
        else:
            print(f"Both positive and negative values present")
        
        # Check for NaN/Inf
        n_nan = df[process].isna().sum()
        n_inf = np.isinf(df[process]).sum()
        if n_nan > 0:
            print(f"{n_nan} NaN values found!")
        if n_inf > 0:
            print(f"{n_inf} Inf values found!")
        if n_nan == 0 and n_inf == 0:
            print(f"No NaN or Inf values")
        
        results[process] = stats
        
        # Plots
        if save_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Overall distribution
            axes[0, 0].hist(df[process], bins=100, edgecolor='none', alpha=0.7)
            axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
            axes[0, 0].set_xlabel(f'{process.capitalize()} ({"cm/s" if process=="velocity" else "cm"})')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title(f'{process.capitalize()} Distribution - All Files')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # Plot 2: Per-file statistics
            per_file_sorted = per_file.sort_values('std', ascending=False)
            x = np.arange(len(per_file_sorted))
            axes[0, 1].bar(x, per_file_sorted['std'], alpha=0.7)
            axes[0, 1].set_xlabel('File (sorted by std)')
            axes[0, 1].set_ylabel(f'Std ({"cm/s" if process=="velocity" else "cm"})')
            axes[0, 1].set_title(f'{process.capitalize()} - Standard Deviation per File')
            axes[0, 1].grid(alpha=0.3)
            
            # Plot 3: Time series examples
            example_files = df['file'].unique()[:4]
            for file in example_files:
                df_file = df[df['file'] == file]
                t = df_file['sample'] * 0.005  # Convert to seconds
                axes[1, 0].plot(t, df_file[process], alpha=0.7, label=file.split('.')[1])
            axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.5)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel(f'{process.capitalize()} ({"cm/s" if process=="velocity" else "cm"})')
            axes[1, 0].set_title(f'{process.capitalize()} - Example Time Series')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            
            # Plot 4: Q-Q plot (check normality)
            from scipy import stats as sp_stats
            sample = df[process].dropna().sample(min(10000, len(df)), random_state=42)
            sp_stats.probplot(sample, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(f'{process.capitalize()} - Q-Q Plot (Normality Check)')
            axes[1, 1].grid(alpha=0.3)
            
            plt.suptitle(f'Process Analysis: {process.capitalize()}', fontsize=14, y=1.00)
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f'process_analysis_{process}.pdf', bbox_inches='tight')
                print(f"\n  Saved plot to {output_dir / f'process_analysis_{process}.pdf'}")
            
            plt.show()
    
    return results

# ===============================================================================================
# ==================================== Increments distribution analysis =========================
# ===============================================================================================

def analyze_increments(df_increments, process_name='displacement',
                      tau_examples=[100, 1000, 5000],
                      output_dir=None, save_plots=True):
    """
    Comprehensive exploratory analysis of increments.
    
    Analyzes:
    - Distribution statistics (mean, std, median, skew, kurtosis)
    - Range analysis (min, max)
    - Fraction |Δx| < 1 vs ≥ 1
    - Symmetry check
    - Heavy tails assessment
    - Scaling with τ
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        Increments with columns ['file', 'station', 'stream', 'tau', 't0', 'increment']
    process_name : str
        Name for labeling
    tau_examples : list
        Specific τ values for detailed plots
    output_dir : Path
        Where to save plots
    save_plots : bool
        Whether to generate plots
    
    Returns
    -------
    df_summary : pd.DataFrame
        Comprehensive summary with all statistics per τ
    """
    from scipy import stats as sp_stats
    
    print(f"\n{'='*80}")
    print(f"INCREMENTS ANALYSIS - {process_name.upper()}")
    print('='*80)
    
    tau_values = sorted(df_increments['tau'].unique())
    
    print(f"\nDataset Info:")
    print(f"  Total increments: {len(df_increments):,}")
    print(f"  Number of files: {df_increments['file'].nunique()}")
    print(f"  Number of τ values: {len(tau_values)}")
    print(f"  τ range: [{min(tau_values)}, {max(tau_values)}]")
    
    # Compute comprehensive statistics per τ
    results = []
    
    for tau in tau_values:
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        abs_increments = np.abs(increments)
        
        n_total = len(increments)
        n_less_1 = np.sum(abs_increments < 1)
        n_geq_1 = np.sum(abs_increments >= 1)
        
        stats = {
            # Basic info
            'tau': tau,
            'n_samples': n_total,
            
            # Signed increment statistics (for symmetry check)
            'mean': np.mean(increments),
            'std': np.std(increments),
            'median': np.median(increments),
            
            # Absolute increment statistics
            'mean_abs': np.mean(abs_increments),
            'median_abs': np.median(abs_increments),
            'std_abs': np.std(abs_increments),
            'min_abs': np.min(abs_increments),
            'max_abs': np.max(abs_increments),
            
            # Distribution shape
            'skewness': sp_stats.skew(increments),
            'kurtosis': sp_stats.kurtosis(increments),  # Excess kurtosis
            
            # Fraction analysis
            'n_less_than_1': n_less_1,
            'n_geq_1': n_geq_1,
            'frac_less_than_1': n_less_1 / n_total,
            'frac_geq_1': n_geq_1 / n_total
        }
        
        results.append(stats)
    
    df_summary = pd.DataFrame(results)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print('='*80)
    print(f"\n{'τ':>6} | {'Mean':>10} | {'Std':>10} | {'Skew':>7} | {'Kurt':>7} | "
          f"{'<1':>7} | {'≥1':>7} | {'Mean|Δx|':>10}")
    print(f"{'-'*6}|{'-'*12}|{'-'*12}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*12}")
    
    for _, row in df_summary.iterrows():
        print(f"{row['tau']:6.0f} | {row['mean']:10.4e} | {row['std']:10.4e} | "
              f"{row['skewness']:7.2f} | {row['kurtosis']:7.2f} | "
              f"{row['frac_less_than_1']:6.1%} | {row['frac_geq_1']:6.1%} | "
              f"{row['mean_abs']:10.4e}")
    
    # Sanity checks
    print(f"\n{'='*80}")
    print(f"SANITY CHECKS")
    print('='*80)
    
    # 1. Symmetry check
    max_mean = df_summary['mean'].abs().max()
    if max_mean < 1e-6:
        print(f"Symmetry: All means ≈ 0 (max |mean| = {max_mean:.2e})")
    else:
        print(f"Asymmetry detected: max |mean| = {max_mean:.2e}")
    
    # 2. Monotonic std increase
    stds = df_summary['std'].values
    if all(stds[i] <= stds[i+1] for i in range(len(stds)-1)):
        print(f"Scaling: Std increases monotonically with τ")
    else:
        print(f"Non-monotonic std scaling detected")
    
    # 3. Heavy tails
    heavy_tail_count = (df_summary['kurtosis'] > 3).sum()
    if heavy_tail_count > 0:
        print(f"Heavy tails: {heavy_tail_count}/{len(tau_values)} τ values have kurtosis > 3")
    else:
        print(f" No heavy tails detected (all kurtosis ≤ 3)")
    
    # 4. Fraction < 1
    avg_frac_less_1 = df_summary['frac_less_than_1'].mean()
    print(f"Average fraction |Δx| < 1: {avg_frac_less_1:.1%}")
    
    # Plots (if requested)
    if save_plots:
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        # Plot 1-3: Distributions for specific τ
        for idx, tau in enumerate(tau_examples[:3]):
            if tau not in tau_values:
                continue
            
            inc_tau = df_increments[df_increments['tau'] == tau]['increment']
            
            axes[idx].hist(inc_tau, bins=100, density=True, alpha=0.7, edgecolor='none')
            axes[idx].axvline(0, color='red', linestyle='--', linewidth=1)
            axes[idx].set_xlabel(f'Increment ({"cm" if "disp" in process_name else "cm/s"})')
            axes[idx].set_ylabel('Density (log scale)')
            axes[idx].set_yscale('log')
            
            row = df_summary[df_summary['tau'] == tau].iloc[0]
            axes[idx].set_title(f'τ={tau} ({tau*0.005:.2f}s)\n'
                               f'Skew={row["skewness"]:.2f}, Kurt={row["kurtosis"]:.2f}\n'
                               f'|Δx|<1: {row["frac_less_than_1"]:.1%}')
            axes[idx].grid(alpha=0.3)
        
        # Plot 4: Std scaling
        tau_array = df_summary['tau'].values
        std_array = df_summary['std'].values
        
        axes[3].loglog(tau_array, std_array, 'o-', linewidth=2, markersize=6)
        axes[3].set_xlabel('τ (samples)')
        axes[3].set_ylabel('Std(Δx(τ))')
        axes[3].set_title('Std Scaling with τ')
        
        # Reference: normal diffusion
        tau_ref = tau_array[len(tau_array)//2]
        std_ref = std_array[len(std_array)//2]
        axes[3].loglog(tau_array, std_ref * (tau_array/tau_ref)**0.5, 
                      '--', alpha=0.7, label='τ^0.5 (normal diffusion)')
        axes[3].legend()
        axes[3].grid(alpha=0.3, which='both')
        
        # Plot 5: Fraction < 1 vs τ
        axes[4].semilogx(tau_array, df_summary['frac_less_than_1']*100, 'o-')
        axes[4].set_xlabel('τ (samples)')
        axes[4].set_ylabel('% with |Δx| < 1')
        axes[4].set_title('Fraction of Small Increments vs τ')
        axes[4].grid(alpha=0.3)
        
        # Plot 6: Kurtosis vs τ
        axes[5].semilogx(tau_array, df_summary['kurtosis'], 'o-', color='red')
        axes[5].axhline(0, color='gray', linestyle='--', linewidth=0.5, label='Normal')
        axes[5].axhline(3, color='orange', linestyle='--', linewidth=0.5, label='Heavy tail threshold')
        axes[5].set_xlabel('τ (samples)')
        axes[5].set_ylabel('Excess Kurtosis')
        axes[5].set_title('Heavy Tail Assessment')
        axes[5].legend()
        axes[5].grid(alpha=0.3)
        
        plt.suptitle(f'Increment Analysis: {process_name.capitalize()}', fontsize=14)
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f'increment_analysis_{process_name}.pdf', 
                       bbox_inches='tight')
            print(f"\n📊 Saved plot to {output_dir / f'increment_analysis_{process_name}.pdf'}")
        
        plt.show()
    
    return df_summary

# ===============================================================================================
# ==================================== Event onset detection ====================================
# ===============================================================================================

def detect_event_onset(signal, threshold_factor=0.05, min_consecutive=10):
    """
    Detects the onset of the seismic event as the first sample where
    |signal| exceeds threshold_factor * max(|signal|) for at least
    min_consecutive consecutive samples.

    Parameters
    ----------
    signal : np.ndarray
    threshold_factor : float — fraction of max(|signal|) used as threshold
    min_consecutive : int — minimum number of consecutive samples above threshold

    Returns
    -------
    onset : int — index of the first sample of the event window
    """
    threshold = threshold_factor * np.max(np.abs(signal))
    above = np.abs(signal) > threshold
    for i in range(len(signal) - min_consecutive):
        if np.all(above[i:i + min_consecutive]):
            return i
    return 0  # fallback: use full signal

# ===============================================================================================
# ==================================== Event window trimming ====================================
# ===============================================================================================


def trim_to_event_window(df, threshold_factor=0.05, min_consecutive=10,
                         normalized=False):
    """
    Trims each signal in df to the event window, starting from the
    detected onset of the seismic event.

    Parameters
    ----------
    df : pd.DataFrame — preprocessed acceleration data
    threshold_factor : float — passed to detect_event_onset
    min_consecutive : int — passed to detect_event_onset
    normalized : bool — whether to use acceleration_normalized or acceleration

    Returns
    -------
    df_trimmed : pd.DataFrame — same structure as df, with signals trimmed
    df_onsets : pd.DataFrame — onset indices per file [file, station, stream, onset]
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    trimmed = []
    onset_rows = []

    for file in df['file'].unique():
        df_file = df[df['file'] == file].copy()
        signal = df_file[col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]

        onset = detect_event_onset(signal, threshold_factor, min_consecutive)

        df_trimmed = df_file.iloc[onset:].reset_index(drop=True)
        trimmed.append(df_trimmed)
        onset_rows.append({
            'file': file,
            'station': station,
            'stream': stream,
            'onset': onset,
            'samples_before': onset,
            'samples_after': len(signal) - onset
        })

    df_trimmed = pd.concat(trimmed, ignore_index=True)
    df_onsets = pd.DataFrame(onset_rows)
    return df_trimmed, df_onsets

# ===============================================================================================
# ========================== Scaling exponents computation ======================================
# ===============================================================================================

def compute_scaling_exponents(df_moments, output_dir='../figures/03_single_signal/scaling'):
    """
    Estimates scaling exponents zeta(q) for each signal by fitting
    log(moment) vs log(tau) for each q value.
    
    Parameters:
    -----------
    df_moments : pd.DataFrame — output of compute_moment_scaling
    output_dir : str — directory to save figures
    
    Returns:
    --------
    df_exponents : pd.DataFrame with columns [file, station, stream, q, zeta, r2]
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    saved = []
    failed = []

    for file in df_moments['file'].unique():
        station = df_moments[df_moments['file'] == file]['station'].iloc[0]
        stream = df_moments[df_moments['file'] == file]['stream'].iloc[0]
        filepath = f'{output_dir}/scaling_{station}_{stream}.pdf'
        df_file = df_moments[df_moments['file'] == file]

        q_values = sorted(df_file['q'].unique())
        zeta_values = []
        r2_values = []

        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 1. log(moment) vs log(tau) for each q
            for i, q in enumerate(q_values):
                df_q = df_file[df_file['q'] == q]
                log_tau = np.log(df_q['tau'].values)
                log_moment = np.log(df_q['moment'].values)

                # Linear fit
                slope, intercept, r, p, se = stats.linregress(log_tau, log_moment)
                zeta_values.append(slope)
                r2_values.append(r**2)

                c = colors[i % len(colors)]
                axes[0].plot(log_tau, log_moment, 'o', color=c, markersize=4, alpha=0.7)
                axes[0].plot(log_tau, slope * log_tau + intercept, '-',
                            color=c, linewidth=1.2, label=f'q={q} (ζ={slope:.2f})')

            axes[0].set_xlabel('log(τ)')
            axes[0].set_ylabel('log(M_q(τ))')
            axes[0].set_title(f'Moment scaling — {station} {stream}')
            axes[0].legend(fontsize=8)

            # 2. zeta(q) vs q — linearity check
            axes[1].plot(q_values, zeta_values, 'o-', color=colors[0],
                        linewidth=1.2, markersize=5, label='ζ(q)')
            # Reference line: linear scaling (normal diffusion)
            q_arr = np.array(q_values)
            axes[1].plot(q_arr, 0.5 * q_arr, '--', color='gray',
                        linewidth=1, label='Linear (normal diffusion)')
            axes[1].set_xlabel('q')
            axes[1].set_ylabel('ζ(q)')
            axes[1].set_title(f'Scaling exponents — {station} {stream}')
            axes[1].legend()

            plt.suptitle(f'Scaling analysis — {station} {stream}', fontsize=13)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)

        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()

        for q, zeta, r2 in zip(q_values, zeta_values, r2_values):
            rows.append({
                'file': file,
                'station': station,
                'stream': stream,
                'q': q,
                'zeta': round(zeta, 4),
                'r2': round(r2, 4)
            })

    # Summary plot — zeta(q) vs q for all signals
    fig, ax = plt.subplots(figsize=(8, 6))
    for file in df_moments['file'].unique():
        station = df_moments[df_moments['file'] == file]['station'].iloc[0]
        stream = df_moments[df_moments['file'] == file]['stream'].iloc[0]
        df_file = pd.DataFrame(rows)
        df_file = df_file[df_file['file'] == file]
        ax.plot(df_file['q'], df_file['zeta'], '-', linewidth=0.8, alpha=0.5)
    q_arr = np.array(sorted(df_moments['q'].unique()))
    ax.plot(q_arr, 0.5 * q_arr, '--', color='black', linewidth=1.5,
            label='Linear (normal diffusion)')
    ax.set_xlabel('q')
    ax.set_ylabel('ζ(q)')
    ax.set_title('Scaling exponents — all signals')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scaling_summary.pdf', bbox_inches='tight')
    plt.close()

    df_exponents = pd.DataFrame(rows)

    print(f"Saved: {len(saved)}/{df_moments['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")

    return df_exponents

# ===============================================================================================
# ========================== Linearity test for scaling exponents ===============================
# ===============================================================================================

def test_scaling_linearity(df_exponents, output_dir='../figures/03_single_signal/scaling'):
    """
    Tests linearity of zeta(q) vs q by comparing linear and quadratic fits
    using AIC/BIC.
    
    Parameters:
    -----------
    df_exponents : pd.DataFrame — output of compute_scaling_exponents
    output_dir : str — directory to save figures
    
    Returns:
    --------
    df_linearity : pd.DataFrame with columns [file, station, stream, 
                   aic_linear, aic_quadratic, bic_linear, bic_quadratic,
                   best_fit, nonlinear]
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    saved = []
    failed = []

    for file in df_exponents['file'].unique():
        station = df_exponents[df_exponents['file'] == file]['station'].iloc[0]
        stream = df_exponents[df_exponents['file'] == file]['stream'].iloc[0]
        df_file = df_exponents[df_exponents['file'] == file]

        q = df_file['q'].values
        zeta = df_file['zeta'].values
        n = len(q)

        # Linear fit: zeta(q) = a*q
        slope_l, intercept_l, r_l, _, _ = stats.linregress(q, zeta)
        residuals_l = zeta - (slope_l * q + intercept_l)
        loglik_l = -n/2 * np.log(np.sum(residuals_l**2) / n)
        aic_l = 2*2 - 2*loglik_l
        bic_l = 2*np.log(n) - 2*loglik_l

        # Quadratic fit: zeta(q) = a*q + b*q^2
        coeffs_q = np.polyfit(q, zeta, 2)
        zeta_pred_q = np.polyval(coeffs_q, q)
        residuals_q = zeta - zeta_pred_q
        loglik_q = -n/2 * np.log(np.sum(residuals_q**2) / n)
        aic_q = 2*3 - 2*loglik_q
        bic_q = 3*np.log(n) - 2*loglik_q

        best_fit = 'linear' if aic_l < aic_q else 'quadratic'
        nonlinear = best_fit == 'quadratic'

        rows.append({
            'file': file,
            'station': station,
            'stream': stream,
            'aic_linear': round(aic_l, 4),
            'aic_quadratic': round(aic_q, 4),
            'bic_linear': round(bic_l, 4),
            'bic_quadratic': round(bic_q, 4),
            'best_fit': best_fit,
            'nonlinear': nonlinear,
            'slope_linear': round(slope_l, 4),
            'quad_coeff_a': round(coeffs_q[1], 4),
            'quad_coeff_b': round(coeffs_q[0], 4),
        })

        # Plot
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(q, zeta, 'o', color=colors[0], markersize=6, label='ζ(q)')
            q_fine = np.linspace(q.min(), q.max(), 100)
            ax.plot(q_fine, slope_l * q_fine + intercept_l, '-',
                   color=colors[1], linewidth=1.5, label=f'Linear (AIC={aic_l:.2f})')
            ax.plot(q_fine, np.polyval(coeffs_q, q_fine), '--',
                   color=colors[2], linewidth=1.5, label=f'Quadratic (AIC={aic_q:.2f})')
            ax.plot(q_fine, 0.5 * q_fine, ':', color='gray',
                   linewidth=1, label='Normal diffusion (slope=0.5)')
            ax.set_xlabel('q')
            ax.set_ylabel('ζ(q)')
            ax.set_title(f'Linearity test — {station} {stream}\nBest fit: {best_fit}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/linearity_{station}_{stream}.pdf', bbox_inches='tight')
            plt.close()
            saved.append(file)
        except Exception as e:
            failed.append((file, str(e)))
            plt.close()

    df_linearity = pd.DataFrame(rows)

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AIC linear vs quadratic
    x = np.arange(len(df_linearity))
    width = 0.35
    axes[0].bar(x - width/2, df_linearity['aic_linear'], width,
               color=colors[0], label='Linear')
    axes[0].bar(x + width/2, df_linearity['aic_quadratic'], width,
               color=colors[1], label='Quadratic')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_linearity.iterrows()],
                             rotation=90, fontsize=7)
    axes[0].set_title('AIC: linear vs quadratic fit')
    axes[0].set_ylabel('AIC')
    axes[0].legend()

    # Nonlinear count
    counts = df_linearity['best_fit'].value_counts()
    axes[1].bar(counts.index, counts.values, color=[colors[0], colors[2]],
               edgecolor='none')
    axes[1].set_title('Best fit by AIC')
    axes[1].set_ylabel('Count')

    plt.suptitle('Scaling linearity test summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/linearity_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {len(saved)}/{df_exponents['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")
    print(f"\nBest fit by AIC:")
    print(df_linearity['best_fit'].value_counts())

    return df_linearity

# ===============================================================================================
# ========================== Piecewise scaling fit ==============================================
# ===============================================================================================


def fit_piecewise_scaling(df_exponents, output_dir='../figures/03_single_signal/scaling'):
    """
    Fits a piecewise linear model to zeta(q) vs q to detect strong anomalous diffusion.
    Finds the optimal breakpoint q* that separates two linear regimes.
    
    Parameters:
    -----------
    df_exponents : pd.DataFrame — output of compute_scaling_exponents
    output_dir : str — directory to save figures
    
    Returns:
    --------
    df_piecewise : pd.DataFrame with columns [file, station, stream,
                   q_break, slope_low, slope_high, r2_low, r2_high]
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    saved = []
    failed = []

    for file in df_exponents['file'].unique():
        station = df_exponents[df_exponents['file'] == file]['station'].iloc[0]
        stream = df_exponents[df_exponents['file'] == file]['stream'].iloc[0]
        df_file = df_exponents[df_exponents['file'] == file]

        q = df_file['q'].values
        zeta = df_file['zeta'].values

        # Find optimal breakpoint by minimizing total residual sum of squares
        best_rss = np.inf
        best_break = None
        best_slope_low = None
        best_slope_high = None
        best_r2_low = None
        best_r2_high = None

        for i in range(2, len(q) - 2):  # at least 2 points on each side
            q_low, zeta_low = q[:i], zeta[:i]
            q_high, zeta_high = q[i:], zeta[i:]

            slope_l, intercept_l, r_l, _, _ = stats.linregress(q_low, zeta_low)
            slope_h, intercept_h, r_h, _, _ = stats.linregress(q_high, zeta_high)

            rss = np.sum((zeta_low - (slope_l * q_low + intercept_l))**2) + \
                  np.sum((zeta_high - (slope_h * q_high + intercept_h))**2)

            if rss < best_rss:
                best_rss = rss
                best_break = q[i]
                best_slope_low = slope_l
                best_slope_high = slope_h
                best_r2_low = r_l**2
                best_r2_high = r_h**2
                best_intercept_low = intercept_l
                best_intercept_high = intercept_h

        rows.append({
            'file': file,
            'station': station,
            'stream': stream,
            'q_break': round(best_break, 4),
            'slope_low': round(best_slope_low, 4),
            'slope_high': round(best_slope_high, 4),
            'r2_low': round(best_r2_low, 4),
            'r2_high': round(best_r2_high, 4),
        })

        # Plot
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(q, zeta, 'o', color=colors[0], markersize=6, label='ζ(q)')

            q_low = q[q < best_break]
            q_high = q[q >= best_break]
            ax.plot(q_low, best_slope_low * q_low + best_intercept_low, '-',
                   color=colors[1], linewidth=1.5,
                   label=f'Low q (slope={best_slope_low:.2f})')
            ax.plot(q_high, best_slope_high * q_high + best_intercept_high, '-',
                   color=colors[2], linewidth=1.5,
                   label=f'High q (slope={best_slope_high:.2f})')
            ax.axvline(best_break, color='gray', linewidth=0.8,
                      linestyle=':', label=f'q* = {best_break:.2f}')
            ax.plot(q, 0.5 * q, '--', color='gray', linewidth=1,
                   label='Normal diffusion (slope=0.5)')
            ax.set_xlabel('q')
            ax.set_ylabel('ζ(q)')
            ax.set_title(f'Piecewise linear fit — {station} {stream}\n'
                        f'q*={best_break:.2f}, slope_low={best_slope_low:.2f}, '
                        f'slope_high={best_slope_high:.2f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/piecewise_{station}_{stream}.pdf', bbox_inches='tight')
            plt.close()
            saved.append(file)
        except Exception as e:
            failed.append((file, str(e)))
            plt.close()

    df_piecewise = pd.DataFrame(rows)

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. slope_low vs slope_high
    axes[0].scatter(df_piecewise['slope_low'], df_piecewise['slope_high'],
                   color=colors[0], edgecolors='white', linewidths=0.5, s=60)
    axes[0].axline((0, 0), slope=1, color='gray', linewidth=0.8,
                  linestyle='--', label='slope_low = slope_high')
    axes[0].set_xlabel('Slope (low q)')
    axes[0].set_ylabel('Slope (high q)')
    axes[0].set_title('Slope comparison')
    axes[0].legend()

    # 2. q_break distribution
    axes[1].hist(df_piecewise['q_break'], bins=10, color=colors[1], edgecolor='none')
    axes[1].set_xlabel('q*')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Breakpoint distribution')

    # 3. slope_low and slope_high per signal
    x = np.arange(len(df_piecewise))
    axes[2].bar(x - 0.2, df_piecewise['slope_low'], 0.4,
               color=colors[0], label='Low q slope')
    axes[2].bar(x + 0.2, df_piecewise['slope_high'], 0.4,
               color=colors[2], label='High q slope')
    axes[2].axhline(0.5, color='gray', linewidth=0.8, linestyle='--',
                   label='Normal diffusion (0.5)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_piecewise.iterrows()],
                             rotation=90, fontsize=7)
    axes[2].set_title('Slopes by signal')
    axes[2].set_ylabel('Slope')
    axes[2].legend()

    plt.suptitle('Piecewise linear fit summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/piecewise_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {len(saved)}/{df_exponents['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")

    return df_piecewise

# ===============================================================================================
# ========================== Increment tail exponents ===========================================
# ===============================================================================================

def compute_increment_tail_exponents(df_increments, tau_values_plot=None, top_fraction=0.1):
    """
    Estimates the tail exponent of the increment distribution |Delta x(tau)|
    for each signal and each tau value, using two methods:
 
        1. Hill estimator: alpha_hill = (mean of log(x_i / x_k))^{-1}
           where x_1 >= ... >= x_n are the ordered |increments| and
           k = floor(top_fraction * n) is the number of tail observations.
 
        2. Linear fit on the log-log CCDF tail: fits a line to
           log P(|Delta x| > u) vs log u on the top top_fraction of values.
           The slope gives -fit_exp (so fit_exp > 0 means heavy tail).
 
    Parameters
    ----------
    df_increments : pd.DataFrame
        Output of compute_moment_scaling_* with save_increments=True.
        Must contain columns [file, station, stream, tau, increment].
    tau_values_plot : list of int or None
        Subset of tau values to use. If None, uses all tau values in df_increments.
    top_fraction : float
        Fraction of largest |increment| values used for tail estimation (default: 0.1).
 
    Returns
    -------
    df_tail : pd.DataFrame
        Columns: [file, station, stream, tau, hill_exp, fit_exp, r2_fit, n_tail]
    """
    if tau_values_plot is not None:
        df_increments = df_increments[df_increments['tau'].isin(tau_values_plot)]
 
    rows = []
    for (file, tau), group in df_increments.groupby(['file', 'tau']):
        station = group['station'].iloc[0]
        stream = group['stream'].iloc[0]
        abs_inc = np.abs(group['increment'].values)
        abs_inc = abs_inc[abs_inc > 0]
        if len(abs_inc) < 20:
            continue
 
        sorted_inc = np.sort(abs_inc)[::-1]
        k = max(2, int(top_fraction * len(sorted_inc)))
        tail = sorted_inc[:k]
 
        # Hill estimator
        hill_exp = 1.0 / np.mean(np.log(tail / tail[-1]))
 
        # Log-log CCDF fit
        n = len(abs_inc)
        threshold = tail[-1]
        ccdf_x = tail
        ccdf_y = np.arange(1, k + 1) / n
        log_x = np.log(ccdf_x)
        log_y = np.log(ccdf_y)
        slope, _, r, _, _ = stats.linregress(log_x, log_y)
        fit_exp = -slope  # positive value = heavy tail
        r2_fit = r ** 2
 
        rows.append({
            'file': file, 'station': station, 'stream': stream,
            'tau': tau, 'hill_exp': round(hill_exp, 4),
            'fit_exp': round(fit_exp, 4), 'r2_fit': round(r2_fit, 4),
            'n_tail': k
        })
 
    return pd.DataFrame(rows)

# ===============================================================================================
# ========================== Summary ============================================================
# ===============================================================================================

def build_scaling_summary(df_exponents, df_piecewise, process_name):
    """
    Build summary table for moment scaling analysis of a given process.
    
    Parameters
    ----------
    df_exponents : pd.DataFrame
        Scaling exponents from compute_scaling_exponents()
    df_piecewise : pd.DataFrame
        Piecewise fit results from fit_piecewise_scaling()
    process_name : str
        'acceleration', 'velocity', or 'displacement'
    
    Returns
    -------
    pd.DataFrame
        Summary with columns: file, zeta_q2, zeta_q4, q_break, slope_low, 
        slope_high, r2_scaling
    """
    summary_rows = []
    
    for file in df_exponents['file'].unique():
        # Get exponents for this file
        df_file = df_exponents[df_exponents['file'] == file]
        
        # Extract key zeta values
        zeta_q2 = df_file[df_file['q'] == 2.0]['zeta'].values
        zeta_q4 = df_file[df_file['q'] == 4.0]['zeta'].values
        
        # Get piecewise fit parameters
        df_pw = df_piecewise[df_piecewise['file'] == file]
        
        if len(df_pw) > 0:
            q_break = df_pw['q_break'].values[0]
            slope_low = df_pw['slope_low'].values[0]
            slope_high = df_pw['slope_high'].values[0]
            r2 = df_pw.get('r2', pd.Series([np.nan])).values[0]  # R² if available
        else:
            q_break = slope_low = slope_high = r2 = np.nan
        
        summary_rows.append({
            'file': file,
            f'zeta_q2_{process_name}': zeta_q2[0] if len(zeta_q2) > 0 else np.nan,
            f'zeta_q4_{process_name}': zeta_q4[0] if len(zeta_q4) > 0 else np.nan,
            f'q_break_{process_name}': q_break,
            f'slope_low_{process_name}': slope_low,
            f'slope_high_{process_name}': slope_high,
            f'r2_scaling_{process_name}': r2
        })
    
    return pd.DataFrame(summary_rows)
