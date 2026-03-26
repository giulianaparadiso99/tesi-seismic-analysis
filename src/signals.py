import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ==================================== Gaussian fit analysis ====================================
# ===============================================================================================

def gaussian_fit_analysis(df_acc_clean, bins=100, log_scale=False, normalized=True,
                          output_dir='../figures/gaussian_fit'):
    
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    xlabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'
    
    saved = []
    failed = []
    results = []

    for file in df_acc_clean['file'].unique():
        signal = df_acc_clean[df_acc_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        filepath = f'{output_dir}/gaussian_fit_{station}_{stream}.pdf'

        # Fit Gaussian
        mu, std = stats.norm.fit(signal)

        # Anderson-Darling test
        ad_result = stats.anderson(signal, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance level
        ad_significant = ad_stat > ad_critical

        # Kurtosis and skewness (Fisher's definition: kurtosis=0 for Gaussian)
        kurt = stats.kurtosis(signal, fisher=True)
        skew = stats.skew(signal)

        results.append({
            'file': file,
            'station': station,
            'stream': stream,
            'mu': round(mu, 6),
            'std': round(std, 6),
            'kurtosis': round(kurt, 4),
            'skewness': round(skew, 4),
            'ad_statistic': round(ad_stat, 4),
            'ad_critical_5pct': round(ad_critical, 4),
            'non_gaussian': ad_significant
        })

        # Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 1. Histogram + Gaussian fit
            axes[0].hist(signal, bins=bins, color=colors[0], edgecolor='none',
                        density=True, alpha=0.7, label='Empirical PDF')
            x = np.linspace(signal.min(), signal.max(), 500)
            axes[0].plot(x, stats.norm.pdf(x, mu, std), color=colors[2],
                        linewidth=1.5, label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')
            if log_scale:
                axes[0].set_yscale('log')
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel('Probability density')
            axes[0].set_title(f'Gaussian fit — {station} {stream}\n'
                            f'Kurt={kurt:.2f}, Skew={skew:.2f}, AD={ad_stat:.2f} (crit={ad_critical:.2f})')
            axes[0].legend()

            # 2. Q-Q plot vs Gaussian
            (osm, osr), (slope, intercept, r) = stats.probplot(signal, dist='norm')
            axes[1].scatter(osm, osr, color=colors[0], s=2, alpha=0.5, label='Data')
            axes[1].plot(osm, slope * np.array(osm) + intercept,
                        color=colors[2], linewidth=1.5, label='Gaussian')
            axes[1].set_xlabel('Theoretical quantiles')
            axes[1].set_ylabel('Sample quantiles')
            axes[1].set_title(f'Q-Q plot vs Gaussian — {station} {stream}\nR²={r**2:.4f}')
            axes[1].legend()

            plt.suptitle(f'Gaussian fit — {station} {stream}', fontsize=13)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)
        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()

    # Summary dataframe
    df_results = pd.DataFrame(results)

    # Summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Kurtosis per station
    df_sorted_kurt = df_results.sort_values('kurtosis')
    axes[0, 0].bar(range(len(df_sorted_kurt)), df_sorted_kurt['kurtosis'],
                   color=[colors[0] if not ng else colors[3] 
                          for ng in df_sorted_kurt['non_gaussian']],
                   edgecolor='none')
    axes[0, 0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0, 0].set_xticks(range(len(df_sorted_kurt)))
    axes[0, 0].set_xticklabels([f"{r['station']}\n{r['stream']}" 
                                  for _, r in df_sorted_kurt.iterrows()],
                                 rotation=90, fontsize=7)
    axes[0, 0].set_title('Kurtosis by signal')
    axes[0, 0].set_ylabel('Kurtosis (Fisher)')

    # 2. Skewness per station
    df_sorted_skew = df_results.sort_values('skewness')
    axes[0, 1].bar(range(len(df_sorted_skew)), df_sorted_skew['skewness'],
                   color=[colors[0] if not ng else colors[3] 
                          for ng in df_sorted_skew['non_gaussian']],
                   edgecolor='none')
    axes[0, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0, 1].set_xticks(range(len(df_sorted_skew)))
    axes[0, 1].set_xticklabels([f"{r['station']}\n{r['stream']}" 
                                  for _, r in df_sorted_skew.iterrows()],
                                 rotation=90, fontsize=7)
    axes[0, 1].set_title('Skewness by signal')
    axes[0, 1].set_ylabel('Skewness')

    # 3. Anderson-Darling statistic
    df_sorted_ad = df_results.sort_values('ad_statistic', ascending=False)
    axes[1, 0].bar(range(len(df_sorted_ad)), df_sorted_ad['ad_statistic'],
                   color=[colors[3] if ng else colors[0] 
                          for ng in df_sorted_ad['non_gaussian']],
                   edgecolor='none')
    axes[1, 0].axhline(df_results['ad_critical_5pct'].iloc[0], color='black',
                        linewidth=0.8, linestyle='--', label='Critical value (5%)')
    axes[1, 0].set_xticks(range(len(df_sorted_ad)))
    axes[1, 0].set_xticklabels([f"{r['station']}\n{r['stream']}" 
                                  for _, r in df_sorted_ad.iterrows()],
                                 rotation=90, fontsize=7)
    axes[1, 0].set_title('Anderson-Darling statistic by signal')
    axes[1, 0].set_ylabel('AD statistic')
    axes[1, 0].legend()

    # 4. Kurtosis vs skewness
    stream_colors = {'E': colors[0], 'N': colors[1], 'Z': colors[2]}
    for _, row in df_results.iterrows():
        component = row['stream'][-1]
        axes[1, 1].scatter(row['skewness'], row['kurtosis'],
                           color=stream_colors.get(component, colors[0]),
                           edgecolors='white', linewidths=0.5, s=60, zorder=5)
    axes[1, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1, 1].axvline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1, 1].set_xlabel('Skewness')
    axes[1, 1].set_ylabel('Kurtosis (Fisher)')
    axes[1, 1].set_title('Kurtosis vs skewness by component')
    for component, c in stream_colors.items():
        axes[1, 1].scatter([], [], color=c, label=component)
    axes[1, 1].legend(title='Component')

    plt.suptitle('Gaussian fit summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gaussian_fit_summary.pdf', bbox_inches='tight')
    plt.close()

    # Print check
    print(f"Saved: {len(saved)}/{df_acc_clean['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")
    print(f"Non-Gaussian signals (AD test, p<5%): {df_results['non_gaussian'].sum()}/{len(df_results)}")

    return df_results

# ===============================================================================================
# ==================================== Heavy-tail assessment ====================================
# ===============================================================================================

def heavy_tail_assessment(df_acc_clean, normalized=True, output_dir='../figures/heavy_tail',
                          resume=False, partial_path='../data/processed/heavy_tail_results_partial.parquet'):
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    xlabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'
    
    saved = []
    failed = []
    results = []

    # Resume from partial results if requested
    if resume:
        try:
            df_partial = pd.read_parquet(partial_path)
            processed_files = set(df_partial['file'].values)
            results = df_partial.to_dict('records')
            print(f"Resuming from {len(processed_files)} already processed signals")
        except Exception:
            print("No partial results found, starting from scratch")
            processed_files = set()
            results = []
    else:
        processed_files = set()
        results = []

    for file in df_acc_clean['file'].unique():
        if file in processed_files:
            continue

        signal = df_acc_clean[df_acc_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        filepath = f'{output_dir}/heavy_tail_{station}_{stream}.pdf'

        # --- Fit distributions ---
        # Gaussian
        mu_g, std_g = stats.norm.fit(signal)
        loglik_g = np.sum(stats.norm.logpdf(signal, mu_g, std_g))
        k_g = 2
        aic_g = 2 * k_g - 2 * loglik_g
        bic_g = k_g * np.log(len(signal)) - 2 * loglik_g

        # Laplace
        mu_l, b_l = stats.laplace.fit(signal)
        loglik_l = np.sum(stats.laplace.logpdf(signal, mu_l, b_l))
        k_l = 2
        aic_l = 2 * k_l - 2 * loglik_l
        bic_l = k_l * np.log(len(signal)) - 2 * loglik_l

        # Student-t
        df_t, mu_t, scale_t = stats.t.fit(signal)
        loglik_t = np.sum(stats.t.logpdf(signal, df_t, mu_t, scale_t))
        k_t = 3
        aic_t = 2 * k_t - 2 * loglik_t
        bic_t = k_t * np.log(len(signal)) - 2 * loglik_t

        # Levy stable — subsample for computational efficiency
        n_subsample = 5000
        if len(signal) > n_subsample:
            signal_sample = np.random.choice(signal, size=n_subsample, replace=False)
        else:
            signal_sample = signal
        alpha_s, beta_s, loc_s, scale_s = stats.levy_stable.fit(signal_sample, method='mle')
        loglik_s = np.sum(stats.levy_stable.logpdf(signal, alpha_s, beta_s, loc_s, scale_s))
        k_s = 4
        aic_s = 2 * k_s - 2 * loglik_s
        bic_s = k_s * np.log(len(signal)) - 2 * loglik_s

        # Best fit by AIC
        aic_dict = {'Gaussian': aic_g, 'Laplace': aic_l, 
                    'Student-t': aic_t, 'Levy-stable': aic_s}
        best_fit = min(aic_dict, key=aic_dict.get)

        # --- Power law exponent from CCDF ---
        abs_signal = np.abs(signal)
        threshold = np.percentile(abs_signal, 90)  # fit on top 10% of values
        tail = abs_signal[abs_signal > threshold]
        # Hill estimator
        tail_sorted = np.sort(tail)[::-1]
        hill_exp = 1 / np.mean(np.log(tail_sorted / tail_sorted[-1]))

        results.append({
            'file': file,
            'station': station,
            'stream': stream,
            'aic_gaussian': round(aic_g, 2),
            'aic_laplace': round(aic_l, 2),
            'aic_student_t': round(aic_t, 2),
            'aic_levy_stable': round(aic_s, 2),
            'bic_gaussian': round(bic_g, 2),
            'bic_laplace': round(bic_l, 2),
            'bic_student_t': round(bic_t, 2),
            'bic_levy_stable': round(bic_s, 2),
            'best_fit_aic': best_fit,
            'student_t_df': round(df_t, 4),
            'levy_alpha': round(alpha_s, 4),
            'levy_beta': round(beta_s, 4),
            'power_law_exp': round(hill_exp, 4),
        })
        # Incremental save after each signal
        df_partial = pd.DataFrame(results)
        try:
            df_partial.to_parquet('../data/processed/heavy_tail_results_partial.parquet', index=False)
        except Exception as e:
            print(f"Warning: could not save partial results: {e}")
    
        # --- Plot ---
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            x = np.linspace(signal.min(), signal.max(), 500)

            # 1. PDF with fits
            axes[0].hist(signal, bins=100, color=colors[0], edgecolor='none',
                        density=True, alpha=0.7, label='Empirical')
            axes[0].plot(x, stats.norm.pdf(x, mu_g, std_g),
                        color=colors[1], linewidth=1.5, label='Gaussian')
            axes[0].plot(x, stats.laplace.pdf(x, mu_l, b_l),
                        color=colors[2], linewidth=1.5, label='Laplace')
            axes[0].plot(x, stats.t.pdf(x, df_t, mu_t, scale_t),
                        color=colors[3], linewidth=1.5, label=f'Student-t (df={df_t:.1f})')
            axes[0].plot(x, stats.levy_stable.pdf(x, alpha_s, beta_s, loc_s, scale_s),
                        color=colors[3], linewidth=1.5, 
                        label=f'Lévy stable (α={alpha_s:.2f}, β={beta_s:.2f})')
            axes[0].set_yscale('log')
            axes[0].set_xlabel(xlabel)
            axes[0].set_ylabel('Probability density (log scale)')
            axes[0].set_title(f'PDF fits — {station} {stream}\nBest: {best_fit} (AIC)')
            axes[0].legend()

            # 2. Q-Q plot vs Gaussian
            (osm, osr), (slope, intercept, r) = stats.probplot(signal, dist='norm')
            axes[1].scatter(osm, osr, color=colors[0], s=2, alpha=0.5, label='Data')
            axes[1].plot(osm, slope * np.array(osm) + intercept,
                        color=colors[2], linewidth=1.5, label='Gaussian')
            axes[1].set_xlabel('Theoretical quantiles')
            axes[1].set_ylabel('Sample quantiles')
            axes[1].set_title(f'Q-Q plot vs Gaussian — {station} {stream}')
            axes[1].legend()

            # 3. Log-log CCDF
            abs_signal_sorted = np.sort(abs_signal)[::-1]
            ccdf = np.arange(1, len(abs_signal_sorted) + 1) / len(abs_signal_sorted)
            axes[2].loglog(abs_signal_sorted, ccdf, color=colors[0],
                        linewidth=0.8, label='Empirical CCDF')
            # Power law fit line on tail
            x_tail = np.linspace(threshold, abs_signal_sorted.max(), 100)
            c = ccdf[np.searchsorted(-abs_signal_sorted, -threshold)]
            axes[2].loglog(x_tail, c * (x_tail / threshold) ** (-hill_exp),
                        color=colors[3], linewidth=1.5,
                        linestyle='--', label=f'Power law (α={hill_exp:.2f})')
            axes[2].axvline(threshold, color='gray', linewidth=0.8,
                        linestyle=':', label='Threshold (90th pct)')
            axes[2].set_xlabel(f'|{xlabel}|')
            axes[2].set_ylabel('CCDF')
            axes[2].set_title(f'Log-log CCDF — {station} {stream}')
            axes[2].legend()

            plt.suptitle(f'Heavy-tail assessment — {station} {stream}', fontsize=13)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)

        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()

    # Summary dataframe
    df_results = pd.DataFrame(results)

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. AIC comparison
    x = np.arange(len(df_results))
    width = 0.25
    axes[0].bar(x - width, df_results['aic_gaussian'], width,
               color=colors[0], label='Gaussian')
    axes[0].bar(x, df_results['aic_laplace'], width,
               color=colors[1], label='Laplace')
    axes[0].bar(x + width, df_results['aic_student_t'], width,
               color=colors[2], label='Student-t')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_results.iterrows()],
                             rotation=90, fontsize=7)
    axes[0].set_title('AIC comparison')
    axes[0].set_ylabel('AIC')
    axes[0].legend()

    # 2. Student-t degrees of freedom
    df_sorted = df_results.sort_values('student_t_df')
    axes[1].bar(range(len(df_sorted)), df_sorted['student_t_df'],
               color=colors[1], edgecolor='none')
    axes[1].set_xticks(range(len(df_sorted)))
    axes[1].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_sorted.iterrows()],
                             rotation=90, fontsize=7)
    axes[1].set_title('Student-t degrees of freedom\n(lower = heavier tails)')
    axes[1].set_ylabel('Degrees of freedom')

    # 3. Power law exponent
    df_sorted_pl = df_results.sort_values('power_law_exp')
    axes[2].bar(range(len(df_sorted_pl)), df_sorted_pl['power_law_exp'],
               color=colors[2], edgecolor='none')
    axes[2].set_xticks(range(len(df_sorted_pl)))
    axes[2].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_sorted_pl.iterrows()],
                             rotation=90, fontsize=7)
    axes[2].set_title('Power law exponent (Hill estimator)\n(lower = heavier tails)')
    axes[2].set_ylabel('α')

    plt.suptitle('Heavy-tail assessment summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heavy_tail_summary.pdf', bbox_inches='tight')
    plt.close()

    # Check
    print(f"Saved: {len(saved)}/{df_acc_clean['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")
    print(f"\nBest fit by AIC:")
    print(df_results['best_fit_aic'].value_counts())

    return df_results

# ===============================================================================================
# ==================================== Moment scaling - Acceleration ============================
# ===============================================================================================

def compute_moment_scaling_acc(df_acc_clean, q_values, tau_values, normalized=True,
                               save_increments=False):
    """
    Computes q-th order moments of acceleration increments at different
    time scales tau.
 
    The process is the acceleration signal a(t) itself. Increments are
    defined as:
 
        Delta_a(tau, t0) = a(t0 + tau) - a(t0)
 
    The q-th order moment is the temporal average:
 
        M_q(tau) = < |Delta_a(tau, t0)|^q >_{t0}
 
    Parameters
    ----------
    df_acc_clean : pd.DataFrame
    q_values : list of float
    tau_values : list of int
    normalized : bool
    save_increments : bool
        If True, also return a DataFrame with all raw increments.
 
    Returns
    -------
    df_moments : pd.DataFrame with columns [file, station, stream, q, tau, moment]
    df_increments : pd.DataFrame with columns [file, station, stream, tau, increment]
        Only returned if save_increments=True.
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    rows = []
    inc_rows = []
 
    for file in df_acc_clean['file'].unique():
        signal = df_acc_clean[df_acc_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        N = len(signal)
 
        for tau in tau_values:
            if (N - tau) < 10:
                continue
 
            # Increments of acceleration: a(t0 + tau) - a(t0)
            increments = signal[tau:] - signal[:-tau]
 
            if save_increments:
                for inc in increments:
                    inc_rows.append({
                        'file': file, 'station': station, 'stream': stream,
                        'tau': tau, 'increment': inc
                    })
 
            for q in q_values:
                moment = np.mean(np.abs(increments) ** q)
                rows.append({
                    'file': file, 'station': station, 'stream': stream,
                    'q': q, 'tau': tau, 'moment': moment
                })
 
    df_moments = pd.DataFrame(rows)
    if save_increments:
        return df_moments, pd.DataFrame(inc_rows)
    return df_moments

# ===============================================================================================
# ==================================== Moment scaling - Velocity ================================
# ===============================================================================================

def compute_moment_scaling_vel(df_acc_clean, q_values, tau_values, normalized=True,
                               dt=0.005, save_increments=False):
    """
    Computes q-th order moments of velocity increments at different
    time scales tau.
 
    The velocity is obtained by integrating the acceleration once:
 
        v(t) = cumsum(a(t)) * dt
 
    Increments are defined as:
 
        Delta_v(tau, t0) = v(t0 + tau) - v(t0)
 
    The q-th order moment is the temporal average:
 
        M_q(tau) = < |Delta_v(tau, t0)|^q >_{t0}
 
    Parameters
    ----------
    df_acc_clean : pd.DataFrame
    q_values : list of float
    tau_values : list of int
    normalized : bool
    dt : float — sampling interval in seconds (default: 0.005 s = 200 Hz)
    save_increments : bool
        If True, also return a DataFrame with all raw increments.
 
    Returns
    -------
    df_moments : pd.DataFrame with columns [file, station, stream, q, tau, moment]
    df_increments : pd.DataFrame with columns [file, station, stream, tau, increment]
        Only returned if save_increments=True.
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    rows = []
    inc_rows = []
 
    for file in df_acc_clean['file'].unique():
        signal = df_acc_clean[df_acc_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        N = len(signal)
 
        # Integrate acceleration once to get velocity
        v = np.cumsum(signal) * dt
 
        for tau in tau_values:
            if (N - tau) < 10:
                continue
 
            # Increments of velocity: v(t0 + tau) - v(t0)
            increments = v[tau:] - v[:-tau]
 
            if save_increments:
                for inc in increments:
                    inc_rows.append({
                        'file': file, 'station': station, 'stream': stream,
                        'tau': tau, 'increment': inc
                    })
 
            for q in q_values:
                moment = np.mean(np.abs(increments) ** q)
                rows.append({
                    'file': file, 'station': station, 'stream': stream,
                    'q': q, 'tau': tau, 'moment': moment
                })
 
    df_moments = pd.DataFrame(rows)
    if save_increments:
        return df_moments, pd.DataFrame(inc_rows)
    return df_moments

# ===============================================================================================
# ==================================== Moment scaling - Displacement ============================
# ===============================================================================================

def compute_moment_scaling_disp(df_acc, q_values, tau_values, normalized=True,
                                dt=0.005, save_increments=False):
    """
    Computes q-th order moments of displacement increments at different
    time scales tau.
 
    The displacement is obtained by integrating the acceleration twice:
 
        v(t) = cumsum(a(t)) * dt        (velocity)
        x(t) = cumsum(v(t)) * dt        (displacement)
 
    Increments are defined as:
 
        Delta_x(tau, t0) = x(t0 + tau) - x(t0)
 
    The q-th order moment is the temporal average:
 
        M_q(tau) = < |Delta_x(tau, t0)|^q >_{t0}
 
    Parameters
    ----------
    df_acc : pd.DataFrame
    q_values : list of float
    tau_values : list of int
    normalized : bool
    dt : float — sampling interval in seconds (default: 0.005 s = 200 Hz)
    save_increments : bool
        If True, also return a DataFrame with all raw increments.
 
    Returns
    -------
    df_moments : pd.DataFrame with columns [file, station, stream, q, tau, moment]
    df_increments : pd.DataFrame with columns [file, station, stream, tau, increment]
        Only returned if save_increments=True.
    """
    col = 'acceleration_normalized' if normalized else 'acceleration'
    rows = []
    inc_rows = []
 
    for file in df_acc['file'].unique():
        signal = df_acc[df_acc['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        N = len(signal)
 
        # Integrate acceleration twice to get displacement
        v = np.cumsum(signal) * dt
        x = np.cumsum(v) * dt
 
        for tau in tau_values:
            if (N - tau) < 10:
                continue
 
            # Increments of displacement: x(t0 + tau) - x(t0)
            increments = x[tau:] - x[:-tau]
 
            if save_increments:
                for inc in increments:
                    inc_rows.append({
                        'file': file, 'station': station, 'stream': stream,
                        'tau': tau, 'increment': inc
                    })
 
            for q in q_values:
                moment = np.mean(np.abs(increments) ** q)
                rows.append({
                    'file': file, 'station': station, 'stream': stream,
                    'q': q, 'tau': tau, 'moment': moment
                })
 
    df_moments = pd.DataFrame(rows)
    if save_increments:
        return df_moments, pd.DataFrame(inc_rows)
    return df_moments

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
                         normalized=True):
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
            axes[0].set_ylabel('log(<|a|^q>)')
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
# ========================== Displacement autocorrelation =======================================
# ===============================================================================================

def compute_displacement_autocorrelation(df_acc, n_points=50, normalized=True,
                                          output_dir='../figures/03_single_signal/autocorrelation'):
    """
    Computes displacement autocorrelation functions C(t1, t2) = <(a(t1)-a0)(a(t2)-a0)>
    on a logarithmic grid of (t1, t2) pairs, following Vollmer et al. 2021.
    
    Parameters:
    -----------
    df_acc : pd.DataFrame
    n_points : int — number of points in the logarithmic grid for t1 and t2
    normalized : bool — use acceleration_normalized or acceleration
    output_dir : str — directory to save figures
    
    Returns:
    --------
    df_autocorr : pd.DataFrame with columns [file, station, stream, t1, t2, C]
    """
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    rows = []
    saved = []
    failed = []

    for file in df_acc['file'].unique():
        signal = df_acc[df_acc['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        filepath = f'{output_dir}/autocorr_{station}_{stream}.pdf'

        n = len(signal)
        a0 = signal[0]

        # Logarithmic grid of t1 and t2
        t_grid = np.unique(np.logspace(0, np.log10(n - 1), n_points).astype(int))
        t_grid = t_grid[t_grid < n]

        # Compute C(t1, t2) for all pairs in the grid
        C_matrix = np.zeros((len(t_grid), len(t_grid)))
        for i, t1 in enumerate(t_grid):
            for j, t2 in enumerate(t_grid):
                C_matrix[i, j] = (signal[t1] - a0) * (signal[t2] - a0)

        for i, t1 in enumerate(t_grid):
            for j, t2 in enumerate(t_grid):
                rows.append({
                    'file': file,
                    'station': station,
                    'stream': stream,
                    't1': t1,
                    't2': t2,
                    'C': C_matrix[i, j]
                })

        # Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 1. Heatmap of C(t1, t2)
            im = axes[0].pcolormesh(t_grid, t_grid, C_matrix,
                                    cmap='inferno', shading='auto')
            plt.colorbar(im, ax=axes[0], label='C(t1, t2)')
            axes[0].set_xlabel('t1')
            axes[0].set_ylabel('t2')
            axes[0].set_title(f'Autocorrelation C(t1, t2) — {station} {stream}')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')

            # 2. C(t, t) diagonal — scaling behavior
            diag = np.diag(C_matrix)
            axes[1].loglog(t_grid, np.abs(diag), color=colors[0],
                          linewidth=1.2, label='C(t, t)')
            axes[1].set_xlabel('t')
            axes[1].set_ylabel('|C(t, t)|')
            axes[1].set_title(f'Diagonal scaling — {station} {stream}')
            axes[1].legend()

            plt.suptitle(f'Displacement autocorrelation — {station} {stream}', fontsize=13)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)

        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()

    df_autocorr = pd.DataFrame(rows)

    print(f"Saved: {len(saved)}/{df_acc['file'].nunique()} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")

    # Build dictionary of C matrices for scaling analysis
    C_matrices = {}
    for file in df_acc['file'].unique():
        station = df_acc[df_acc['file'] == file]['station'].iloc[0] if 'station' in df_acc.columns else file.split('.')[1]
        stream = file.split('.')[3]
        df_file = df_autocorr[df_autocorr['file'] == file]
        t_grid = sorted(df_file['t1'].unique())
        n = len(t_grid)
        C = df_file.pivot(index='t1', columns='t2', values='C').values
        C_matrices[file] = {'C': C, 't_grid': np.array(t_grid), 
                            'station': station, 'stream': stream}

    return df_autocorr, C_matrices

# ===============================================================================================
# ========================== Autocorrelation scaling ============================================
# ===============================================================================================

def analyze_autocorrelation_scaling(C_matrices, output_dir='../figures/03_single_signal/autocorrelation'):
    """
    Analyzes the scaling behavior of displacement autocorrelation functions.
    Estimates the scaling exponent beta from C(t, t) ~ t^beta.
    
    Parameters:
    -----------
    C_matrices : dict — output of compute_displacement_autocorrelation
    output_dir : str — directory to save figures
    
    Returns:
    --------
    df_scaling : pd.DataFrame with columns [file, station, stream, beta, r2]
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    saved = []
    failed = []

    for file, data in C_matrices.items():
        C = data['C']
        t_grid = data['t_grid']
        station = data['station']
        stream = data['stream']
        filepath = f'{output_dir}/autocorr_scaling_{station}_{stream}.pdf'

        # Diagonal C(t, t)
        diag = np.diag(C)
        abs_diag = np.abs(diag)

        # Keep only positive values for log-log fit
        mask = abs_diag > 0
        t_fit = t_grid[mask]
        diag_fit = abs_diag[mask]

        # Estimate scaling exponent beta: C(t,t) ~ t^beta
        slope, intercept, r, _, _ = stats.linregress(np.log(t_fit), np.log(diag_fit))
        beta = slope

        rows.append({
            'file': file,
            'station': station,
            'stream': stream,
            'beta': round(beta, 4),
            'r2': round(r**2, 4)
        })

        # Plot
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 1. C(t,t) with power law fit
            axes[0].loglog(t_grid, abs_diag, 'o', color=colors[0],
                          markersize=4, alpha=0.7, label='|C(t, t)|')
            t_fine = np.linspace(t_fit.min(), t_fit.max(), 100)
            axes[0].loglog(t_fine, np.exp(intercept) * t_fine**beta, '-',
                          color=colors[2], linewidth=1.5,
                          label=f'Power law fit (β={beta:.2f})')
            # Reference lines for normal diffusion (beta=1)
            axes[0].loglog(t_fine, np.exp(intercept) * t_fine**1, '--',
                          color='gray', linewidth=1, label='β=1 (normal diffusion)')
            axes[0].set_xlabel('t')
            axes[0].set_ylabel('|C(t, t)|')
            axes[0].set_title(f'Diagonal scaling — {station} {stream}\nβ={beta:.2f}, R²={r**2:.4f}')
            axes[0].legend()

            # 2. C(t1, t2) heatmap
            im = axes[1].pcolormesh(t_grid, t_grid, C, cmap='inferno', shading='auto')
            plt.colorbar(im, ax=axes[1], label='C(t1, t2)')
            axes[1].set_xscale('log')
            axes[1].set_yscale('log')
            axes[1].set_xlabel('t1')
            axes[1].set_ylabel('t2')
            axes[1].set_title(f'C(t1, t2) heatmap — {station} {stream}')

            plt.suptitle(f'Autocorrelation scaling — {station} {stream}', fontsize=13)
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)

        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()

    df_scaling = pd.DataFrame(rows)

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Beta distribution
    df_sorted = df_scaling.sort_values('beta')
    axes[0].bar(range(len(df_sorted)), df_sorted['beta'],
               color=[colors[0] if b > 0 else colors[3] for b in df_sorted['beta']],
               edgecolor='none')
    axes[0].axhline(1, color='gray', linewidth=0.8, linestyle='--',
                   label='β=1 (normal diffusion)')
    axes[0].set_xticks(range(len(df_sorted)))
    axes[0].set_xticklabels([f"{r['station']}\n{r['stream']}"
                              for _, r in df_sorted.iterrows()],
                             rotation=90, fontsize=7)
    axes[0].set_title('Scaling exponent β by signal')
    axes[0].set_ylabel('β')
    axes[0].legend()

    # 2. Beta vs R²
    axes[1].scatter(df_scaling['beta'], df_scaling['r2'],
                   color=colors[0], edgecolors='white', linewidths=0.5, s=60)
    axes[1].axvline(1, color='gray', linewidth=0.8, linestyle='--',
                   label='β=1 (normal diffusion)')
    axes[1].set_xlabel('β')
    axes[1].set_ylabel('R²')
    axes[1].set_title('β vs R²')
    axes[1].legend()

    plt.suptitle('Autocorrelation scaling summary', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/autocorr_scaling_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {len(saved)}/{len(C_matrices)} individual plots")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All individual plots saved successfully!")

    return df_scaling

# ===============================================================================================
# ========================== PDF comparison by distance group ===================================
# ===============================================================================================

def plot_pdfs_by_group(df_acc_agg, groups, group_colors, bins=100, 
                       log_scale=True, normalized=True,
                       output_dir='../figures/04_aggregated'):
    """
    Plots empirical PDFs for each distance group on the same axes
    for visual comparison.
    
    Parameters:
    -----------
    df_acc_agg : pd.DataFrame — aggregated preprocessed acceleration data
                 must contain a 'DISTANCE_GROUP' column
    groups : list — distance group labels (e.g. ['Near', 'Mid', 'Far'])
    group_colors : list — colors for each group
    bins : int — number of histogram bins
    log_scale : bool — use log scale on y axis
    normalized : bool — use acceleration_normalized or acceleration
    output_dir : str — directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    xlabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, group in enumerate(groups):
        signal = df_acc_agg[df_acc_agg['DISTANCE_GROUP'] == group][col].values
        ax.hist(signal, bins=bins, color=group_colors[i], edgecolor='none',
                density=True, alpha=0.6, label=group)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density' + (' (log scale)' if log_scale else ''))
    ax.set_title('PDF comparison across distance groups')
    ax.legend(title='Distance group')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pdf_by_group.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: pdf_by_group.pdf")

# ===============================================================================================
# ========================== Gaussian fit by distance group =====================================
# ===============================================================================================

def gaussian_fit_by_group(df_acc_agg, groups, bins=100, log_scale=True,
                           normalized=True, output_dir='../figures/04_aggregated'):
    """
    Performs Gaussian fit and normality assessment for each distance group.
    
    Parameters:
    -----------
    df_acc_agg : pd.DataFrame — must contain a 'DISTANCE_GROUP' column
    groups : list — distance group labels
    bins : int — number of histogram bins
    log_scale : bool — use log scale on y axis
    normalized : bool — use acceleration_normalized or acceleration
    output_dir : str — directory to save figures

    Returns:
    --------
    df_results : pd.DataFrame with columns [group, mu, std, kurtosis, 
                 skewness, ad_statistic, ad_critical_5pct, non_gaussian]
    """
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    xlabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'
    results = []

    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5))

    for i, group in enumerate(groups):
        signal = df_acc_agg[df_acc_agg['DISTANCE_GROUP'] == group][col].values

        # Gaussian fit
        mu, std = stats.norm.fit(signal)

        # Anderson-Darling test
        ad_result = stats.anderson(signal, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]
        ad_significant = ad_stat > ad_critical

        # Kurtosis and skewness
        kurt = stats.kurtosis(signal, fisher=True)
        skew = stats.skew(signal)

        results.append({
            'group': group,
            'mu': round(mu, 6),
            'std': round(std, 6),
            'kurtosis': round(kurt, 4),
            'skewness': round(skew, 4),
            'ad_statistic': round(ad_stat, 4),
            'ad_critical_5pct': round(ad_critical, 4),
            'non_gaussian': ad_significant
        })

        # Plot
        axes[i].hist(signal, bins=bins, color=colors[i % len(colors)],
                    edgecolor='none', density=True, alpha=0.7, label='Empirical')
        x = np.linspace(signal.min(), signal.max(), 500)
        axes[i].plot(x, stats.norm.pdf(x, mu, std), color=colors[2],
                    linewidth=1.5, label=f'Gaussian (μ={mu:.2f}, σ={std:.2f})')
        if log_scale:
            axes[i].set_yscale('log')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Probability density')
        axes[i].set_title(f'{group} stations\nKurt={kurt:.2f}, Skew={skew:.2f}, '
                         f'AD={ad_stat:.2f} (crit={ad_critical:.2f})')
        axes[i].legend()

    plt.suptitle('Gaussian fit by distance group', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gaussian_fit_by_group.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: gaussian_fit_by_group.pdf")

    return pd.DataFrame(results)

# ===============================================================================================
# ========================== Heavy-tail assessment by distance group ============================
# ===============================================================================================

def heavy_tail_by_group(df_acc_agg, groups, normalized=True,
                         output_dir='../figures/04_aggregated'):
    """
    Performs heavy-tail assessment for each distance group.
    Fits Gaussian, Laplace, Student-t and Lévy stable distributions
    and compares them using AIC.

    Parameters:
    -----------
    df_acc_agg : pd.DataFrame — must contain a 'DISTANCE_GROUP' column
    groups : list — distance group labels
    normalized : bool — use acceleration_normalized or acceleration
    output_dir : str — directory to save figures

    Returns:
    --------
    df_results : pd.DataFrame with columns [group, aic_gaussian, aic_laplace,
                 aic_student_t, aic_levy_stable, best_fit, student_t_df,
                 levy_alpha, levy_beta, power_law_exp]
    """
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    xlabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'
    results = []

    for i, group in enumerate(groups):
        signal = df_acc_agg[df_acc_agg['DISTANCE_GROUP'] == group][col].values

        # Gaussian
        mu_g, std_g = stats.norm.fit(signal)
        aic_g = 2*2 - 2*np.sum(stats.norm.logpdf(signal, mu_g, std_g))

        # Laplace
        mu_l, b_l = stats.laplace.fit(signal)
        aic_l = 2*2 - 2*np.sum(stats.laplace.logpdf(signal, mu_l, b_l))

        # Student-t
        df_t, mu_t, scale_t = stats.t.fit(signal)
        aic_t = 2*3 - 2*np.sum(stats.t.logpdf(signal, df_t, mu_t, scale_t))

        # Levy stable
        n_subsample = 5000
        signal_sample = np.random.choice(signal, size=n_subsample, replace=False)
        alpha_s, beta_s, loc_s, scale_s = stats.levy_stable.fit(signal_sample, method='mle')
        aic_s = 2*4 - 2*np.sum(stats.levy_stable.logpdf(signal, alpha_s, beta_s, loc_s, scale_s))

        # Hill estimator
        abs_signal = np.abs(signal)
        threshold = np.percentile(abs_signal, 90)
        tail = abs_signal[abs_signal > threshold]
        tail_sorted = np.sort(tail)[::-1]
        hill_exp = 1 / np.mean(np.log(tail_sorted / tail_sorted[-1]))

        aic_dict = {'Gaussian': aic_g, 'Laplace': aic_l,
                    'Student-t': aic_t, 'Levy-stable': aic_s}
        best_fit = min(aic_dict, key=aic_dict.get)

        results.append({
            'group': group,
            'aic_gaussian': round(aic_g, 2),
            'aic_laplace': round(aic_l, 2),
            'aic_student_t': round(aic_t, 2),
            'aic_levy_stable': round(aic_s, 2),
            'best_fit': best_fit,
            'student_t_df': round(df_t, 4),
            'levy_alpha': round(alpha_s, 4),
            'levy_beta': round(beta_s, 4),
            'power_law_exp': round(hill_exp, 4),
        })

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x = np.linspace(signal.min(), signal.max(), 500)

        # 1. PDF with fits
        axes[0].hist(signal, bins=100, color=colors[i % len(colors)],
                    edgecolor='none', density=True, alpha=0.7, label='Empirical')
        axes[0].plot(x, stats.norm.pdf(x, mu_g, std_g),
                    color=colors[1], linewidth=1.5, label='Gaussian')
        axes[0].plot(x, stats.laplace.pdf(x, mu_l, b_l),
                    color=colors[2], linewidth=1.5, label='Laplace')
        axes[0].plot(x, stats.t.pdf(x, df_t, mu_t, scale_t),
                    color=colors[3], linewidth=1.5, label=f'Student-t (df={df_t:.2f})')
        axes[0].plot(x, stats.levy_stable.pdf(x, alpha_s, beta_s, loc_s, scale_s),
                    color=colors[0], linewidth=1.5, linestyle='--',
                    label=f'Lévy stable (α={alpha_s:.2f})')
        axes[0].set_yscale('log')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Probability density (log scale)')
        axes[0].set_title(f'PDF fits — {group}\nBest: {best_fit} (AIC)')
        axes[0].legend()

        # 2. Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(signal, dist='norm')
        axes[1].scatter(osm, osr, color=colors[i % len(colors)], s=2, alpha=0.5)
        axes[1].plot(osm, slope * np.array(osm) + intercept,
                    color=colors[2], linewidth=1.5, label='Gaussian')
        axes[1].set_xlabel('Theoretical quantiles')
        axes[1].set_ylabel('Sample quantiles')
        axes[1].set_title(f'Q-Q plot — {group}\nR²={r**2:.4f}')
        axes[1].legend()

        # 3. Log-log CCDF
        abs_signal_sorted = np.sort(abs_signal)[::-1]
        ccdf = np.arange(1, len(abs_signal_sorted) + 1) / len(abs_signal_sorted)
        axes[2].loglog(abs_signal_sorted, ccdf, color=colors[i % len(colors)],
                      linewidth=0.8, label='Empirical CCDF')
        x_tail = np.linspace(threshold, abs_signal_sorted.max(), 100)
        c = ccdf[np.searchsorted(-abs_signal_sorted, -threshold)]
        axes[2].loglog(x_tail, c * (x_tail / threshold) ** (-hill_exp),
                      color=colors[3], linewidth=1.5, linestyle='--',
                      label=f'Power law (α={hill_exp:.2f})')
        axes[2].axvline(threshold, color='gray', linewidth=0.8, linestyle=':',
                       label='Threshold (90th pct)')
        axes[2].set_xlabel(f'|{xlabel}|')
        axes[2].set_ylabel('CCDF')
        axes[2].set_title(f'Log-log CCDF — {group}')
        axes[2].legend()

        plt.suptitle(f'Heavy-tail assessment — {group} stations', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/heavy_tail_{group.lower()}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: heavy_tail_{group.lower()}.pdf")

    return pd.DataFrame(results)