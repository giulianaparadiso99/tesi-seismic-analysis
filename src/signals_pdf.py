"""
signals_pdf.py
--------------
Probability density function (PDF) analysis for seismic acceleration signals.
This module provides functions to assess the statistical distribution of
seismic increments, test for Gaussianity, and characterize heavy-tailed
behavior through various distribution fits and statistical tests.

The module is organized into two main analysis pipelines:

    1. Gaussian fit analysis          — test whether increment distributions
                                         follow a Gaussian (normal) distribution
       • Anderson-Darling test for normality
       • Kurtosis and skewness computation
       • Visual comparison with fitted Gaussian
       • Summary statistics across all signals

    2. Heavy tail analysis            — characterize non-Gaussian tails and
                                         identify power-law behavior
       • Comparison with Gaussian, Laplace, Student-t, Lévy stable distributions
       • Kolmogorov-Smirnov goodness-of-fit tests
       • Tail exponent estimation
       • Visual Q-Q plots and distribution overlays
       • Summary statistics for tail classification

Both pipelines process individual signals (one plot per station/channel) and
generate summary visualizations aggregating results across all signals. Figures
are saved as PDF files in the specified output directory, with automatic
directory creation if needed.

The analysis supports both normalized (standardized) and raw acceleration data,
with configurable binning, log-scale plotting, and statistical significance
thresholds.

Usage:
    from src.signals_pdf import gaussian_fit_analysis, heavy_tail_analysis
    
    # Example: Gaussian fit analysis
    df_results = gaussian_fit_analysis(
        df_acc_clean,
        bins=100,
        normalized=True,
        output_dir='../figures/03_single_signal/03a_pdf_analysis/gaussian_fit'
    )
    
    # Example: Heavy tail analysis
    df_tail = heavy_tail_analysis(
        df_acc_clean,
        normalized=True,
        output_dir='../figures/03_single_signal/03a_pdf_analysis/heavy_tail'
    )
"""

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
                          output_dir='../figures/03_single_signal/03a_pdf_analysis/gaussian_fit'):
    
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

def heavy_tail_assessment(df_acc_clean, normalized=True, output_dir='../figures/03_single_signal/03a_pdf_analysis/heavy_tail',
                          resume=False, partial_path='../data/processed/03a_pdf_analysis/heavy_tail_results_partial.parquet'):
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
