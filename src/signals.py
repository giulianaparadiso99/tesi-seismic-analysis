import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.plot_settings import set_plot_style
colors = set_plot_style()

def plot_empirical_pdfs(df_acc_clean, bins=100, log_scale=False, normalized=True, output_dir='../figures/pdf_single'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    col = 'acceleration_normalized' if normalized else 'acceleration'
    saved = []
    failed = []
    
    for file in df_acc_clean['file'].unique():
        signal = df_acc_clean[df_acc_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        filepath = f'{output_dir}/pdf_{station}_{stream}.pdf'
        
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(signal, bins=bins, color=colors[0], edgecolor='none', density=True)
            if log_scale:
                ax.set_yscale('log')
            ax.set_xlabel('Normalized acceleration' if normalized else 'Acceleration (cm/s²)')
            ax.set_ylabel('Probability density')
            ax.set_title(f'Empirical PDF — {station} {stream}')
            plt.tight_layout()
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            saved.append(filepath)
        except Exception as e:
            failed.append((filepath, str(e)))
            plt.close()
    
    # Check
    print(f"Saved: {len(saved)}/{df_acc_clean['file'].nunique()} PDFs")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All PDFs saved successfully!")

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
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(signal, bins=bins, color=colors[0], edgecolor='none',
                    density=True, alpha=0.7, label='Empirical PDF')
            x = np.linspace(signal.min(), signal.max(), 500)
            ax.plot(x, stats.norm.pdf(x, mu, std), color=colors[2],
                    linewidth=1.5, label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')
            if log_scale:
                ax.set_yscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Probability density')
            ax.set_title(f'Gaussian fit — {station} {stream}\n'
                        f'Kurt={kurt:.2f}, Skew={skew:.2f}, AD={ad_stat:.2f} (crit={ad_critical:.2f})')
            ax.legend()
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