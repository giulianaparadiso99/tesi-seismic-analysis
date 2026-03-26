import os
import numpy as np
import matplotlib.pyplot as plt
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ==================================== Empirical PDFs ===========================================
# ===============================================================================================

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

# ===============================================================================================
# ================================ Event onset diagnostic =======================================
# ===============================================================================================

def plot_onset_diagnostic(df_acc, df_onsets, n_examples=4, normalized=True,
                           output_dir='../figures/03_single_signal/scaling/event_window'):
    """
    Plots representative signals with the detected event onset marked.

    Parameters
    ----------
    df_acc : pd.DataFrame
        Full (untrimmed) preprocessed acceleration data.
    df_onsets : pd.DataFrame
        Output of trim_to_event_window — contains columns
        [file, station, stream, onset, samples_before, samples_after].
    n_examples : int
        Number of example signals to show (shown in a grid).
    normalized : bool
        Whether to use acceleration_normalized or acceleration.
    output_dir : str or Path
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    col = 'acceleration_normalized' if normalized else 'acceleration'
    ylabel = 'Normalized acceleration' if normalized else 'Acceleration (cm/s²)'

    example_files = df_acc['file'].unique()[:n_examples]
    ncols = 2
    nrows = int(np.ceil(n_examples / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = axes.flat if n_examples > 1 else [axes]

    for ax, file in zip(axes, example_files):
        station = file.split('.')[1]
        stream = file.split('.')[3]
        signal = df_acc[df_acc['file'] == file][col].values
        onset_row = df_onsets[df_onsets['file'] == file]

        if onset_row.empty:
            ax.set_visible(False)
            continue

        onset = onset_row['onset'].values[0]

        ax.plot(signal, color=colors[0], linewidth=0.5, alpha=0.8)
        ax.axvline(onset, color='crimson', linewidth=1.5,
                   linestyle='--', label=f'Onset: sample {onset}')
        ax.set_title(f'{station} {stream}')
        ax.set_xlabel('Sample')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    # Hide unused axes if n_examples is odd
    for ax in list(axes)[n_examples:]:
        ax.set_visible(False)

    plt.suptitle('Event onset detection — representative signals', fontsize=13)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'onset_diagnostic.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


# ===============================================================================================
# ============================= Event window length distribution ================================
# ===============================================================================================

def plot_onset_distribution(df_onsets,
                             output_dir='../figures/03_single_signal/scaling/event_window'):
    """
    Plots the distribution of onset indices and event window lengths.

    Parameters
    ----------
    df_onsets : pd.DataFrame
        Output of trim_to_event_window — contains columns
        [file, station, stream, onset, samples_before, samples_after].
    output_dir : str or Path
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df_onsets['onset'], bins=20,
                 color=colors[0], edgecolor='none')
    axes[0].set_xlabel('Onset sample')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of onset indices')

    axes[1].hist(df_onsets['samples_after'], bins=20,
                 color=colors[1], edgecolor='none')
    axes[1].set_xlabel('Samples after onset')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of event window length')

    plt.suptitle('Event window statistics', fontsize=13)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'onset_distribution.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    print(f"Mean onset:         {df_onsets['onset'].mean():.0f} samples")
    print(f"Mean window length: {df_onsets['samples_after'].mean():.0f} samples")
    print(f"Min window length:  {df_onsets['samples_after'].min():.0f} samples")

# ===============================================================================================
# ================================ Increment distributions ======================================
# ===============================================================================================

def plot_increment_distributions(df_increments, df_tail,
                                  tau_values_plot=None, top_fraction=0.1,
                                  output_dir='../figures/03_single_signal/scaling/displacement/event_window/increments'):
    """
    Plots the empirical CCDF of |Delta x(tau)| in log-log scale for each signal
    and for selected tau values, with tail exponent fits overlaid.
    Also produces three summary plots:
        1. Median CCDF across all signals for each tau.
        2. Tail exponents (Hill and fit) as a function of tau.
        3. Hill vs fit exponent scatter.

    Parameters
    ----------
    df_increments : pd.DataFrame
        Output of compute_moment_scaling_* with save_increments=True.
        Must contain columns [file, station, stream, tau, increment].
    df_tail : pd.DataFrame
        Output of compute_increment_tail_exponents.
        Must contain columns [file, station, stream, tau, hill_exp, fit_exp, r2_fit].
    tau_values_plot : list of int or None
        Subset of tau values to plot. If None, uses all tau values in df_increments.
    top_fraction : float
        Fraction of largest values used for the tail fit lines (default: 0.1).
    output_dir : str or Path
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    if tau_values_plot is None:
        tau_values_plot = sorted(df_increments['tau'].unique())

    df_increments = df_increments[df_increments['tau'].isin(tau_values_plot)]
    df_tail = df_tail[df_tail['tau'].isin(tau_values_plot)]

    tau_colors = plt.cm.viridis(np.linspace(0, 0.9, len(tau_values_plot)))
    files = df_increments['file'].unique()
    saved, failed = [], []

    # ------------------------------------------------------------------
    # Individual plots — one per signal
    # ------------------------------------------------------------------
    for file in files:
        station = df_increments[df_increments['file'] == file]['station'].iloc[0]
        stream = df_increments[df_increments['file'] == file]['stream'].iloc[0]

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            for ci, tau in enumerate(tau_values_plot):
                grp = df_increments[(df_increments['file'] == file) & (df_increments['tau'] == tau)]
                if grp.empty:
                    continue
                abs_inc = np.sort(np.abs(grp['increment'].values))[::-1]
                abs_inc = abs_inc[abs_inc > 0]
                n = len(abs_inc)
                ccdf_y = np.arange(1, n + 1) / n
                ax.loglog(abs_inc, ccdf_y, color=tau_colors[ci], linewidth=0.8,
                          alpha=0.7, label=f'τ={tau}')

                # Tail fit line — computed in log space to avoid overflow
                tail_row = df_tail[(df_tail['file'] == file) & (df_tail['tau'] == tau)]
                if not tail_row.empty:
                    fit_exp = tail_row['fit_exp'].values[0]
                    k = max(2, int(top_fraction * n))
                    threshold = abs_inc[k - 1]
                    x_fit = np.linspace(threshold, abs_inc[0], 50)
                    log_c = np.log(ccdf_y[k - 1]) + fit_exp * np.log(threshold)
                    y_fit = np.exp(log_c - fit_exp * np.log(x_fit))
                    ax.loglog(x_fit, y_fit, '--', color=tau_colors[ci], linewidth=1.2, alpha=0.9)

            ax.set_xlabel('|Δx(τ)|')
            ax.set_ylabel('P(|Δx| > x)')
            ax.set_title(f'Increment CCDF — {station} {stream}')
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'increments_{station}_{stream}.pdf'), bbox_inches='tight')
            plt.close()
            saved.append(file)
        except Exception as e:
            failed.append((file, str(e)))
            plt.close()

    print(f"Individual plots saved: {len(saved)}/{len(files)}")
    if failed:
        for f, e in failed:
            print(f"  Failed: {f} — {e}")

    # ------------------------------------------------------------------
    # Summary plot 1 — median CCDF across all signals for each tau
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ci, tau in enumerate(tau_values_plot):
            grp = df_increments[df_increments['tau'] == tau]
            all_abs = np.sort(np.abs(grp['increment'].values))[::-1]
            all_abs = all_abs[all_abs > 0]
            n = len(all_abs)
            ccdf_y = np.arange(1, n + 1) / n
            ax.loglog(all_abs, ccdf_y, color=tau_colors[ci], linewidth=1.0,
                      alpha=0.8, label=f'τ={tau}')
        ax.set_xlabel('|Δx(τ)|')
        ax.set_ylabel('P(|Δx| > x)')
        ax.set_title('Increment CCDF — all signals pooled by τ')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'increments_summary_ccdf.pdf'), bbox_inches='tight')
        plt.close()
        print("Summary CCDF saved.")
    except Exception as e:
        print(f"Error saving summary CCDF: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Summary plot 2 — tail exponents vs tau (Hill and fit)
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        tau_vals = sorted(df_tail['tau'].unique())

        hill_median = df_tail.groupby('tau')['hill_exp'].median()
        hill_q25 = df_tail.groupby('tau')['hill_exp'].quantile(0.25)
        hill_q75 = df_tail.groupby('tau')['hill_exp'].quantile(0.75)

        fit_median = df_tail.groupby('tau')['fit_exp'].median()
        fit_q25 = df_tail.groupby('tau')['fit_exp'].quantile(0.25)
        fit_q75 = df_tail.groupby('tau')['fit_exp'].quantile(0.75)

        ax.plot(tau_vals, hill_median, 'o-', color=colors[0], linewidth=1.5, label='Hill estimator')
        ax.fill_between(tau_vals, hill_q25, hill_q75, color=colors[0], alpha=0.2)
        ax.plot(tau_vals, fit_median, 's-', color=colors[1], linewidth=1.5, label='CCDF fit')
        ax.fill_between(tau_vals, fit_q25, fit_q75, color=colors[1], alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlabel('τ (samples)')
        ax.set_ylabel('Tail exponent')
        ax.set_title('Tail exponents vs τ — median ± IQR across all signals')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'increments_summary_exponents.pdf'), bbox_inches='tight')
        plt.close()
        print("Summary exponents saved.")
    except Exception as e:
        print(f"Error saving summary exponents: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Summary plot 3 — Hill vs fit exponent scatter
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df_tail['hill_exp'], df_tail['fit_exp'],
                        c=np.log10(df_tail['tau']), cmap='viridis',
                        s=20, alpha=0.6, edgecolors='none')
        plt.colorbar(sc, ax=ax, label='log₁₀(τ)')
        ax.axline((0, 0), slope=1, color='gray', linewidth=0.8,
                  linestyle='--', label='Hill = Fit')
        ax.set_xlabel('Hill exponent')
        ax.set_ylabel('CCDF fit exponent')
        ax.set_title('Hill vs CCDF fit exponent\n(colored by log τ)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'increments_summary_hill_vs_fit.pdf'), bbox_inches='tight')
        plt.close()
        print("Hill vs fit scatter saved.")
    except Exception as e:
        print(f"Error saving Hill vs fit scatter: {e}")
        plt.close()