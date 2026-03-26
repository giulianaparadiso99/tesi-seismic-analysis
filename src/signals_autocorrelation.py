import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.plot_settings import set_plot_style
colors = set_plot_style()

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