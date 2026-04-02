import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ==================================== Moment scaling - Acceleration ============================
# ===============================================================================================

def compute_moment_scaling_acc(df_acc, q_values, tau_values, normalized=True,
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
    df_acc : pd.DataFrame
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
 
    for file in df_acc['file'].unique():
        signal = df_acc[df_acc['file'] == file][col].values
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

def compute_moment_scaling_vel(df_acc, q_values, tau_values, normalized=True,
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
# ==================================== Increments distribution check ============================
# ===============================================================================================

def check_increments_distribution(df_increments, tau_values=None):
    """
    Analyzes the distribution of increments: counts how many are < 1 vs > 1.
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        DataFrame with columns [file, station, stream, tau, increment]
    tau_values : list of int, optional
        Specific tau values to analyze. If None, analyzes all tau values.
    
    Returns
    -------
    df_summary : pd.DataFrame
        Summary with columns [tau, total_increments, abs_less_than_1, 
        abs_greater_than_1, frac_less_than_1, frac_greater_than_1, 
        mean_abs_increment, median_abs_increment]
    """
    if tau_values is None:
        tau_values = sorted(df_increments['tau'].unique())
    
    results = []
    
    for tau in tau_values:
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        abs_increments = np.abs(increments)
        
        total = len(increments)
        less_than_1 = np.sum(abs_increments < 1)
        greater_than_1 = np.sum(abs_increments >= 1)
        
        results.append({
            'tau': tau,
            'total_increments': total,
            'abs_less_than_1': less_than_1,
            'abs_greater_than_1': greater_than_1,
            'frac_less_than_1': less_than_1 / total,
            'frac_greater_than_1': greater_than_1 / total,
            'mean_abs_increment': np.mean(abs_increments),
            'median_abs_increment': np.median(abs_increments),
            'std_abs_increment': np.std(abs_increments),
            'min_abs_increment': np.min(abs_increments),
            'max_abs_increment': np.max(abs_increments)
        })
    
    df_summary = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"INCREMENTS DISTRIBUTION SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal number of tau values analyzed: {len(tau_values)}")
    print(f"\nOverall statistics:")
    print(f"  - Average fraction |increment| < 1: {df_summary['frac_less_than_1'].mean():.2%}")
    print(f"  - Average fraction |increment| >= 1: {df_summary['frac_greater_than_1'].mean():.2%}")
    print(f"\nBy tau scale:")
    print(df_summary.to_string(index=False))
    print(f"{'='*70}\n")
    
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
