"""
window_detection.py
-------------------
Functions for identifying temporal windows in seismic signals to separate
different dynamical regimes (pre-arrival, P-wave, S-wave, coda).

The module provides four complementary approaches:
    1. Visual inspection - Interactive plotting for manual identification
    2. PGA-based - Heuristic windows relative to Peak Ground Acceleration
    3. STA/LTA - Automatic onset detection using Short-Term/Long-Term Average
    4. Combined wrapper - Applies all methods and generates comparison report

Main differences between approaches:
    - Visual: Manual but most accurate for complex signals
    - PGA-based: Fast, works well for clear events
    - STA/LTA: Industry standard, robust for noisy data
    - Combined: Comprehensive analysis with multiple estimates

Usage:
    from src.window_detection import identify_windows_combined
    
    # Single file analysis
    windows = identify_windows_combined(signal, file_name='IT.ACC.00.HNE.D.ASC')
    
    # All files analysis
    df_windows = identify_windows_all_files(df_acc)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ================================ 1. Visual Inspection =========================================
# ===============================================================================================

def plot_signal_for_visual_inspection(signal, file_name='', sampling_rate=200, 
                                       output_dir=None):
    """
    Create diagnostic plots for manual window identification.
    
    Generates a 3-panel figure showing:
    1. Raw acceleration signal
    2. Energy (|a|²) in log scale
    3. Energy envelope (smoothed)
    
    Parameters
    ----------
    signal : array
        Acceleration signal (cm/s²)
    file_name : str
        File identifier for plot title
    sampling_rate : int
        Sampling frequency (Hz)
    output_dir : str, optional
        Directory to save figure
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes for interactive inspection
    
    Examples
    --------
    >>> signal = df[df['file'] == file]['acceleration'].values
    >>> fig, axes = plot_signal_for_visual_inspection(signal, file_name=file)
    >>> plt.show()
    >>> # Manually identify boundaries from plot
    >>> t0_pre = 0
    >>> t0_p = 14000  # Read from plot
    >>> t0_s = 22000  # Read from plot
    >>> t0_coda = 44000  # Read from plot
    """
    time = np.arange(len(signal)) / sampling_rate
    energy = signal ** 2
    envelope = uniform_filter1d(energy, size=int(sampling_rate))  # 1s window
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # 1. Acceleration
    axes[0].plot(time, signal, linewidth=0.5, color='black', alpha=0.7)
    axes[0].set_ylabel('Acceleration (cm/s²)', fontsize=12)
    axes[0].set_title(f'Signal: {file_name}', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='red', linewidth=0.5, alpha=0.3)
    
    # 2. Energy (log scale)
    axes[1].plot(time, energy, linewidth=0.5, color='red', alpha=0.7)
    axes[1].set_ylabel('Energy (|a|²)', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)
    
    # 3. Envelope (smoothed energy)
    axes[2].plot(time, envelope, linewidth=1.5, color='orange')
    axes[2].set_ylabel('Energy Envelope', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        station = file_name.split('.')[1] if '.' in file_name else 'unknown'
        stream = file_name.split('.')[3] if len(file_name.split('.')) > 3 else 'unknown'
        filepath = os.path.join(output_dir, f'visual_inspection_{station}_{stream}.pdf')
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    return fig, axes

# ===============================================================================================
# ================================ 2. PGA-Based Detection =======================================
# ===============================================================================================

def identify_windows_pga_based(signal, sampling_rate=200, 
                               pre_p_duration=50.0, p_duration=40.0, 
                               s_duration=110.0):
    """
    Identify temporal windows based on Peak Ground Acceleration (PGA).
    
    Uses heuristic time windows relative to PGA:
    - Pre-arrival: from start to (PGA - pre_p_duration)
    - P-wave: from (PGA - pre_p_duration) to (PGA - 10s)
    - S-wave: from (PGA - 10s) to (PGA + s_duration)
    - Coda: from (PGA + s_duration) to end
    
    Parameters
    ----------
    signal : array
        Acceleration signal (cm/s²)
    sampling_rate : int
        Sampling frequency (Hz)
    pre_p_duration : float
        Time before PGA to start P-wave window (seconds)
    p_duration : float
        Approximate P-wave duration (seconds)
    s_duration : float
        Duration after PGA for S-wave window (seconds)
    
    Returns
    -------
    dict
        Window boundaries: {window_name: {'start': idx, 'end': idx}}
    
    Examples
    --------
    >>> windows = identify_windows_pga_based(signal)
    >>> print(windows['p_wave'])  # {'start': 14567, 'end': 22567}
    """
    # Find PGA
    pga_idx = np.argmax(np.abs(signal))
    pga_time = pga_idx / sampling_rate
    
    # Convert durations to samples
    pre_p_samples = int(pre_p_duration * sampling_rate)
    p_end_offset = int(10 * sampling_rate)  # 10s before PGA
    s_samples = int(s_duration * sampling_rate)
    
    # Define windows
    windows = {
        'pre_arrival': {
            'start': 0,
            'end': max(0, pga_idx - pre_p_samples)
        },
        'p_wave': {
            'start': max(0, pga_idx - pre_p_samples),
            'end': max(0, pga_idx - p_end_offset)
        },
        's_wave': {
            'start': max(0, pga_idx - p_end_offset),
            'end': min(len(signal), pga_idx + s_samples)
        },
        'coda': {
            'start': min(len(signal), pga_idx + s_samples),
            'end': len(signal)
        }
    }
    
    # Print summary
    for name, bounds in windows.items():
        start_time = bounds['start'] / sampling_rate
        end_time = bounds['end'] / sampling_rate
        duration = (bounds['end'] - bounds['start']) / sampling_rate
        n_samples = bounds['end'] - bounds['start']
    return windows

# ===============================================================================================
# ================================ 3. STA/LTA Detection =========================================
# ===============================================================================================

def compute_sta_lta(signal, sta_window=200, lta_window=4000):
    """
    Compute STA/LTA (Short-Term Average / Long-Term Average) ratio.
    
    Standard seismological method for onset detection.
    
    Parameters
    ----------
    signal : array
        Acceleration signal
    sta_window : int
        Short-term average window (samples)
        Default: 200 samples = 1s at 200 Hz
    lta_window : int
        Long-term average window (samples)
        Default: 4000 samples = 20s at 200 Hz
    
    Returns
    -------
    array
        STA/LTA ratio time series
    """
    # Energy signal
    energy = signal ** 2
    
    # STA: short-term average
    sta = uniform_filter1d(energy, size=sta_window, mode='constant')
    
    # LTA: long-term average
    lta = uniform_filter1d(energy, size=lta_window, mode='constant')
    
    # STA/LTA ratio (avoid division by zero)
    ratio = sta / (lta + 1e-10)
    
    return ratio


def detect_onset_sta_lta(signal, threshold=3.0, sta_window=200, lta_window=4000):
    """
    Detect P-wave onset using STA/LTA picker.
    
    Parameters
    ----------
    signal : array
        Acceleration signal
    threshold : float
        STA/LTA threshold for onset detection
        Typical values: 2-5 (higher = more conservative)
    sta_window : int
        Short-term average window (samples)
    lta_window : int
        Long-term average window (samples)
    
    Returns
    -------
    int
        P-wave onset sample index
    array
        STA/LTA ratio for diagnostic purposes
    """
    ratio = compute_sta_lta(signal, sta_window, lta_window)
    
    # P-wave onset: first time ratio exceeds threshold
    onset_candidates = np.where(ratio > threshold)[0]
    
    if len(onset_candidates) > 0:
        p_onset = onset_candidates[0]
    else:
        p_onset = 0
        print(f"  Warning: No P-wave onset detected with threshold={threshold}")
        print(f"  Max STA/LTA ratio: {ratio.max():.2f}")
    
    return p_onset, ratio

def identify_windows_sta_lta(signal, sampling_rate=200, threshold=3.0,
                             s_duration=110.0):
    """
    Identify temporal windows using STA/LTA onset detection.
    
    Combines STA/LTA for P-onset with PGA for S-wave identification.
    
    Parameters
    ----------
    signal : array
        Acceleration signal
    sampling_rate : int
        Sampling frequency (Hz)
    threshold : float
        STA/LTA threshold for P-onset
    s_duration : float
        Duration after PGA for S-wave window (seconds)
    
    Returns
    -------
    dict
        Window boundaries
    dict
        Diagnostic info: {'p_onset': idx, 'pga_idx': idx, 'sta_lta_ratio': array}
    """
    # Detect P-wave onset
    p_onset, sta_lta_ratio = detect_onset_sta_lta(signal, threshold=threshold)
    
    # Find PGA (assume S-wave arrival)
    pga_idx = np.argmax(np.abs(signal))
    
    # S-wave duration in samples
    s_samples = int(s_duration * sampling_rate)
    
    # Define windows
    windows = {
        'pre_arrival': {
            'start': 0,
            'end': p_onset
        },
        'p_wave': {
            'start': p_onset,
            'end': pga_idx
        },
        's_wave': {
            'start': pga_idx,
            'end': min(len(signal), pga_idx + s_samples)
        },
        'coda': {
            'start': min(len(signal), pga_idx + s_samples),
            'end': len(signal)
        }
    }
    
    # Diagnostic info
    diagnostics = {
        'p_onset': p_onset,
        'pga_idx': pga_idx,
        'sta_lta_ratio': sta_lta_ratio,
        'sta_lta_max': sta_lta_ratio.max(),
        'threshold': threshold
    }
    
    # Print summary
    print(f"\nSTA/LTA detection:")
    print(f"  P-onset at sample {p_onset} ({p_onset/sampling_rate:.1f}s)")
    print(f"  PGA at sample {pga_idx} ({pga_idx/sampling_rate:.1f}s)")
    print(f"  Max STA/LTA ratio: {sta_lta_ratio.max():.2f} (threshold: {threshold})")
    print(f"\n  Window boundaries:")
    for name, bounds in windows.items():
        start_time = bounds['start'] / sampling_rate
        end_time = bounds['end'] / sampling_rate
        duration = (bounds['end'] - bounds['start']) / sampling_rate
        n_samples = bounds['end'] - bounds['start']
        print(f"    {name:12s}: [{bounds['start']:6d}, {bounds['end']:6d}] = "
              f"[{start_time:6.1f}s, {end_time:6.1f}s] "
              f"(duration: {duration:5.1f}s, {n_samples:6d} samples)")
    
    return windows, diagnostics


# ===============================================================================================
# ================================ 4. Combined Analysis =========================================
# ===============================================================================================

def identify_windows_combined(signal, file_name='', sampling_rate=200, 
                              sta_lta_threshold=3.0, output_dir=None):
    """
    Apply all window detection methods and generate comparison report.
    
    Runs PGA-based and STA/LTA methods, creates visualization comparing results.
    
    Parameters
    ----------
    signal : array
        Acceleration signal
    file_name : str
        File identifier for labeling
    sampling_rate : int
        Sampling frequency (Hz)
    sta_lta_threshold : float
        STA/LTA threshold
    output_dir : str, optional
        Directory to save diagnostic plots
    
    Returns
    -------
    dict
        Recommended windows (from STA/LTA as primary method)
    dict
        All results: {'pga': windows, 'sta_lta': windows, 'diagnostics': ...}
    """
    print("="*70)
    print(f"Window Detection: {file_name}")
    print("="*70)
    
    # Method 1: PGA-based
    windows_pga = identify_windows_pga_based(signal, sampling_rate)
    
    # Method 2: STA/LTA
    windows_sta_lta, diagnostics = identify_windows_sta_lta(
        signal, sampling_rate, threshold=sta_lta_threshold
    )
    
    # Visualization
    if output_dir or True:  # Always create figure for inspection
        time = np.arange(len(signal)) / sampling_rate
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Subplot 1: Signal with PGA windows
        axes[0].plot(time, signal, linewidth=0.5, color='black', alpha=0.7)
        axes[0].set_ylabel('Acceleration (cm/s²)', fontsize=11)
        axes[0].set_title(f'PGA-Based Windows: {file_name}', fontsize=12)
        axes[0].grid(alpha=0.3)
        
        window_colors = {'pre_arrival': colors[0], 'p_wave': colors[1],
                        's_wave': colors[2], 'coda': colors[3]}
        
        for name, bounds in windows_pga.items():
            start_time = bounds['start'] / sampling_rate
            end_time = bounds['end'] / sampling_rate
            axes[0].axvspan(start_time, end_time, alpha=0.15,
                           color=window_colors[name], label=name)
        axes[0].legend(loc='upper right', fontsize=9)
        
        # Subplot 2: Signal with STA/LTA windows
        axes[1].plot(time, signal, linewidth=0.5, color='black', alpha=0.7)
        axes[1].set_ylabel('Acceleration (cm/s²)', fontsize=11)
        axes[1].set_title(f'STA/LTA Windows: {file_name}', fontsize=12)
        axes[1].grid(alpha=0.3)
        
        for name, bounds in windows_sta_lta.items():
            start_time = bounds['start'] / sampling_rate
            end_time = bounds['end'] / sampling_rate
            axes[1].axvspan(start_time, end_time, alpha=0.15,
                           color=window_colors[name])
        
        # Mark P-onset and PGA
        p_time = diagnostics['p_onset'] / sampling_rate
        pga_time = diagnostics['pga_idx'] / sampling_rate
        axes[1].axvline(p_time, color='blue', linestyle='--', 
                       linewidth=2, label=f'P-onset ({p_time:.1f}s)')
        axes[1].axvline(pga_time, color='red', linestyle='--',
                       linewidth=2, label=f'PGA ({pga_time:.1f}s)')
        axes[1].legend(loc='upper right', fontsize=9)
        
        # Subplot 3: Energy envelope
        energy = signal ** 2
        envelope = uniform_filter1d(energy, size=int(sampling_rate))
        axes[2].plot(time, envelope, linewidth=1.5, color='orange')
        axes[2].set_ylabel('Energy Envelope', fontsize=11)
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)
        
        # Subplot 4: STA/LTA ratio
        axes[3].plot(time, diagnostics['sta_lta_ratio'], linewidth=1, color='purple')
        axes[3].axhline(sta_lta_threshold, color='red', linestyle='--',
                       linewidth=2, label=f'Threshold ({sta_lta_threshold})')
        axes[3].axvline(p_time, color='blue', linestyle='--', linewidth=1)
        axes[3].set_ylabel('STA/LTA Ratio', fontsize=11)
        axes[3].set_xlabel('Time (s)', fontsize=11)
        axes[3].legend(loc='upper right', fontsize=9)
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            station = file_name.split('.')[1] if '.' in file_name else 'unknown'
            stream = file_name.split('.')[3] if len(file_name.split('.')) > 3 else 'unknown'
            filepath = os.path.join(output_dir, 
                                   f'window_detection_{station}_{stream}.pdf')
            plt.savefig(filepath, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
            plt.close()
        else:
            plt.show()
    
    # Compile results
    results = {
        'pga': windows_pga,
        'sta_lta': windows_sta_lta,
        'diagnostics': diagnostics,
        'file_name': file_name
    }
    
    # Return STA/LTA as recommended (more robust)
    print("\n" + "="*70)
    print("Recommended windows: STA/LTA method")
    print("="*70)
    
    return windows_sta_lta, results


# ===============================================================================================
# ================================ 5. Batch Processing ==========================================
# ===============================================================================================

def identify_windows_all_files(df, sampling_rate=200, sta_lta_threshold=3.0,
                               output_dir=None, max_files=None):
    """
    Apply window detection to all files in dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Acceleration data with columns ['file', 'acceleration']
    sampling_rate : int
        Sampling frequency (Hz)
    sta_lta_threshold : float
        STA/LTA threshold
    output_dir : str, optional
        Directory to save individual diagnostic plots
    max_files : int, optional
        Limit number of files to process (for testing)
    
    Returns
    -------
    pd.DataFrame
        Window boundaries for all files
        Columns: [file, window_name, start, end, start_time_s, end_time_s, 
                  duration_s, n_samples, method]
    """
    all_windows = []
    
    files = df['file'].unique()
    if max_files:
        files = files[:max_files]
    
    print(f"\nProcessing {len(files)} files...")
    
    for i, file in enumerate(files):
        signal = df[df['file'] == file]['acceleration'].values
        
        print(f"\n[{i+1}/{len(files)}] {file}")
        
        # Detect windows
        windows, results = identify_windows_combined(
            signal, file_name=file, sampling_rate=sampling_rate,
            sta_lta_threshold=sta_lta_threshold, output_dir=output_dir
        )
        
        # Store results
        for window_name, bounds in windows.items():
            all_windows.append({
                'file': file,
                'station': file.split('.')[1],
                'stream': file.split('.')[3],
                'window_name': window_name,
                'start': bounds['start'],
                'end': bounds['end'],
                'start_time_s': bounds['start'] / sampling_rate,
                'end_time_s': bounds['end'] / sampling_rate,
                'duration_s': (bounds['end'] - bounds['start']) / sampling_rate,
                'n_samples': bounds['end'] - bounds['start'],
                'method': 'sta_lta'
            })
    
    df_windows = pd.DataFrame(all_windows)
    
    print(f"\n{'='*70}")
    print(f"Completed: {len(files)} files processed")
    print(f"Total windows detected: {len(df_windows)}")
    print(f"{'='*70}")
    
    return df_windows
