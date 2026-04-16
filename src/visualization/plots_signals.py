import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ========================= Signals — signal length distribution ================================
# ===============================================================================================
 
def plot_signal_length_distribution(signal_lengths, output_dir=None):
    """
    Histogram of signal lengths (number of samples) across all files.
 
    Parameters
    ----------
    signal_lengths : pd.Series
        Series with one entry per file, values = number of samples.
        Typically: df_acc.groupby('file')['sample'].max() + 1
    output_dir : str or Path or None
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(signal_lengths, bins=15, color=colors[0], edgecolor='white', linewidth=0.5)
    ax.set_title('Signal length distribution')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Count')
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'signal_length_distribution.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ============================= Signals — example signals =======================================
# ===============================================================================================
 
def plot_three_components(df, output_dir='../figures/exploratory', 
                         max_stations=None, normalized=True):
    """
    Plot three-component accelerograms for each station.
    
    Creates one figure per station with the 3 available components stacked vertically.
    Handles different channel naming conventions:
    - HNE, HNN, HNZ (standard high-gain broadband)
    - HGE, HGN, HGZ (high-gain)
    - HN1, HN2, HNZ (alternative horizontal naming)
    - Other variations
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with acceleration data
        Must have columns: 'file', 'acceleration' or 'acceleration_normalized'
    output_dir : str
        Directory to save figures (default: '../figures/exploratory')
    max_stations : int, optional
        Maximum number of stations to plot (for testing)
    normalized : bool
        If True, use 'acceleration_normalized', else 'acceleration'
    
    Returns
    -------
    list of str
        List of saved figure paths
    
    Examples
    --------
    >>> # Plot all stations (raw acceleration)
    >>> saved_figs = plot_three_components(df, normalized=False)
    >>> 
    >>> # Plot first 5 stations (normalized)
    >>> saved_figs = plot_three_components(df, max_stations=5)
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract station names and streams from file names
    df['station'] = df['file'].apply(lambda x: x.split('.')[1])
    df['stream'] = df['file'].apply(lambda x: x.split('.')[3])
    
    stations = df['station'].unique()
    
    if max_stations is not None:
        stations = stations[:max_stations]
    
    # Choose acceleration column
    acc_col = 'acceleration_normalized' if normalized else 'acceleration'
    
    if acc_col not in df.columns:
        raise ValueError(f"Column '{acc_col}' not found in DataFrame")
    
    saved_figures = []
    
    print(f"Plotting {len(stations)} stations...")
    
    for station in stations:
        # Get data for this station (all available components)
        df_station = df[df['station'] == station].copy()
        
        # Get available streams for this station
        streams = sorted(df_station['stream'].unique())
        
        if len(streams) != 3:
            print(f"  Warning: Station {station} has {len(streams)} components "
                  f"({', '.join(streams)}), skipping")
            continue
        
        # Define sorting order based on channel type
        # Priority: vertical (Z) last, then try to order horizontals
        def get_stream_order(stream):
            """
            Assign order priority for plotting.
            Vertical component (Z) should be last.
            Horizontal components ordered alphabetically or by convention.
            """
            # Vertical component always last
            if stream.endswith('Z'):
                return 2
            # Try to order horizontals: E before N, or 1 before 2
            elif stream.endswith('E') or stream.endswith('1'):
                return 0
            elif stream.endswith('N') or stream.endswith('2'):
                return 1
            else:
                # Fallback: alphabetical
                return ord(stream[-1])
        
        df_station['stream_order'] = df_station['stream'].apply(get_stream_order)
        df_station = df_station.sort_values('stream_order')
        
        # Get ordered streams
        ordered_streams = df_station['stream'].unique()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        # Get sampling rate and create time array
        first_file = df_station['file'].iloc[0]
        n_samples = len(df_station[df_station['file'] == first_file])
        sampling_rate = 200  # Hz (from metadata)
        time = np.arange(n_samples) / sampling_rate
        
        # Plot each component
        for i, stream in enumerate(ordered_streams):
            df_component = df_station[df_station['stream'] == stream]
            
            if len(df_component) == 0:
                continue
            
            # Get signal
            signal = df_component[acc_col].values
            
            ax = axes[i]
            
            # Plot acceleration
            ax.plot(time, signal, 'k-', linewidth=0.5)
            
            # Labels and formatting
            # Add component orientation info in ylabel
            if stream.endswith('E') or stream.endswith('1'):
                orientation = 'E-W / 1'
            elif stream.endswith('N') or stream.endswith('2'):
                orientation = 'N-S / 2'
            elif stream.endswith('Z'):
                orientation = 'Vertical'
            else:
                orientation = ''
            
            ylabel = f'{stream}'
            if orientation:
                ylabel += f'\n({orientation})'
            ylabel += '\n'
            
            if normalized:
                ylabel += 'Normalized\nacceleration'
            else:
                ylabel += 'Acceleration\n(cm/s²)'
            
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Title only on first subplot
            if i == 0:
                title = f'Station {station} - Three-Component Accelerogram'
                if normalized:
                    title += ' (Normalized)'
                # Add info about channel naming
                channel_info = ', '.join(ordered_streams)
                title += f'\n[{channel_info}]'
                ax.set_title(title, fontsize=14, fontweight='bold')
        
        # X-axis label only on bottom subplot
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        fig_name = f'{station}_three_components'
        if normalized:
            fig_name += '_normalized'
        fig_path = os.path.join(output_dir, f'{fig_name}.png')
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        saved_figures.append(fig_path)
        
        # Print with channel info
        print(f"Saved: {fig_name}.png [{', '.join(ordered_streams)}]")
    
    print(f"\nTotal figures saved: {len(saved_figures)}")
    
    return saved_figures
 
 
# ===============================================================================================
# ==================== Signals — raw acceleration distributions =================================
# ===============================================================================================
 
def plot_acceleration_distributions(df_acc, df_meta_clean,
                                     streams=('HNE', 'HNN', 'HNZ'), output_dir=None):
    """
    Two plots: global acceleration distribution (log y-scale) and
    per-component overlay (log y-scale).
 
    Parameters
    ----------
    df_acc : pd.DataFrame
        Must contain columns [file, acceleration].
    df_meta_clean : pd.DataFrame
        Must contain columns [file, STREAM].
    streams : tuple of str
    output_dir : str or Path or None
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
 
    # Global distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_acc['acceleration'], bins=100, color=colors[1], edgecolor='none')
    ax.set_yscale('log')
    ax.set_title('Acceleration distribution (log scale)')
    ax.set_xlabel('Acceleration (cm/s²)')
    ax.set_ylabel('Count (log scale)')
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'acceleration_distribution.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()
 
    # Per-component overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, stream in enumerate(streams):
        files = df_meta_clean[df_meta_clean['STREAM'] == stream]['file'].values
        acc_values = df_acc[df_acc['file'].isin(files)]['acceleration'].values
        ax.hist(acc_values, bins=100, color=colors[i], alpha=0.6,
                label=stream, edgecolor='none')
    ax.set_yscale('log')
    ax.set_title('Acceleration distribution by component (log scale)')
    ax.set_xlabel('Acceleration (cm/s²)')
    ax.set_ylabel('Count (log scale)')
    ax.legend(title='Component')
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'acceleration_by_component.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ================= Signals — post-preprocessing check - PDF analysis pipeline ==================
# ===============================================================================================
 
def plot_postcheck_pdf(df_acc_raw, df_acc_clean, output_dir=None):
    """
    2x2 summary figure for the single signal preprocessing pipeline:
    residual means, std distribution, baseline correction example,
    normalized signal example.
 
    Parameters
    ----------
    df_acc_raw : pd.DataFrame
        Raw accelerations before preprocessing. Must contain [file, acceleration].
    df_acc_clean : pd.DataFrame
        Preprocessed accelerations. Must contain
        [file, acceleration, acceleration_normalized].
    output_dir : str or Path or None
    """
    baseline_check = df_acc_clean.groupby('file')['acceleration'].mean()
    norm_check     = df_acc_clean.groupby('file')['acceleration_normalized'].std()
 
    example_file  = df_acc_clean['file'].unique()[0]
    example_raw   = df_acc_raw[df_acc_raw['file'] == example_file]['acceleration'].values
    example_bc    = df_acc_clean[df_acc_clean['file'] == example_file]['acceleration'].values
    example_norm  = df_acc_clean[df_acc_clean['file'] == example_file]['acceleration_normalized'].values
    station_label = example_file.split('.')[1] + ' ' + example_file.split('.')[3]
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = np.arange(len(example_raw))
 
    # Residual means
    axes[0, 0].hist(baseline_check.values, bins=30, color=colors[0], edgecolor='none')
    axes[0, 0].axvline(0, color='black', linewidth=1, linestyle='--', label='Expected: 0')
    axes[0, 0].set_title('Residual mean per signal\n(after baseline correction)')
    axes[0, 0].set_xlabel('Mean (cm/s²)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
 
    # Std distribution
    axes[0, 1].hist(norm_check.values, bins=30, color=colors[1], edgecolor='none')
    axes[0, 1].axvline(1, color='black', linewidth=1, linestyle='--', label='Expected: 1')
    axes[0, 1].set_title('Standard deviation per signal\n(after normalization)')
    axes[0, 1].set_xlabel('Std')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
 
    # Baseline correction example
    axes[1, 0].plot(t, example_raw, color=colors[2], linewidth=0.5,
                    alpha=0.7, label='Raw')
    axes[1, 0].plot(t, example_bc,  color=colors[0], linewidth=0.5,
                    alpha=0.9, label='Baseline-corrected')
    axes[1, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1, 0].set_title(f'Baseline correction — {station_label}')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Acceleration (cm/s²)')
    axes[1, 0].legend(fontsize=9)
 
    # Normalized signal example
    axes[1, 1].plot(t, example_norm, color=colors[1], linewidth=0.5,
                    alpha=0.8, label='Normalized signal')
    axes[1, 1].axhline( 0, color='black',   linewidth=1,   linestyle='--', label='Mean = 0')
    axes[1, 1].axhline( 1, color=colors[3], linewidth=0.8, linestyle=':',  label='±1 std')
    axes[1, 1].axhline(-1, color=colors[3], linewidth=0.8, linestyle=':')
    axes[1, 1].set_title(f'Normalized signal — {station_label}')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Normalized acceleration')
    axes[1, 1].legend(fontsize=9)
 
    plt.suptitle('Post-preprocessing check — single signal pipeline', fontsize=14)
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'postcheck_single.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()

 
# ===============================================================================================
# ================ Signals — post-preprocessing check (moment scaling pipeline) =================
# ===============================================================================================
 
def plot_postcheck_moment_scaling(df_acc_raw, df_acc_long, threshold=48000, output_dir=None):
    """
    Post-preprocessing check plots for the long signals pipeline.
    
    Parameters
    ----------
    df_acc_raw : pd.DataFrame
        Raw acceleration data (before filtering)
    df_acc_long : pd.DataFrame
        Preprocessed long signals (after filtering, baseline correction, NO normalization)
    threshold : int
        Minimum samples threshold used for filtering
    output_dir : str or Path
        Directory to save the figure
    """
    from pathlib import Path  # Se non già importato all'inizio del file
    import numpy as np  # Se non già importato
    
    signal_lengths_raw = df_acc_raw.groupby('file')['sample'].max() + 1
    signal_lengths_long = df_acc_long.groupby('file')['sample'].max() + 1
    baseline_check_agg = df_acc_long.groupby('file')['acceleration'].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Signal lengths before and after filtering
    axes[0].hist(signal_lengths_raw, bins=20, alpha=0.7, 
                 label=f'Before (n={len(signal_lengths_raw)})', color=colors[0])
    axes[0].hist(signal_lengths_long, bins=20, alpha=0.7,
                 label=f'After (n={len(signal_lengths_long)})', color=colors[1])
    axes[0].axvline(threshold, color='gray', linestyle='--', linewidth=1,
                   label=f'Threshold: {threshold:,}')
    axes[0].set_xlabel('Number of samples')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Signal lengths before and after filtering')
    axes[0].legend()
    
    # Plot 2: Residual mean per signal (baseline correction check)
    axes[1].hist(baseline_check_agg, bins=10, color=colors[2], edgecolor='none')
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8, label='Expected: 0')
    axes[1].set_xlabel('Mean (cm/s²)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual mean per signal\n(after baseline correction)\nExpected: 0')
    axes[1].legend()
    axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # Plot 3: Physical units preserved (show std distribution)
    std_check = df_acc_long.groupby('file')['acceleration'].std()
    axes[2].hist(std_check, bins=15, color=colors[3], edgecolor='none')
    axes[2].set_xlabel('Std (cm/s²)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Standard deviation per signal\n(physical units preserved)\nNOT normalized')
    
    plt.suptitle('Post-preprocessing check — moment scaling pipeline', fontsize=13)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / '02_postcheck_moment_scaling.pdf', bbox_inches='tight')
    
    plt.show()

# ===============================================================================================
# ==================================== Empirical PDFs ===========================================
# ===============================================================================================

def plot_empirical_pdfs(df_acc_clean, bins=100, log_scale=False, normalized=True, output_dir='../figures/03_single_signal/03a_pdf_analysis/pdf_single'):
    
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

def plot_onset_diagnostic(df_acc, df_onsets, n_examples=4, normalized=False,
                           output_dir='../figures/03_single_signal/03b_moment_scaling/event_window'):
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
                             output_dir='../figures/03_single_signal/03b_moment_scaling/event_window'):
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
