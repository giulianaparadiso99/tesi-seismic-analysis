import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src import set_plot_style
colors, colors1 = set_plot_style()

# ===============================================================================================
# ========================= Signals — signal length distribution ================================
# ===============================================================================================
 
def plot_signal_length_distribution(signal_lengths, output_dir=None, prefix=''):
    """
    Histogram of signal lengths (number of samples) across all files.
    
    Parameters
    ----------
    signal_lengths : pd.Series
        Series with one entry per file, values = number of samples.
        Typically: df_signals.groupby('file')['sample'].max() + 1
    output_dir : str or Path or None
    prefix : str
        Prefix for output filename (e.g., 'acc', 'vel', 'dis').
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(signal_lengths, bins=15, color=colors[0], edgecolor='white', linewidth=0.5)
    ax.set_title('Signal length distribution')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Count')
    plt.tight_layout()
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'signal_length_distribution_{prefix}.pdf' if prefix else 'signal_length_distribution.pdf'
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ============================= Signals — example signals =======================================
# ===============================================================================================
 
def plot_station_waveforms(df, signal_column='acceleration', signal_unit='cm/s²',
                          output_dir='../figures/exploratory', 
                          max_stations=None, normalized=True, prefix=''):
    """
    Plot multi-component waveforms for each station.
    
    Creates one figure per station with the 3 available components stacked vertically.
    Handles different channel naming conventions:
    - HNE, HNN, HNZ (standard high-gain broadband)
    - HGE, HGN, HGZ (high-gain)
    - HN1, HN2, HNZ (alternative horizontal naming)
    - Other variations
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with signal data
        Must have columns: 'file', signal_column or f'{signal_column}_normalized'
    signal_column : str
        Name of the signal column (e.g., 'acceleration', 'velocity', 'displacement')
    signal_unit : str
        Unit label (e.g., 'cm/s²', 'cm/s', 'cm')
    output_dir : str
        Directory to save figures
    max_stations : int, optional
        Maximum number of stations to plot (for testing)
    normalized : bool
        If True, use '{signal_column}_normalized', else signal_column
    prefix : str
        Prefix for output filenames (e.g., 'acc', 'vel', 'dis')
    
    Returns
    -------
    list of str
        List of saved figure paths
    
    Examples
    --------
    >>> # Plot all stations (raw signal)
    >>> saved_figs = plot_station_waveforms(df, signal_column='velocity', 
    ...                                      signal_unit='cm/s', normalized=False)
    >>> 
    >>> # Plot first 5 stations (normalized)
    >>> saved_figs = plot_station_waveforms(df, max_stations=5)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract station names and streams from file names
    df['station'] = df['file'].apply(lambda x: x.split('.')[1])
    df['stream'] = df['file'].apply(lambda x: x.split('.')[3])
    
    stations = df['station'].unique()
    
    if max_stations is not None:
        stations = stations[:max_stations]
    
    # Choose signal column
    signal_col = f'{signal_column}_normalized' if normalized else signal_column
    
    if signal_col not in df.columns:
        raise ValueError(f"Column '{signal_col}' not found in DataFrame")
    
    saved_figures = []
    
    signal_name = signal_column.capitalize()
    
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
        def get_stream_order(stream):
            """
            Assign order priority for plotting.
            Vertical component (Z) should be last.
            Horizontal components ordered alphabetically or by convention.
            """
            if stream.endswith('Z'):
                return 2
            elif stream.endswith('E') or stream.endswith('1'):
                return 0
            elif stream.endswith('N') or stream.endswith('2'):
                return 1
            else:
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
            signal = df_component[signal_col].values
            
            ax = axes[i]
            
            # Plot signal
            ax.plot(time, signal, 'k-', linewidth=0.5)
            
            # Labels and formatting
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
                ylabel += f'Normalized\n{signal_column}'
            else:
                ylabel += f'{signal_name}\n({signal_unit})'
            
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Title only on first subplot
            if i == 0:
                title = f'Station {station} - Three-Component Waveforms'
                if normalized:
                    title += ' (Normalized)'
                channel_info = ', '.join(ordered_streams)
                title += f'\n[{channel_info}]'
                ax.set_title(title, fontsize=14, fontweight='bold')
        
        # X-axis label only on bottom subplot
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        fig_name = f'{station}_waveforms'
        if prefix:
            fig_name = f'{prefix}_{fig_name}'
        if normalized:
            fig_name += '_normalized'
        fig_path = os.path.join(output_dir, f'{fig_name}.png')
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        saved_figures.append(fig_path)
        
        print(f"Saved: {fig_name}.png [{', '.join(ordered_streams)}]")
    
    print(f"\nTotal figures saved: {len(saved_figures)}")
    
    return saved_figures
 
 
# ===============================================================================================
# ==================== Signals — raw acceleration distributions =================================
# ===============================================================================================
 
def plot_signals_distributions(df_signals, df_meta_clean,
                               signal_column='acceleration',
                               signal_unit='cm/s²',
                               streams=('HNE', 'HNN', 'HNZ'),
                               output_dir=None,
                               prefix=''):
    """
    Two plots: global signal distribution (log y-scale) and
    per-component overlay (log y-scale).
    
    Parameters
    ----------
    df_signals : pd.DataFrame
        Must contain columns [file, signal_column].
    df_meta_clean : pd.DataFrame
        Must contain columns [file, STREAM].
    signal_column : str
        Name of the signal column (e.g., 'acceleration', 'velocity', 'displacement').
    signal_unit : str
        Unit label for x-axis (e.g., 'cm/s²', 'cm/s', 'cm').
    streams : tuple of str
    output_dir : str or Path or None
    prefix : str
        Prefix for output filenames (e.g., 'acc', 'vel', 'dis').
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    signal_name = signal_column.capitalize()
    
    # Global distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_signals[signal_column], bins=100, color=colors[1], edgecolor='none')
    ax.set_yscale('log')
    ax.set_title(f'{signal_name} distribution (log scale)')
    ax.set_xlabel(f'{signal_name} ({signal_unit})')
    ax.set_ylabel('Count (log scale)')
    plt.tight_layout()
    
    if output_dir is not None:
        filename = f'signal_distribution_{prefix}.pdf' if prefix else 'signal_distribution.pdf'
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.show()
    plt.close()
    
    # Per-component overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, stream in enumerate(streams):
        files = df_meta_clean[df_meta_clean['STREAM'] == stream]['file'].values
        signal_values = df_signals[df_signals['file'].isin(files)][signal_column].values
        ax.hist(signal_values, bins=100, color=colors[i], alpha=0.6,
                label=stream, edgecolor='none')
    
    ax.set_yscale('log')
    ax.set_title(f'{signal_name} distribution by component (log scale)')
    ax.set_xlabel(f'{signal_name} ({signal_unit})')
    ax.set_ylabel('Count (log scale)')
    ax.legend(title='Component')
    plt.tight_layout()
    
    if output_dir is not None:
        filename = f'signal_by_component_{prefix}.pdf' if prefix else 'signal_by_component.pdf'
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.show()
    plt.close()
 
# ===============================================================================================
# ================= Signals — post-preprocessing check - PDF analysis pipeline ==================
# ===============================================================================================
 
def plot_postcheck_pdf(df_raw, df_clean, signal_column='acceleration', output_dir=None, prefix=''):
    """
    2x2 summary figure for the single signal preprocessing pipeline:
    residual means, std distribution, baseline correction example,
    normalized signal example.
 
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw signals before preprocessing. Must contain [file, signal_column].
    df_clean : pd.DataFrame
        Preprocessed signals. Must contain [file, signal_column, {signal_column}_normalized].
    signal_column : str
        Name of the signal column (e.g., 'acceleration', 'velocity', 'displacement')
    output_dir : str or Path or None
    prefix : str
        Prefix for output filename (e.g., 'acc', 'vel', 'dis').
    """
    normalized_col = f'{signal_column}_normalized'
    
    baseline_check = df_clean.groupby('file')[signal_column].mean()
    norm_check = df_clean.groupby('file')[normalized_col].std()
 
    example_file = df_clean['file'].unique()[0]
    example_raw = df_raw[df_raw['file'] == example_file][signal_column].values
    example_bc = df_clean[df_clean['file'] == example_file][signal_column].values
    example_norm = df_clean[df_clean['file'] == example_file][normalized_col].values
    station_label = example_file.split('.')[1] + ' ' + example_file.split('.')[3]
    
    # Determine unit label
    if signal_column == 'acceleration':
        unit_label = 'cm/s²'
    elif signal_column == 'velocity':
        unit_label = 'cm/s'
    elif signal_column == 'displacement':
        unit_label = 'cm'
    else:
        unit_label = ''
    
    signal_name = signal_column.capitalize()
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t = np.arange(len(example_raw))
 
    # Residual means
    axes[0, 0].hist(baseline_check.values, bins=30, color=colors[0], edgecolor='none')
    axes[0, 0].axvline(0, color='black', linewidth=1, linestyle='--', label='Expected: 0')
    axes[0, 0].set_title('Residual mean per signal\n(after baseline correction)')
    axes[0, 0].set_xlabel(f'Mean ({unit_label})')
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
    axes[1, 0].plot(t, example_bc, color=colors[0], linewidth=0.5,
                    alpha=0.9, label='Baseline-corrected')
    axes[1, 0].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1, 0].set_title(f'Baseline correction — {station_label}')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel(f'{signal_name} ({unit_label})')
    axes[1, 0].legend(fontsize=9)
 
    # Normalized signal example
    axes[1, 1].plot(t, example_norm, color=colors[1], linewidth=0.5,
                    alpha=0.8, label='Normalized signal')
    axes[1, 1].axhline(0, color='black', linewidth=1, linestyle='--', label='Mean = 0')
    axes[1, 1].axhline(1, color=colors[3], linewidth=0.8, linestyle=':', label='±1 std')
    axes[1, 1].axhline(-1, color=colors[3], linewidth=0.8, linestyle=':')
    axes[1, 1].set_title(f'Normalized signal — {station_label}')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel(f'Normalized {signal_column}')
    axes[1, 1].legend(fontsize=9)
 
    plt.suptitle('Post-preprocessing check — single signal pipeline', fontsize=14)
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'postcheck_single_{prefix}.pdf' if prefix else 'postcheck_single.pdf'
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()

 
# ===============================================================================================
# ================ Signals — post-preprocessing check (moment scaling pipeline) =================
# ===============================================================================================
 
def plot_postcheck_moment_scaling(df_raw, df_long, signal_column='acceleration',
                                  signal_unit='cm/s²', threshold=48000,
                                  output_dir=None, prefix=''):
    """
    Post-preprocessing check plots for the long signals pipeline.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw signal data (before filtering)
    df_long : pd.DataFrame
        Preprocessed long signals (after filtering, baseline correction, NO normalization)
    signal_column : str
        Name of the signal column (e.g., 'acceleration', 'velocity', 'displacement')
    signal_unit : str
        Unit label (e.g., 'cm/s²', 'cm/s', 'cm')
    threshold : int
        Minimum samples threshold used for filtering
    output_dir : str or Path
        Directory to save the figure
    prefix : str
        Prefix for output filename (e.g., 'acc', 'vel', 'dis')
    """
    from pathlib import Path
    import numpy as np
    
    signal_lengths_raw = df_raw.groupby('file')['sample'].max() + 1
    signal_lengths_long = df_long.groupby('file')['sample'].max() + 1
    baseline_check_agg = df_long.groupby('file')[signal_column].mean()
    
    signal_name = signal_column.capitalize()
    
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
    axes[1].set_xlabel(f'Mean ({signal_unit})')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual mean per signal\n(after baseline correction)\nExpected: 0')
    axes[1].legend()
    axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    # Plot 3: Physical units preserved (show std distribution)
    std_check = df_long.groupby('file')[signal_column].std()
    axes[2].hist(std_check, bins=15, color=colors[3], edgecolor='none')
    axes[2].set_xlabel(f'Std ({signal_unit})')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Standard deviation per signal\n(physical units preserved)\nNOT normalized')
    
    plt.suptitle('Post-preprocessing check — moment scaling pipeline', fontsize=13)
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'postcheck_moment_scaling_{prefix}.pdf' if prefix else 'postcheck_moment_scaling.pdf'
        plt.savefig(output_dir / filename, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()

# ===============================================================================================
# ==================================== Empirical PDFs ===========================================
# ===============================================================================================

def plot_empirical_pdfs(df_clean, signal_column='acceleration', signal_unit='cm/s²',
                        bins=100, log_scale=False, normalized=True,
                        output_dir='../figures/03_single_signal/03a_pdf_analysis/pdf_single',
                        prefix=''):
    """
    Plot empirical PDF for each signal.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Preprocessed signal data
    signal_column : str
        Name of the signal column (e.g., 'acceleration', 'velocity', 'displacement')
    signal_unit : str
        Unit label (e.g., 'cm/s²', 'cm/s', 'cm')
    bins : int
        Number of histogram bins
    log_scale : bool
        If True, use log scale on y-axis
    normalized : bool
        If True, use normalized column
    output_dir : str or Path
        Directory to save figures
    prefix : str
        Prefix for output filenames (e.g., 'acc', 'vel', 'dis')
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    col = f'{signal_column}_normalized' if normalized else signal_column
    signal_name = signal_column.capitalize()
    
    saved = []
    failed = []
    
    for file in df_clean['file'].unique():
        signal = df_clean[df_clean['file'] == file][col].values
        station = file.split('.')[1]
        stream = file.split('.')[3]
        
        filename = f'pdf_{station}_{stream}'
        if prefix:
            filename = f'{prefix}_{filename}'
        filepath = f'{output_dir}/{filename}.pdf'
        
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(signal, bins=bins, color=colors[0], edgecolor='none', density=True)
            if log_scale:
                ax.set_yscale('log')
            
            if normalized:
                ax.set_xlabel(f'Normalized {signal_column}')
            else:
                ax.set_xlabel(f'{signal_name} ({signal_unit})')
            
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
    print(f"Saved: {len(saved)}/{df_clean['file'].nunique()} PDFs")
    if failed:
        print("Failed:")
        for f, e in failed:
            print(f"  - {f}: {e}")
    else:
        print("All PDFs saved successfully!")