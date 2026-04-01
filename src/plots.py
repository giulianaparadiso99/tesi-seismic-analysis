import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from adjustText import adjust_text
from src.plot_settings import set_plot_style
colors = set_plot_style()

# ===============================================================================================
# ============================= Metadata — column types pie chart ================================
# ===============================================================================================
 
def plot_column_types_pie(df, output_dir=None):
    """
    Pie chart of column types distribution after preprocessing.
 
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed metadata dataframe.
    output_dir : str or Path or None
        Directory to save the figure. If None, the figure is not saved.
    """
    type_counts = df.dtypes.astype(str).value_counts()
 
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        type_counts,
        labels=type_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
 
    ax.set_title('\nColumn types distribution')
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'column_types_distribution.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — numerical distributions ==================================
# ===============================================================================================
 
def plot_numerical_distributions(df, num_cols, output_dir=None):
    """
    Grid of histograms for the specified numerical columns.
 
    Parameters
    ----------
    df : pd.DataFrame
    num_cols : list of str
        Numerical column names to plot.
    output_dir : str or Path or None
    """
    ncols = 3
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
 
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=15, color=colors[0],
                     edgecolor='white', linewidth=0.5)
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_ylabel('Count')
 
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)
 
    plt.suptitle('Numerical columns distributions', y=1.01)
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'numerical_distributions.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — categorical distributions ================================
# ===============================================================================================
 
def plot_categorical_distributions(df, cat_cols, output_dir=None):
    """
    Bar charts for categorical columns. Columns with few short categories
    are plotted vertically (saved as *_few.pdf); columns with many or long
    categories are plotted horizontally (saved as *_many.pdf).
 
    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list of str
    output_dir : str or Path or None
    """
    few_cats  = [c for c in cat_cols
                 if df[c].nunique() <= 8 and df[c].str.len().max() <= 10]
    many_cats = [c for c in cat_cols
                 if df[c].nunique() > 8  or  df[c].str.len().max() > 10]
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
 
    # --- few categories: vertical bars ---
    if few_cats:
        fig, axes = plt.subplots(1, len(few_cats), figsize=(4 * len(few_cats), 4))
        if len(few_cats) == 1:
            axes = [axes]
        for i, col in enumerate(few_cats):
            counts = df[col].value_counts()
            axes[i].bar(counts.index, counts.values, color=colors[1],
                        edgecolor='white', linewidth=0.5)
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_ylabel('Count')
            if df[col].str.len().max() > 10:
                axes[i].tick_params(axis='x', rotation=45, labelsize=6)
            else:
                axes[i].tick_params(axis='x', rotation=30)
        plt.suptitle('Categorical columns distributions', y=1.01)
        plt.tight_layout()
        if output_dir is not None:
            path = os.path.join(output_dir, 'categorical_distributions_few.pdf')
            plt.savefig(path, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.show()
        plt.close()
 
    # --- many categories: horizontal bars ---
    if many_cats:
        fig, axes = plt.subplots(1, len(many_cats), figsize=(8 * len(many_cats), 10))
        if len(many_cats) == 1:
            axes = [axes]
        for i, col in enumerate(many_cats):
            counts = df[col].value_counts()
            axes[i].barh(counts.index, counts.values, color=colors[2],
                         edgecolor='white', linewidth=0.5)
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_xlabel('Count')
            axes[i].tick_params(axis='y', labelsize=7)
            axes[i].invert_yaxis()
        plt.suptitle('Categorical columns distributions', y=1.01)
        plt.tight_layout()
        if output_dir is not None:
            path = os.path.join(output_dir, 'categorical_distributions_many.pdf')
            plt.savefig(path, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.show()
        plt.close()
 
 
# ===============================================================================================
# ============================== Metadata — correlation matrix ==================================
# ===============================================================================================
 
def plot_correlation_matrix(corr, title, output_path=None):
    """
    Heatmap of a correlation (or correlation difference) matrix.
    Reused for: global matrix, per-component matrices, per-group matrices,
    correlation difference matrices.
 
    Parameters
    ----------
    corr : pd.DataFrame
        Square matrix of correlation coefficients.
    title : str
        Plot title.
    output_path : str or Path or None
        Full file path (including filename) to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='inferno',
        center=0,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 9},
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation coefficient'}
    )
    ax.grid(False)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
 
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ====================== Metadata — significant correlation differences =========================
# ===============================================================================================
 
def plot_significant_corr_diff(diff, significant_mask, title, output_path=None):
    """
    Heatmap showing only statistically significant correlation differences
    (non-significant cells are masked).
 
    Parameters
    ----------
    diff : pd.DataFrame
        Matrix of correlation differences (g1 - g2).
    significant_mask : pd.DataFrame of bool
        Boolean mask — True where the difference is significant (p < alpha).
    title : str
    output_path : str or Path or None
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        diff,
        annot=True,
        fmt='.2f',
        cmap='inferno',
        center=0,
        vmin=-2,
        vmax=2,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        annot_kws={'size': 9},
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation difference'},
        mask=~significant_mask
    )
    ax.grid(False)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
 
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ================================= Metadata — station map ======================================
# ===============================================================================================
 
def plot_station_map(df, event_lat, event_lon, output_path=None):
    """
    Map of seismic stations colored by PGA, with epicenter and station labels.
 
    Parameters
    ----------
    df : pd.DataFrame
        Must contain STATION_LONGITUDE_DEGREE, STATION_LATITUDE_DEGREE,
        PGA_CM/S^2, STATION_CODE.
    event_lat : float
    event_lon : float
    output_path : str or Path or None
    """
    fig, ax = plt.subplots(figsize=(10, 10))
 
    scatter = ax.scatter(
        df['STATION_LONGITUDE_DEGREE'],
        df['STATION_LATITUDE_DEGREE'],
        c=df['PGA_CM/S^2'],
        cmap='inferno',
        s=80,
        zorder=5,
        label='Seismic stations',
        edgecolors='black',
        linewidths=1.5
    )
 
    texts = []
    for _, row in df.drop_duplicates('STATION_CODE').iterrows():
        x_off = row['STATION_LONGITUDE_DEGREE'] - 0.15 \
            if row['STATION_CODE'] == 'SURF' \
            else row['STATION_LONGITUDE_DEGREE']
        texts.append(ax.text(
            x_off, row['STATION_LATITUDE_DEGREE'],
            row['STATION_CODE'],
            fontsize=10, color='white', fontweight='bold',
            path_effects=[
                plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
            ],
            zorder=6
        ))
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.5, 1.5))
 
    ax.scatter(event_lon, event_lat, marker='*', color='red', s=400, zorder=7,
               label='Epicenter', edgecolors='black', linewidths=1.5)
 
    ctx.add_basemap(ax, crs='EPSG:4326',
                    source=ctx.providers.OpenStreetMap.Mapnik, zoom=8)
    plt.colorbar(scatter, ax=ax, label='PGA (cm/s²)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Seismic stations and epicenter')
    ax.legend()
    plt.tight_layout()
 
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ============================= Metadata — PGA and duration analysis =============================
# ===============================================================================================
 
def plot_pga_and_duration_by_component(df, components, comp_colors, output_dir=None):
    """
    Three plots: PGA by component (boxplot), PGA vs epicentral distance
    by component (scatter), and signal duration by component (boxplot).
 
    Parameters
    ----------
    df : pd.DataFrame
        Must contain COMPONENT, PGA_CM/S^2, EPICENTRAL_DISTANCE_KM, DURATION_S.
    components : list of str
        e.g. ['E', 'N', 'Z']
    comp_colors : list
        One color per component.
    output_dir : str or Path or None
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
 
    boxplot_kwargs = dict(
        patch_artist=True, widths=0.5,
        medianprops={'color': 'white', 'linewidth': 2},
        whiskerprops={'linewidth': 1.2},
        capprops={'linewidth': 1.2},
        flierprops={'marker': 'o', 'markersize': 5,
                    'markeredgecolor': 'gray', 'markerfacecolor': 'none'}
    )
 
    # 1. PGA by component
    fig, ax = plt.subplots(figsize=(6, 5))
    comp_data = [df[df['COMPONENT'] == c]['PGA_CM/S^2'].dropna().values
                 for c in components]
    bp = ax.boxplot(comp_data, tick_labels=components, **boxplot_kwargs)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(comp_colors[i])
        patch.set_alpha(0.85)
    ax.set_xlabel('Component')
    ax.set_ylabel('PGA (cm/s²)')
    ax.set_title('PGA by component')
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'pga_by_component.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()
 
    # 2. PGA vs epicentral distance by component
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, comp in enumerate(components):
        subset = df[df['COMPONENT'] == comp]
        ax.scatter(subset['EPICENTRAL_DISTANCE_KM'], subset['PGA_CM/S^2'],
                   color=comp_colors[i], edgecolors='white', linewidths=0.5,
                   s=70, label=comp, alpha=0.9, zorder=5)
    ax.set_xlabel('Epicentral distance (km)')
    ax.set_ylabel('PGA (cm/s²)')
    ax.set_title('PGA vs epicentral distance by component')
    ax.legend(title='Component')
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'pga_vs_distance_by_component.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()
 
    # 3. Duration by component
    fig, ax = plt.subplots(figsize=(6, 5))
    dur_data = [df[df['COMPONENT'] == c]['DURATION_S'].dropna().values
                for c in components]
    bp2 = ax.boxplot(dur_data, tick_labels=components, **boxplot_kwargs)
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(comp_colors[i])
        patch.set_alpha(0.85)
    ax.set_xlabel('Component')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Signal duration by component')
    plt.tight_layout()
    if output_dir is not None:
        path = os.path.join(output_dir, 'duration_by_component.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — PGA correlation by distance group ========================
# ===============================================================================================
 
def plot_pga_correlation_by_group(df_pga_corr, groups, group_colors, output_dir=None):
    """
    Grouped bar chart of PGA correlation with all numerical variables,
    one bar group per distance group.
 
    Parameters
    ----------
    df_pga_corr : pd.DataFrame
        DataFrame with variables as index and distance groups as columns,
        containing Pearson correlation coefficients with PGA.
        Typically built as:
            pd.DataFrame({g: corr_by_group[g] for g in groups})
    groups : list of str
    group_colors : list
    output_dir : str or Path or None
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df_pga_corr))
    width = 0.25
 
    for i, group in enumerate(groups):
        ax.bar(
            [xi + i * width for xi in x],
            df_pga_corr[group],
            width=width,
            label=group,
            color=group_colors[i],
            edgecolor='white',
            linewidth=0.5
        )
 
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(df_pga_corr.index, rotation=45, ha='right')
    ax.set_xlabel('Numerical variables')
    ax.set_ylabel('Correlation coefficient')
    ax.set_title('PGA correlation with numerical variables by distance group')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(title='Distance group')
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'pga_correlation_by_distance.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()

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
 
def plot_example_signals(df_acc, df_meta_clean, streams=('HNE', 'HNN', 'HNZ'), output_dir=None):
    """
    One subplot per stream showing a representative acceleration time series.
 
    Parameters
    ----------
    df_acc : pd.DataFrame
        Must contain columns [file, sample, acceleration].
    df_meta_clean : pd.DataFrame
        Must contain columns [file, STREAM].
    streams : tuple of str
        Stream codes to plot, one panel each.
    output_dir : str or Path or None
    """
    fig, axes = plt.subplots(len(streams), 1, figsize=(12, 3 * len(streams)),
                             sharex=False)
    if len(streams) == 1:
        axes = [axes]
 
    for i, stream in enumerate(streams):
        example_file = df_meta_clean[df_meta_clean['STREAM'] == stream]['file'].iloc[0]
        signal = df_acc[df_acc['file'] == example_file]
        axes[i].plot(signal['sample'], signal['acceleration'],
                     color=colors[i], linewidth=0.5)
        axes[i].set_title(f'Example signal — {stream}')
        axes[i].set_ylabel('Acceleration (cm/s²)')
 
    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'example_signals.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()
 
 
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
# ================= Signals — post-preprocessing check (single pipeline) =======================
# ===============================================================================================
 
def plot_postcheck_single(df_acc_raw, df_acc_clean, output_dir=None):
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
# ================ Signals — post-preprocessing check (aggregated pipeline) ====================
# ===============================================================================================
 
def plot_postcheck_long(df_acc_raw, df_acc_long, threshold=48000, output_dir=None):
    """
    1x3 summary figure for the aggregated (long signals) preprocessing pipeline:
    signal length distribution before/after truncation, residual means, std distribution.
 
    Parameters
    ----------
    df_acc_raw : pd.DataFrame
        Raw accelerations before preprocessing. Must contain [file, sample].
    df_acc_long : pd.DataFrame
        Preprocessed long-signal accelerations. Must contain
        [file, sample, acceleration, acceleration_normalized].
    threshold : int
        Truncation threshold in samples (default: 48000).
    output_dir : str or Path or None
    """
    signal_lengths_raw  = df_acc_raw.groupby('file')['sample'].max() + 1
    signal_lengths_long = df_acc_long.groupby('file')['sample'].max() + 1
    baseline_check_agg  = df_acc_long.groupby('file')['acceleration'].mean()
    norm_check_agg      = df_acc_long.groupby('file')['acceleration_normalized'].std()
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    # Signal lengths before and after truncation
    axes[0].hist(signal_lengths_raw.values, bins=20, color=colors[2],
                 edgecolor='none', alpha=0.7,
                 label=f'Before (n={len(signal_lengths_raw)})')
    axes[0].hist(signal_lengths_long.values, bins=5, color=colors[0],
                 edgecolor='none', alpha=0.9,
                 label=f'After (n={len(signal_lengths_long)})')
    axes[0].axvline(threshold, color='black', linewidth=1, linestyle='--',
                    label=f'Threshold: {threshold:,}')
    axes[0].set_title('Signal lengths before and after truncation')
    axes[0].set_xlabel('Number of samples')
    axes[0].set_ylabel('Count')
    axes[0].legend(fontsize=9)
 
    # Residual mean distribution
    axes[1].hist(baseline_check_agg.values, bins=20, color=colors[0], edgecolor='none')
    axes[1].axvline(0, color='black', linewidth=1, linestyle='--', label='Expected: 0')
    axes[1].set_title('Residual mean per signal\n(aggregated, after baseline correction)')
    axes[1].set_xlabel('Mean (cm/s²)')
    axes[1].set_ylabel('Count')
    axes[1].legend()
 
    # Std distribution
    axes[2].hist(norm_check_agg.values, bins=20, color=colors[1], edgecolor='none')
    axes[2].axvline(1, color='black', linewidth=1, linestyle='--', label='Expected: 1')
    axes[2].set_title('Standard deviation per signal\n(aggregated, after normalization)')
    axes[2].set_xlabel('Std')
    axes[2].set_ylabel('Count')
    axes[2].legend()
 
    plt.suptitle('Post-preprocessing check — long signals pipeline', fontsize=14)
    plt.tight_layout()
 
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'postcheck_long.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f"Saved: {path}")
 
    plt.show()
    plt.close()

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
# ================================ Increment distribution =======================================
# ===============================================================================================

def plot_increments_distribution(df_increments, tau_values=None, bins=100, 
                                 normalized=True, output_dir='../figures/03_increments'):
    """
    Plots the distribution of increments for different tau values.
    
    Parameters
    ----------
    df_increments : pd.DataFrame
        DataFrame with columns [file, station, stream, tau, increment]
    tau_values : list of int, optional
        Specific tau values to plot. If None, selects representative values.
    bins : int
        Number of histogram bins
    normalized : bool
        If True, plots are for normalized signals
    output_dir : str
        Directory to save figures
    
    Returns
    -------
    None (saves figures to disk)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select tau values if not provided
    if tau_values is None:
        all_tau = sorted(df_increments['tau'].unique())
        # Select ~6 representative values (log-spaced if possible)
        if len(all_tau) > 6:
            indices = np.linspace(0, len(all_tau)-1, 6, dtype=int)
            tau_values = [all_tau[i] for i in indices]
        else:
            tau_values = all_tau
    
    # 1. Histogram for different tau values
    n_tau = len(tau_values)
    n_cols = 3
    n_rows = int(np.ceil(n_tau / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_tau > 1 else [axes]
    
    for i, tau in enumerate(tau_values):
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        abs_increments = np.abs(increments)
        
        # Statistics
        mean_inc = np.mean(abs_increments)
        median_inc = np.median(abs_increments)
        frac_less_1 = np.mean(abs_increments < 1)
        
        # Histogram
        axes[i].hist(increments, bins=bins, color=colors[i % len(colors)],
                    edgecolor='none', alpha=0.7, density=True)
        axes[i].axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        axes[i].axvline(-1, color='red', linewidth=1, linestyle=':', alpha=0.5, label='|Δ|=1')
        axes[i].axvline(1, color='red', linewidth=1, linestyle=':', alpha=0.5)
        axes[i].set_xlabel('Increment')
        axes[i].set_ylabel('Probability density')
        axes[i].set_title(f'τ = {tau}\n⟨|Δ|⟩={mean_inc:.3f}, frac(|Δ|<1)={frac_less_1:.2%}')
        axes[i].legend()
    
    # Hide unused subplots
    for i in range(n_tau, len(axes)):
        axes[i].set_visible(False)
    
    signal_type = 'normalized' if normalized else 'raw'
    plt.suptitle(f'Increments distribution ({signal_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/increments_histogram.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/increments_histogram.pdf")
    
    # 2. Log-scale histogram (to see tails better)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_tau > 1 else [axes]
    
    for i, tau in enumerate(tau_values):
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        abs_increments = np.abs(increments)
        
        axes[i].hist(abs_increments, bins=bins, color=colors[i % len(colors)],
                    edgecolor='none', alpha=0.7, density=True)
        axes[i].axvline(1, color='red', linewidth=1.5, linestyle='--', label='|Δ|=1')
        axes[i].set_xlabel('|Increment|')
        axes[i].set_ylabel('Probability density')
        axes[i].set_yscale('log')
        axes[i].set_title(f'τ = {tau}')
        axes[i].legend()
    
    for i in range(n_tau, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'|Increments| distribution - log scale ({signal_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/increments_histogram_log.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/increments_histogram_log.pdf")
    
    # 3. Evolution of statistics with tau
    tau_all = sorted(df_increments['tau'].unique())
    stats_by_tau = []
    
    for tau in tau_all:
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        abs_increments = np.abs(increments)
        
        stats_by_tau.append({
            'tau': tau,
            'mean': np.mean(abs_increments),
            'median': np.median(abs_increments),
            'std': np.std(abs_increments),
            'frac_less_1': np.mean(abs_increments < 1)
        })
    
    df_stats = pd.DataFrame(stats_by_tau)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean vs tau
    axes[0, 0].plot(df_stats['tau'], df_stats['mean'], 'o-', color=colors[0], linewidth=2)
    axes[0, 0].axhline(1, color='red', linestyle='--', linewidth=1, label='threshold = 1')
    axes[0, 0].set_xlabel('τ')
    axes[0, 0].set_ylabel('⟨|Increment|⟩')
    axes[0, 0].set_title('Mean absolute increment vs τ')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Median vs tau
    axes[0, 1].plot(df_stats['tau'], df_stats['median'], 'o-', color=colors[1], linewidth=2)
    axes[0, 1].axhline(1, color='red', linestyle='--', linewidth=1, label='threshold = 1')
    axes[0, 1].set_xlabel('τ')
    axes[0, 1].set_ylabel('Median |Increment|')
    axes[0, 1].set_title('Median absolute increment vs τ')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Std vs tau
    axes[1, 0].plot(df_stats['tau'], df_stats['std'], 'o-', color=colors[2], linewidth=2)
    axes[1, 0].set_xlabel('τ')
    axes[1, 0].set_ylabel('Std(|Increment|)')
    axes[1, 0].set_title('Standard deviation vs τ')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Fraction < 1 vs tau
    axes[1, 1].plot(df_stats['tau'], df_stats['frac_less_1'], 'o-', color=colors[3], linewidth=2)
    axes[1, 1].axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('τ')
    axes[1, 1].set_ylabel('Fraction with |Δ| < 1')
    axes[1, 1].set_title('Fraction of increments < 1 vs τ')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Increment statistics evolution ({signal_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/increments_statistics_vs_tau.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/increments_statistics_vs_tau.pdf")
    
    # 4. Q-Q plot to check distribution shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, tau in enumerate(tau_values[:6]):  # Max 6 for this plot
        increments = df_increments[df_increments['tau'] == tau]['increment'].values
        
        # Q-Q plot vs Gaussian
        (osm, osr), (slope, intercept, r) = stats.probplot(increments, dist='norm')
        axes[i].scatter(osm, osr, color=colors[i % len(colors)], s=10, alpha=0.5)
        axes[i].plot(osm, slope * np.array(osm) + intercept, 
                    color='red', linewidth=2, label=f'R²={r**2:.3f}')
        axes[i].set_xlabel('Theoretical quantiles')
        axes[i].set_ylabel('Sample quantiles')
        axes[i].set_title(f'Q-Q plot (τ={tau})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Q-Q plots vs Gaussian ({signal_type})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/increments_qq_plots.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/increments_qq_plots.pdf")
    
    print(f"\nAll plots saved to: {output_dir}/")

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
# ========================= Scaling — Universal rescaling plot ==================================
# ===============================================================================================

def plot_universal_rescaling(df_exponents, df_piecewise, df_alpha,
                              title_suffix='', output_path=None):
    """
    Universal rescaling plot following Vollmer et al. (2024), Eq. (10).
 
    For each signal and each moment order q, computes the rescaled exponent:
        y(q) = (zeta(q) - q/2) / (slope_high - slope_low)
    and plots it against x = alpha / q, where:
        - slope_low  = xi  (low-q slope from piecewise fit)
        - slope_high = zeta (high-q slope from piecewise fit)
        - alpha      = Hill exponent of displacement increments at small tau
                       (column 'power_law_exp' in df_alpha)
 
    Only signals where slope_high > slope_low are retained, to avoid
    inverting the sign of the rescaled y due to an unphysical piecewise fit.
 
    Under strong anomalous diffusion, all points should collapse onto the
    universal curve y = min(alpha/q, 1).
 
    Parameters
    ----------
    df_exponents : pd.DataFrame
        Output of compute_scaling_exponents.
        Must contain columns [file, station, stream, q, zeta].
    df_piecewise : pd.DataFrame
        Output of fit_piecewise_scaling.
        Must contain columns [file, station, stream, slope_low, slope_high].
    df_alpha : pd.DataFrame
        Must contain columns [station, stream, power_law_exp].
        Recommended source: df_tail_disp_event filtered to a small tau
        (e.g. tau=10), using the hill_exp column renamed to power_law_exp.
        Using the Hill exponent of displacement increments is physically
        more appropriate than using the exponent of the acceleration PDF.
    title_suffix : str
        Appended to the figure title, e.g. 'displacement, event window'.
    output_path : str or Path or None
    """
    # --- Merge all ingredients per signal ---
    df = df_exponents.merge(
        df_piecewise[['file', 'station', 'stream', 'slope_low', 'slope_high']],
        on=['file', 'station', 'stream']
    ).merge(
        df_alpha[['station', 'stream', 'power_law_exp']],
        on=['station', 'stream']
    )
 
    # Keep only signals with finite alpha and physically consistent piecewise fit
    df = df[np.isfinite(df['power_law_exp'])].copy()
    n_before = df['file'].nunique()
    df = df[df['slope_high'] > df['slope_low']].copy()
    n_after = df['file'].nunique()
    if n_before != n_after:
        print(f"Note: {n_before - n_after} signals excluded "
              f"(slope_high <= slope_low — unphysical piecewise fit)")
 
    # --- Compute rescaled quantities ---
    df['x'] = df['power_law_exp'] / df['q']
    df['y'] = (df['zeta'] - df['q'] / 2) / (df['slope_high'] - df['slope_low'])
 
    # --- Universal curve ---
    x_theory = np.linspace(0.0, 3.0, 500)
    y_theory  = np.minimum(x_theory, 1.0)
 
    # --- Grouped statistics (median ± IQR per q) ---
    grouped = df.groupby('q').agg(
        x_med=('x',  'median'),
        x_q25=('x',  lambda v: v.quantile(0.25)),
        x_q75=('x',  lambda v: v.quantile(0.75)),
        y_med=('y',  'median'),
        y_q25=('y',  lambda v: v.quantile(0.25)),
        y_q75=('y',  lambda v: v.quantile(0.75))
    ).reset_index()
 
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    q_values = sorted(df['q'].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))
 
    # Left — all signals overlaid, colored by q
    for ci, q in enumerate(q_values):
        subset = df[df['q'] == q]
        axes[0].scatter(subset['x'], subset['y'],
                        color=cmap[ci], s=25, alpha=0.6,
                        edgecolors='none', label=f'q={q}')
    axes[0].plot(x_theory, y_theory, 'k-', linewidth=2,
                 label='Universal curve', zorder=10)
    axes[0].axhline(1, color='gray', linewidth=0.8, linestyle='--')
    axes[0].axvline(1, color='gray', linewidth=0.8, linestyle='--')
    axes[0].set_xlabel(r'$\alpha / q$')
    axes[0].set_ylabel(r'$(\zeta(q) - q/2)\,/\,(\zeta - \xi)$')
    axes[0].set_title('Universal rescaling — all signals\n(colored by $q$)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xlim(0, 3)
    axes[0].set_ylim(-0.5, 2.5)
 
    # Right — median ± IQR per q, with error bars on both axes
    axes[1].plot(x_theory, y_theory, 'k-', linewidth=2,
                 label='Universal curve', zorder=10)
    axes[1].errorbar(
        grouped['x_med'], grouped['y_med'],
        xerr=[grouped['x_med'] - grouped['x_q25'],
              grouped['x_q75'] - grouped['x_med']],
        yerr=[grouped['y_med'] - grouped['y_q25'],
              grouped['y_q75'] - grouped['y_med']],
        fmt='o', color=colors[0], capsize=4, linewidth=1.5,
        markersize=6, label='Median ± IQR across signals'
    )
    for _, row in grouped.iterrows():
        axes[1].annotate(f"q={row['q']:.1f}",
                         (row['x_med'], row['y_med']),
                         textcoords='offset points', xytext=(6, 4),
                         fontsize=7, color='gray')
    axes[1].axhline(1, color='gray', linewidth=0.8, linestyle='--')
    axes[1].axvline(1, color='gray', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel(r'$\alpha / q$')
    axes[1].set_ylabel(r'$(\zeta(q) - q/2)\,/\,(\zeta - \xi)$')
    axes[1].set_title('Universal rescaling — median ± IQR across signals\n(by $q$)')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 3)
    axes[1].set_ylim(-0.5, 2.5)
 
    title = 'Universal rescaling plot'
    if title_suffix:
        title += f' — {title_suffix}'
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
 
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
    # --- Summary ---
    df['y_theory'] = np.minimum(df['x'], 1.0)
    df['residual'] = np.abs(df['y'] - df['y_theory'])
    tol = 0.2
    frac_close = (df['residual'] < tol).mean()
    n_signals   = df['file'].nunique()
    print(f"Signals used: {n_signals}")
    print(f"Fraction of points within {tol} of universal curve: {frac_close:.2f}")
    print(f"Median residual from universal curve: {df['residual'].median():.3f}")