import os
import numpy as np
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

# ===============================================================================================
# =================== Scaling — R² diagnostic for log-log fit quality ==========================
# ===============================================================================================

def plot_r2_diagnostic(df_exponents, title_suffix='', threshold=0.9, output_path=None):
    """
    Two-panel diagnostic of the log-log fit quality for moment scaling.

    Left panel: boxplot of R² distribution across all signals, by moment order q.
    Right panel: fraction of signals with R² above threshold, by moment order q.

    Parameters
    ----------
    df_exponents : pd.DataFrame
        Output of compute_scaling_exponents. Must contain columns [q, r2].
    title_suffix : str
        Appended to the figure title, e.g. 'displacement, event window'.
    threshold : float
        R² threshold for the fraction plot (default: 0.9).
    output_path : str or Path or None
    """
    q_values = sorted(df_exponents['q'].unique())
    r2_by_q  = [df_exponents[df_exponents['q'] == q]['r2'].values for q in q_values]
    frac_good = [(r2 > threshold).mean() for r2 in r2_by_q]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — boxplot of R² by q
    bp = axes[0].boxplot(r2_by_q,
                         tick_labels=[str(q) for q in q_values],
                         patch_artist=True, widths=0.5,
                         medianprops={'color': 'white', 'linewidth': 2},
                         whiskerprops={'linewidth': 1.2},
                         capprops={'linewidth': 1.2},
                         flierprops={'marker': 'o', 'markersize': 4,
                                     'markeredgecolor': 'gray',
                                     'markerfacecolor': 'none'})
    for patch in bp['boxes']:
        patch.set_facecolor(colors[0])
        patch.set_alpha(0.8)
    axes[0].axhline(threshold, color='crimson', linewidth=1,
                    linestyle='--', label=f'R² = {threshold}')
    axes[0].set_xlabel('q')
    axes[0].set_ylabel('R²')
    axes[0].set_title(f'R² of log $M_q(\\tau)$ vs log $\\tau$ fit\nby moment order $q$')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=9)

    # Right — fraction of signals above threshold
    axes[1].bar([str(q) for q in q_values], frac_good,
                color=colors[1], edgecolor='white', linewidth=0.5)
    axes[1].axhline(1.0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('q')
    axes[1].set_ylabel(f'Fraction of signals with R² > {threshold}')
    axes[1].set_title(f'Fraction of signals with good fit (R² > {threshold})\n'
                      f'by moment order $q$')
    axes[1].set_ylim(0, 1.1)

    title = 'Log-log fit quality'
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

    # Summary printed to stdout for logging in the notebook
    print(f"R² summary by q (threshold = {threshold}):")
    for q, r2 in zip(q_values, r2_by_q):
        print(f"  q={q}: median R² = {np.median(r2):.3f} — "
              f"fraction > {threshold} = {(r2 > threshold).mean():.2f}")

# ===============================================================================================
# ========================= Scaling — Universal rescaling plot ==================================
# ===============================================================================================

def plot_universal_rescaling(df_exponents, df_piecewise, df_heavy_tail,
                              title_suffix='', output_path=None):
    """
    Universal rescaling plot following Vollmer et al. (2024), Eq. (10).

    For each signal and each moment order q, computes the rescaled exponent:
        y(q) = (zeta(q) - q/2) / (zeta_high - slope_low)
    and plots it against x = alpha / q, where:
        - slope_low  = xi  (low-q slope from piecewise fit)
        - slope_high = zeta (high-q slope from piecewise fit)
        - alpha      = Hill power-law exponent from heavy-tail assessment

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
    df_heavy_tail : pd.DataFrame
        Output of heavy_tail_assessment.
        Must contain columns [station, stream, power_law_exp].
    title_suffix : str
        Appended to the figure title, e.g. 'displacement, event window'.
    output_path : str or Path or None
    """
    # --- Merge all ingredients per signal ---
    df = df_exponents.merge(
        df_piecewise[['file', 'station', 'stream', 'slope_low', 'slope_high']],
        on=['file', 'station', 'stream']
    ).merge(
        df_heavy_tail[['station', 'stream', 'power_law_exp']],
        on=['station', 'stream']
    )

    # Drop signals where the denominator (slope_high - slope_low) is zero
    # or where alpha is not finite — avoids division errors
    df = df[np.isfinite(df['power_law_exp'])]
    df = df[df['slope_high'] != df['slope_low']].copy()

    # --- Compute rescaled quantities ---
    # x = alpha / q
    df['x'] = df['power_law_exp'] / df['q']
    # y = (zeta(q) - q/2) / (slope_high - slope_low)
    df['y'] = (df['zeta'] - df['q'] / 2) / (df['slope_high'] - df['slope_low'])

    # --- Universal curve ---
    x_theory = np.linspace(0.01, 3.0, 500)
    y_theory = np.minimum(x_theory, 1.0)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — all signals overlaid, colored by q
    q_values = sorted(df['q'].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))

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

    # Right — median ± IQR across signals for each q
    median_y = df.groupby('x')['y'].median()  # not ideal, use q-based grouping
    grouped = df.groupby('q').agg(
        x_med=('x', 'median'),
        y_med=('y', 'median'),
        y_q25=('y', lambda v: v.quantile(0.25)),
        y_q75=('y', lambda v: v.quantile(0.75))
    ).reset_index()

    axes[1].plot(x_theory, y_theory, 'k-', linewidth=2,
                 label='Universal curve', zorder=10)
    axes[1].errorbar(
        grouped['x_med'], grouped['y_med'],
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

    # Summary: fraction of points within tolerance of the universal curve
    df['y_theory'] = np.minimum(df['x'], 1.0)
    df['residual'] = np.abs(df['y'] - df['y_theory'])
    tol = 0.2
    frac_close = (df['residual'] < tol).mean()
    print(f"Fraction of points within {tol} of universal curve: {frac_close:.2f}")
    print(f"Median residual from universal curve: {df['residual'].median():.3f}")