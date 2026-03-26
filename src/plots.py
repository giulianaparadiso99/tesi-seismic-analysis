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