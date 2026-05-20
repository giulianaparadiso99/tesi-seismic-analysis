"""
plots_metadata.py
-----------------
Visualization functions for seismic event metadata analysis. This module
provides a comprehensive suite of plotting utilities for exploring the
preprocessed metadata from .ASC file headers, including both event-level
(constant across all files, e.g., EVENT_DATE, EVENT_LATITUDE_DEGREE) and
station-level (varying per file, e.g., STATION_CODE, EPICENTRAL_DISTANCE_KM)
information.

The module is organized into thematic sections, each addressing a specific
aspect of metadata visualization:

    1. Column types distribution        — pie chart of data types after preprocessing
    2. Numerical distributions          — histogram grid for numeric variables
    3. Categorical distributions        — bar charts for categorical variables
    4. Correlation matrix               — heatmap of correlation coefficients
    5. Correlation differences          — scatter plot of significant differences
    6. Station map                      — geographic visualization with PGA overlay
    7. PGA and duration analysis        — boxplots and scatter plots by component
    8. PGA correlation by distance      — grouped bar chart by epicentral distance

All plotting functions accept an optional output directory or path parameter
to save figures as PDF files. Figures are displayed interactively and then
closed to prevent memory accumulation.

Usage:
    from src.visualization.plots_metadata import plot_column_types_pie, plot_station_map
    
    # Example: visualize column types
    plot_column_types_pie(df_meta_clean, output_dir='../figures/01_metadata')
    
    # Example: create station map
    plot_station_map(df_meta_clean, event_lat=42.8, event_lon=13.1,
                     output_path='../figures/01_metadata/station_map.pdf')
"""

from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from adjustText import adjust_text
from src.visualization.plot_settings import set_plot_style

colors, colors1 = set_plot_style()

# Default peak column for backward compatibility
DEFAULT_PEAK_COLUMN = 'PGA_CM/S^2'


# ===============================================================================================
# ======================================= Helper functions ======================================
# ===============================================================================================

def _get_peak_labels(peak_column: str) -> tuple[str, str]:
    """
    Get full label and short label for peak ground motion column.
    
    Parameters
    ----------
    peak_column : str
        Column name (e.g., 'PGA_CM/S^2', 'PGV_CM/S', 'PGD_CM')
    
    Returns
    -------
    tuple[str, str]
        (full_label, short_label) e.g., ('PGA (cm/s²)', 'PGA')
    """
    if 'PGA' in peak_column:
        return 'PGA (cm/s²)', 'PGA'
    elif 'PGV' in peak_column:
        return 'PGV (cm/s)', 'PGV'
    elif 'PGD' in peak_column:
        return 'PGD (cm)', 'PGD'
    else:
        return peak_column, peak_column.split('_')[0]


# ===============================================================================================
# ============================= Metadata — column types pie chart ================================
# ===============================================================================================
 
def plot_column_types_pie(
    df: pd.DataFrame, 
    output_dir: Optional[Union[str, Path]] = None, 
    filename: str = 'column_types_distribution.pdf'
) -> None:
    """
    Pie chart of column types distribution after preprocessing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed metadata dataframe.
    output_dir : str or Path, optional
        Directory to save the figure. If None, the figure is not saved.
    filename : str, default='column_types_distribution.pdf'
        Name of the output file.
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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — numerical distributions ==================================
# ===============================================================================================
 
def plot_numerical_distributions(
    df: pd.DataFrame, 
    num_cols: List[str], 
    output_dir: Optional[Union[str, Path]] = None, 
    filename: str = 'numerical_distributions.pdf'
) -> None:
    """
    Grid of histograms for the specified numerical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed metadata dataframe.
    num_cols : list of str
        Numerical column names to plot.
    output_dir : str or Path, optional
        Directory to save the figure.
    filename : str, default='numerical_distributions.pdf'
        Name of the output file.
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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — categorical distributions ================================
# ===============================================================================================
 
def plot_categorical_distributions(
    df: pd.DataFrame, 
    cat_cols: List[str], 
    output_dir: Optional[Union[str, Path]] = None, 
    prefix: str = ''
) -> None:
    """
    Bar charts for categorical columns. Columns with few short categories
    are plotted vertically (saved as *_few.pdf); columns with many or long
    categories are plotted horizontally (saved as *_many.pdf).
 
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed metadata dataframe.
    cat_cols : list of str
        Categorical column names to plot.
    output_dir : str or Path, optional
        Directory to save figures.
    prefix : str, default=''
        Prefix for output filenames (e.g., 'acc', 'vel', 'dis').
    """
    few_cats  = [c for c in cat_cols
                 if df[c].nunique() <= 8 and df[c].str.len().max() <= 10]
    many_cats = [c for c in cat_cols
                 if df[c].nunique() > 8  or  df[c].str.len().max() > 10]
 
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
 
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
        if output_path is not None:
            filename = f'categorical_distributions_few_{prefix}.pdf' if prefix else 'categorical_distributions_few.pdf'
            save_path = output_path / filename
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")
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
        if output_path is not None:
            filename = f'categorical_distributions_many_{prefix}.pdf' if prefix else 'categorical_distributions_many.pdf'
            save_path = output_path / filename
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        plt.close()
 
 
# ===============================================================================================
# ============================== Metadata — correlation matrix ==================================
# ===============================================================================================
 
def plot_correlation_matrix(
    corr: pd.DataFrame, 
    title: str, 
    output_path: Optional[Union[str, Path]] = None
) -> None:
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
    output_path : str or Path, optional
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
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ====================== Metadata — significant correlation differences =========================
# ===============================================================================================
 
def plot_significant_corr_diff(
    diff: pd.DataFrame, 
    significant_mask: pd.DataFrame, 
    title: str, 
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Heatmap showing only statistically significant correlation differences
    (non-significant cells are masked).
 
    Parameters
    ----------
    diff : pd.DataFrame
        Matrix of correlation differences (g1 - g2).
    significant_mask : pd.DataFrame
        Boolean mask — True where the difference is significant (p < alpha).
    title : str
        Plot title.
    output_path : str or Path, optional
        Full file path to save the figure.
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
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
 
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ================================= Metadata — station map ======================================
# ===============================================================================================

def plot_station_map(
    df: pd.DataFrame, 
    event_lat: float, 
    event_lon: float, 
    output_path: Optional[Union[str, Path]] = None, 
    peak_column: str = DEFAULT_PEAK_COLUMN
) -> None:
    """
    Map of seismic stations colored by peak ground motion, with epicenter and station labels.
    Uses Cartopy for vector-based high-quality output.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain STATION_LONGITUDE_DEGREE, STATION_LATITUDE_DEGREE,
        peak_column, STATION_CODE.
    event_lat : float
        Epicenter latitude in degrees.
    event_lon : float
        Epicenter longitude in degrees.
    output_path : str or Path, optional
        Full file path to save the figure.
    peak_column : str, default='PGA_CM/S^2'
        Column name for peak ground motion.
        Use 'PGV_CM/S' for velocity, 'PGD_CM' for displacement.
    """
    # Check if peak column exists
    if peak_column not in df.columns:
        raise ValueError(f"Column '{peak_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent based on data
    lon_margin = 0.3
    lat_margin = 0.3
    lon_min = df['STATION_LONGITUDE_DEGREE'].min() - lon_margin
    lon_max = df['STATION_LONGITUDE_DEGREE'].max() + lon_margin
    lat_min = df['STATION_LATITUDE_DEGREE'].min() - lat_margin
    lat_max = df['STATION_LATITUDE_DEGREE'].max() + lat_margin
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add geographic features (high resolution)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f5f5f5', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#e3f2fd', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8, edgecolor='#555555', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=1.2, edgecolor='#333333', linestyle='--', zorder=1)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.5, edgecolor='#64b5f6', zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#e3f2fd', edgecolor='#64b5f6', linewidth=0.3, zorder=1)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.4, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    # Plot stations
    scatter = ax.scatter(
        df['STATION_LONGITUDE_DEGREE'],
        df['STATION_LATITUDE_DEGREE'],
        c=df[peak_column],
        cmap='inferno',
        s=100,
        zorder=5,
        label='Seismic stations',
        edgecolors='black',
        linewidths=1.5,
        transform=ccrs.PlateCarree()
    )
    
    # Station labels
    texts = []
    for _, row in df.drop_duplicates('STATION_CODE').iterrows():
        x_off = row['STATION_LONGITUDE_DEGREE'] - 0.15 if row['STATION_CODE'] == 'SURF' else row['STATION_LONGITUDE_DEGREE']
        texts.append(ax.text(
            x_off, 
            row['STATION_LATITUDE_DEGREE'],
            row['STATION_CODE'],
            fontsize=9, 
            color='white', 
            fontweight='bold',
            path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2.5, foreground='black')],
            zorder=6,
            transform=ccrs.PlateCarree()
        ))
    
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.5, 1.5))
    
    # Epicenter
    ax.scatter(
        event_lon, event_lat, 
        marker='*', 
        color='red', 
        s=500, 
        zorder=7,
        label='Epicenter', 
        edgecolors='black', 
        linewidths=1.5,
        transform=ccrs.PlateCarree()
    )
    
    # Colorbar with appropriate label
    peak_label, _ = _get_peak_labels(peak_column)
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, fraction=0.046, shrink=0.8)
    cbar.set_label(peak_label, fontsize=11)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Seismic stations and epicenter', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.pdf')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()

 
# ===============================================================================================
# ============================= Metadata — PGA and duration analysis =============================
# ===============================================================================================
 
def plot_pga_and_duration_by_component(
    df: pd.DataFrame, 
    components: List[str], 
    comp_colors: List, 
    output_dir: Optional[Union[str, Path]] = None, 
    prefix: str = '', 
    peak_column: str = DEFAULT_PEAK_COLUMN
) -> None:
    """
    Three plots: peak ground motion by component (boxplot), peak vs epicentral distance
    by component (scatter), and signal duration by component (boxplot).
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain COMPONENT, peak_column, EPICENTRAL_DISTANCE_KM, DURATION_S.
    components : list of str
        Component names, e.g. ['E', 'N', 'Z'].
    comp_colors : list
        One color per component.
    output_dir : str or Path, optional
        Directory to save figures.
    prefix : str, default=''
        Prefix for output filenames (e.g., 'acc', 'vel', 'dis').
    peak_column : str, default='PGA_CM/S^2'
        Column name for peak ground motion.
    """
    peak_label, peak_short = _get_peak_labels(peak_column)
    
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    
    boxplot_kwargs = dict(
        patch_artist=True, widths=0.5,
        medianprops={'color': 'white', 'linewidth': 2},
        whiskerprops={'linewidth': 1.2},
        capprops={'linewidth': 1.2},
        flierprops={'marker': 'o', 'markersize': 5,
                    'markeredgecolor': 'gray', 'markerfacecolor': 'none'}
    )
    
    # 1. Peak ground motion by component
    fig, ax = plt.subplots(figsize=(6, 5))
    comp_data = [df[df['COMPONENT'] == c][peak_column].dropna().values
                 for c in components]
    bp = ax.boxplot(comp_data, tick_labels=components, **boxplot_kwargs)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(comp_colors[i])
        patch.set_alpha(0.85)
    ax.set_xlabel('Component')
    ax.set_ylabel(peak_label)
    ax.set_title(f'{peak_short} by component')
    plt.tight_layout()
    if output_path is not None:
        filename = f'pga_by_component_{prefix}.pdf' if prefix else 'pga_by_component.pdf'
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
    
    # 2. Peak ground motion vs epicentral distance by component
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, comp in enumerate(components):
        subset = df[df['COMPONENT'] == comp]
        ax.scatter(subset['EPICENTRAL_DISTANCE_KM'], subset[peak_column],
                   color=comp_colors[i], edgecolors='white', linewidths=0.5,
                   s=70, label=comp, alpha=0.9, zorder=5)
    ax.set_xlabel('Epicentral distance (km)')
    ax.set_ylabel(peak_label)
    ax.set_title(f'{peak_short} vs epicentral distance by component')
    ax.legend(title='Component')
    plt.tight_layout()
    if output_path is not None:
        filename = f'pga_vs_distance_by_component_{prefix}.pdf' if prefix else 'pga_vs_distance_by_component.pdf'
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
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
    if output_path is not None:
        filename = f'duration_by_component_{prefix}.pdf' if prefix else 'duration_by_component.pdf'
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()
 
 
# ===============================================================================================
# ========================= Metadata — PGA correlation by distance group ========================
# ===============================================================================================
 
def plot_pga_correlation_by_group(
    df_pga_corr: pd.DataFrame, 
    groups: List[str], 
    group_colors: List, 
    output_dir: Optional[Union[str, Path]] = None, 
    prefix: str = '', 
    peak_column: str = DEFAULT_PEAK_COLUMN
) -> None:
    """
    Grouped bar chart of peak ground motion correlation with all numerical variables,
    one bar group per distance group.
    
    Parameters
    ----------
    df_pga_corr : pd.DataFrame
        DataFrame with variables as index and distance groups as columns,
        containing Pearson correlation coefficients with peak ground motion.
        Typically built as:
            pd.DataFrame({g: corr_by_group[g] for g in groups})
    groups : list of str
        Distance group names (e.g., ['Near', 'Mid', 'Far']).
    group_colors : list
        One color per group.
    output_dir : str or Path, optional
        Directory to save the figure.
    prefix : str, default=''
        Prefix for output filename (e.g., 'acc', 'vel', 'dis').
    peak_column : str, default='PGA_CM/S^2'
        Column name for peak ground motion.
    """
    _, peak_short = _get_peak_labels(peak_column)
    
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
    ax.set_title(f'{peak_short} correlation with numerical variables by distance group')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(title='Distance group')
    plt.tight_layout()
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f'pga_correlation_by_distance_{prefix}.pdf' if prefix else 'pga_correlation_by_distance.pdf'
        save_path = output_path / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()