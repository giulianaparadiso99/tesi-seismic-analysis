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
    from src.plots_metadata import plot_column_types_pie, plot_station_map
    
    # Example: visualize column types
    plot_column_types_pie(df_meta_clean, output_dir='../figures/01_metadata')
    
    # Example: create station map
    plot_station_map(df_meta_clean, event_lat=42.8, event_lon=13.1,
                     output_path='../figures/01_metadata/station_map.pdf')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from adjustText import adjust_text
from src.visualization.plot_settings import set_plot_style
colors, colors1 = set_plot_style()

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
    Uses Cartopy for vector-based high-quality output.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain STATION_LONGITUDE_DEGREE, STATION_LATITUDE_DEGREE,
        PGA_CM/S^2, STATION_CODE.
    event_lat : float
    event_lon : float
    output_path : str or Path or None
    """
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
        c=df['PGA_CM/S^2'],
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
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05, fraction=0.046, shrink=0.8)
    cbar.set_label('PGA (cm/s²)', fontsize=11)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Seismic stations and epicenter', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()

import folium
from folium import plugins
import branca.colormap as cm
import os

def plot_station_map_folium(df, event_lat, event_lon, output_path=None):
    """
    Interactive map of seismic stations colored by PGA, with epicenter and station labels.
    Uses Folium for web-based visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain STATION_LONGITUDE_DEGREE, STATION_LATITUDE_DEGREE,
        PGA_CM/S^2, STATION_CODE.
    event_lat : float
    event_lon : float
    output_path : str or Path or None
        If provided, saves HTML file.
    
    Returns
    -------
    folium.Map
        Interactive map object
    """
    # Calculate center
    center_lat = (df['STATION_LATITUDE_DEGREE'].min() + df['STATION_LATITUDE_DEGREE'].max()) / 2
    center_lon = (df['STATION_LONGITUDE_DEGREE'].min() + df['STATION_LONGITUDE_DEGREE'].max()) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain').add_to(m)
    
    # Create colormap for PGA
    pga_min = df['PGA_CM/S^2'].min()
    pga_max = df['PGA_CM/S^2'].max()
    colormap = cm.LinearColormap(
        colors=['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4'],
        vmin=pga_min,
        vmax=pga_max,
        caption='PGA (cm/s²)'
    )
    colormap.add_to(m)
    
    # Add stations
    for _, row in df.iterrows():
        color = colormap(row['PGA_CM/S^2'])
        
        folium.CircleMarker(
            location=[row['STATION_LATITUDE_DEGREE'], row['STATION_LONGITUDE_DEGREE']],
            radius=8,
            color='black',
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=folium.Popup(
                f"<b>{row['STATION_CODE']}</b><br>"
                f"PGA: {row['PGA_CM/S^2']:.2f} cm/s²<br>"
                f"Lat: {row['STATION_LATITUDE_DEGREE']:.4f}<br>"
                f"Lon: {row['STATION_LONGITUDE_DEGREE']:.4f}",
                max_width=200
            ),
            tooltip=f"{row['STATION_CODE']}: {row['PGA_CM/S^2']:.1f} cm/s²"
        ).add_to(m)
        
        # Add station label (always visible)
        folium.Marker(
            location=[row['STATION_LATITUDE_DEGREE'], row['STATION_LONGITUDE_DEGREE']],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size: 10px; 
                    font-weight: bold; 
                    color: white; 
                    text-shadow: -1px -1px 0 black, 1px -1px 0 black, -1px 1px 0 black, 1px 1px 0 black;
                    white-space: nowrap;
                    transform: translate(10px, -5px);
                ">{row['STATION_CODE']}</div>
            """)
        ).add_to(m)
    
    # Add epicenter
    folium.Marker(
        location=[event_lat, event_lon],
        icon=folium.Icon(color='red', icon='star', prefix='fa'),
        popup=folium.Popup("<b>Epicenter</b>", max_width=150),
        tooltip="Epicenter"
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control
    plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    
    # Save if output path provided
    if output_path is not None:
        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
        m.save(str(output_path))
        print(f"Saved: {output_path}")
    
    return m
 
 
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