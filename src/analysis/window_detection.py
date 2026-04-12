"""
window_detection.py
-------------------
Functions for identifying temporal windows in seismic signals to separate
different dynamical regimes (pre-arrival, P-wave, S-wave, coda).

The module provides four complementary approaches:
    1. Visual inspection - Interactive plotting for manual identification
    2. PGA-based - Heuristic windows relative to Peak Ground Acceleration
    3. STA/LTA - Automatic onset detection using Short-Term/Long-Term Average (ObsPy)
    4. Combined wrapper - Applies all methods and generates comparison report

Main differences between approaches:
    - Visual: Manual but most accurate for complex signals
    - PGA-based: Fast, works well for clear events
    - STA/LTA: Industry standard, robust for noisy data (ObsPy implementation)
    - Combined: Comprehensive analysis with multiple estimates

Implementation Notes:
    - STA/LTA functions use ObsPy's optimized C-backend implementation
      (Beyreuther et al., 2010) for better performance and reliability
    - All other functions maintain custom implementations for flexibility

Citation:
    Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y., & 
    Wassermann, J. (2010). ObsPy: A Python toolbox for seismology. 
    Seismological Research Letters, 81(3), 530-533.

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

# ObsPy imports for STA/LTA
try:
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    print("Warning: ObsPy not available. STA/LTA will use fallback implementation.")
    print("Install ObsPy for better performance: pip install obspy")

from src.visualization.plot_settings import set_plot_style
colors = set_plot_style()

