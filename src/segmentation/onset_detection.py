# src/segmentation/onset_detection.py

"""
Onset detection using AR-AIC method.

Functions for detecting P and S wave arrivals using the Autoregressive-Akaike
Information Criterion (AR-AIC) method implemented in ObsPy.

References
----------
Leonard, M., & Kennett, B. L. N. (1999). Multi-component autoregressive 
techniques for the analysis of seismograms. Physics of the Earth and 
Planetary Interiors, 113(1-4), 247-263.
"""

import numpy as np
import pandas as pd
from obspy.signal.trigger import ar_pick

