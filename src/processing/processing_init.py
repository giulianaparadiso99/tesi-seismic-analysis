"""
Processing module for seismic signal analysis.

Provides signal conversion utilities for transforming long-format DataFrames
into nested dictionary structures optimized for onset detection and moment
scaling analysis.

Submodules:
    signal_conversion - Signal format conversion and validation functions
"""

from .signal_conversion import (
    add_time_columns,
    get_station_from_filename,
    get_component_from_filename,
    convert_signals_to_dict,
    get_signal_for_station,
    validate_signals_dict,
    expand_to_component_level
)

__all__ = [
    'add_time_columns',
    'get_station_from_filename',
    'get_component_from_filename',
    'convert_signals_to_dict',
    'get_signal_for_station',
    'validate_signals_dict',
    'expand_to_component_level',
]