"""
Preprocessing module for seismic signals.

Includes data loading from .ASC files, metadata cleaning, signal preprocessing
(baseline correction, filtering, normalization), and integration 
(acceleration → velocity → displacement).

Modules:
    io - Data loading from .ASC archive
    cleaning_metadata - Metadata preprocessing
    cleaning_signals - Signal preprocessing
    signals_integration - Integration (ObsPy-compatible)
"""

from .io import (
    build_metadata,
    build_accelerations,
    build_dataframes,
)

from .cleaning_metadata import clean_metadata

from .cleaning_signals import (
    preprocess_signals,
    validate_preprocessing
)

from .signals_integration import (
    integrate_to_velocity,
    integrate_to_displacement,
    validate_integration
)

from .signal_conversion import (
    get_station_from_filename,
    get_component_from_filename,
    convert_signals_to_dict,
    get_signal_for_station,
    validate_signals_dict
)

__all__ = [
    # I/O
    'build_metadata',
    'build_accelerations',
    'build_dataframes',
    # Metadata
    'clean_metadata',
    # Signals
    'preprocess_signals',
    'validate_preprocessing',
    # Integration
    'integrate_to_velocity',
    'integrate_to_displacement',
    'validate_integration',
]
