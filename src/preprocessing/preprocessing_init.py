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

from .cleaning_metadata import clean_metadata

from .cleaning_signals import (
    preprocess_signals,
    validate_preprocessing
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
