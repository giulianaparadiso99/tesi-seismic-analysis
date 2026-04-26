from scipy.signal import resample
from obspy import Stream, Trace
import numpy as np

def get_station_from_filename(filename):
    """Extract station code from file name."""
    parts = filename.split('.')
    return parts[1] if len(parts) > 1 else filename

def get_component_from_filename(filename):
    """Extract component code from file name."""
    parts = filename.split('.')
    return parts[3] if len(parts) > 3 else filename

def build_component_dict(df_station):
    """Build dictionary of components from station DataFrame."""
    components = {}
    for file_name in df_station['file'].unique():
        component = get_component_from_filename(file_name)
        df_comp = df_station[df_station['file'] == file_name].sort_values('sample')
        components[component] = df_comp['acceleration'].values
    return components

def map_components_to_zne(components):
    """Map available components to Z, N, E standard orientation."""
    available = set(components.keys())
    
    # Vertical
    if 'HNZ' in available:
        Z, Z_name = components['HNZ'], 'HNZ'
    elif 'HGZ' in available:
        Z, Z_name = components['HGZ'], 'HGZ'
    else:
        return None, None, None, None, None, None
    
    # Horizontal
    if 'HNN' in available and 'HNE' in available:
        N, E = components['HNN'], components['HNE']
        N_name, E_name = 'HNN', 'HNE'
    elif 'HGN' in available and 'HGE' in available:
        N, E = components['HGN'], components['HGE']
        N_name, E_name = 'HGN', 'HGE'
    elif 'HN1' in available and 'HN2' in available:
        N, E = components['HN1'], components['HN2']
        N_name, E_name = 'HN1', 'HN2'
    else:
        return None, None, None, None, None, None
    
    return Z, N, E, Z_name, N_name, E_name

def create_obspy_stream(Z, N, E, Z_name, N_name, E_name, station_code, sampling_rate):
    """Create ObsPy Stream from component arrays."""
    stream = Stream()
    
    trace_z = Trace(data=Z)
    trace_z.stats.sampling_rate = sampling_rate
    trace_z.stats.station = station_code
    trace_z.stats.channel = Z_name
    stream.append(trace_z)
    
    trace_n = Trace(data=N)
    trace_n.stats.sampling_rate = sampling_rate
    trace_n.stats.station = station_code
    trace_n.stats.channel = N_name
    stream.append(trace_n)
    
    trace_e = Trace(data=E)
    trace_e.stats.sampling_rate = sampling_rate
    trace_e.stats.station = station_code
    trace_e.stats.channel = E_name
    stream.append(trace_e)
    
    return stream

def resample_components(Z, N, E, n_samples_target):
    """Resample three components to target number of samples."""
    Z_resampled = resample(Z, n_samples_target)
    N_resampled = resample(N, n_samples_target)
    E_resampled = resample(E, n_samples_target)
    return Z_resampled, N_resampled, E_resampled