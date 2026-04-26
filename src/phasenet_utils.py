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

def create_obspy_stream_from_dataframe(df_station, station_code, sampling_rate):
    """
    Create ObsPy Stream directly from station DataFrame.
    
    Parameters
    ----------
    df_station : pd.DataFrame
        DataFrame with columns ['file', 'sample', 'acceleration']
        Contains 3 files (components) for one station
    station_code : str
        Station code
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    stream : obspy.Stream or None
        Stream with 3 traces (Z, N, E) or None if incomplete
    component_names : tuple or None
        (Z_name, N_name, E_name) or None if incomplete
    """
    from obspy import Stream, Trace
    
    # Get available components
    files = df_station['file'].unique()
    components = {get_component_from_filename(f): f for f in files}
    available = set(components.keys())
    
    # Determine vertical component
    if 'HNZ' in available:
        Z_name = 'HNZ'
    elif 'HGZ' in available:
        Z_name = 'HGZ'
    else:
        return None, None
    
    # Determine horizontal components
    if 'HNN' in available and 'HNE' in available:
        N_name, E_name = 'HNN', 'HNE'
    elif 'HGN' in available and 'HGE' in available:
        N_name, E_name = 'HGN', 'HGE'
    elif 'HN1' in available and 'HN2' in available:
        N_name, E_name = 'HN1', 'HN2'
    else:
        return None, None
    
    # Create Stream
    stream = Stream()
    
    for comp_name in [Z_name, N_name, E_name]:
        file_name = components[comp_name]
        df_comp = df_station[df_station['file'] == file_name].sort_values('sample')
        
        trace = Trace(data=df_comp['acceleration'].values)
        trace.stats.sampling_rate = sampling_rate
        trace.stats.station = station_code
        trace.stats.channel = comp_name
        
        stream.append(trace)
    
    return stream, (Z_name, N_name, E_name)