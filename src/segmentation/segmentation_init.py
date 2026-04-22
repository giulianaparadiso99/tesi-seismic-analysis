

# Window Detection
from .theoretical_arrivals import (
    extract_crustal_velocities,
    add_crustal_velocities,
    add_theoretical_arrivals,
    calculate_search_windows,
    calculate_adaptive_windows
)

from .onset_detection import (
    detect_onsets_ar_windowed,
    detect_coda_start,
    detect_coda_start_all_methods,
    add_coda_onsets_to_dataframe,
    compute_coda_method_statistics
)

from .window_segmentation import (
    segment_signal_into_windows,
    segment_all_signals
)

from .window_validation import (
    check_pga_in_s_wave,
    check_monotonicity_station,
    check_snr,
    quality_control_all_stations,
    print_quality_control_summary,
    print_detailed_failures,
    analyze_monotonicity_violations,
    print_violation_summary,
    plot_monotonicity_analysis,
    analyze_residuals_vs_violations
)