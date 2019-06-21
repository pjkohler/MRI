try:
    from main import get_file_list, run_suma, suma_make_fs_spec, neuro_to_radio, slice_timing, vol_reg, scale_detrend, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, roi_subsets, roi_get_data, subject_analysis, group_analyze, group_compare, fft_analysis, vector_projection, hotelling_t2, fit_error_ellipse, read, write
    from utils import bids_organizer, hard_create, bids_lookup
except ImportError:
    from .main import get_file_list, run_suma, suma_make_fs_spec, neuro_to_radio, slice_timing, vol_reg, scale_detrend, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, roi_subsets, roi_get_data, subject_analysis, group_analyze, group_compare, fft_analysis, vector_projection, hotelling_t2, fit_error_ellipse, read, write
    from .utils import bids_organizer, hard_create, bids_lookup

