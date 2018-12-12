try:
    from main import get_file_list, run_suma, neuro_to_radio, slice_timing, vol_reg, scale_detrend, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, subset_rois, roi_get_data, subject_analysis, group_analysis, whole_group, fft_analysis, vector_projection, hotelling_t2, fit_error_ellipse, read, write
    from utils import bids_organizer, hard_create
except ImportError:
    from .main import get_file_list, run_suma, neuro_to_radio, slice_timing, vol_reg, scale_detrend, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, subset_rois, roi_get_data, subject_analysis, group_analysis, whole_group, fft_analysis, vector_projection, hotelling_t2, fit_error_ellipse, read, write
    from .utils import bids_organizer, hard_create

