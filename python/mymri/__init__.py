try:
    from main import get_data_files, run_suma, neuro_to_radio, pre_slice, pre_volreg, pre_scale, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, subset_rois, roi_get_data, subject_analysis, roi_group, whole_group, fft_analysis, hotelling_t2, fit_error_ellipse, read, write
    from utils import bids_organizer, hard_create
except ImportError:
    from .main import get_data_files, run_suma, neuro_to_radio, pre_slice, pre_volreg, pre_scale, vol_to_surf, surf_to_vol, surf_smooth, roi_templates, subset_rois, roi_get_data, subject_analysis, roi_group, whole_group, fft_analysis, hotelling_t2, fit_error_ellipse, read, write
    from .utils import bids_organizer, hard_create

