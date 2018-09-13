try:
    from main import Suma, Neuro2Radio, Pre, Volreg, Scale, Vol2Surf, Surf2Vol, RoiTemplates, RoiSurfData, MriSurfSmooth, MriFFT, HotT2Test, fitErrorEllipse, combineHarmonics, applyFitErrorEllipse, realImagSplit, graphRois
    from utils import BidsOrganizer, HardCreate
except ImportError:
    from .main import Suma, Neuro2Radio, Pre, Volreg, Scale, Vol2Surf, Surf2Vol, RoiTemplates, RoiSurfData, MriSurfSmooth, MriFFT, HotT2Test, fitErrorEllipse, combineHarmonics, applyFitErrorEllipse, realImagSplit, graphRois
    from .utils import BidsOrganizer, HardCreate


