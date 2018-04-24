try:
	from main import Suma, Neuro2Radio, Pre, Volreg, Scale, Vol2Surf, Surf2Vol, RoiTemplates, RoiSurfData, MriFFT, HotT2Test
	from utils import BidsOrganizer, BidsLinks
except ImportError:
	from .main import Suma, Neuro2Radio, Pre, Volreg, Scale, Vol2Surf, Surf2Vol, RoiTemplates, RoiSurfData, MriFFT, HotT2Test
	from .utils import BidsOrganizer, BidsLinks



