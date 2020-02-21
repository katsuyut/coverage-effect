from MACalc import *

Cu = make_baresurface('Cu')
Cu.calcLC('RPBE', env='local')
Cu.make_surface_pymatgen('100', 2, 4)