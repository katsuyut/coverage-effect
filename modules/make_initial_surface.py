from MAInit import *
from MACalc import *

elements = ['Cu', 'Pt']
xcs = ['RPBE', 'RPBE-D2', 'vdW-DF', 'optB88-vdW', 'vdW-DF2', 'BEEF-vdW']
faces = ['100', '111']
unitlengths = [1,2,3,4]
adsorbate = 'CO'
# maxmole = []

for ele in elements:
    bare = make_baresurface(ele)
    for xc in xcs:
        bare.calc_LC(xc)
        for face in faces:
            for unit in unitlengths:
                bare.make_surface_pymatgen(face, unit, 4)

for ele in elements:
    for xc in xcs:
        for face in faces:
            for unit in unitlengths:
                name = ele + '_' + face + '_' + 'u' + str(unit) + '_' + xc
                adsname = adsorbate + '_' + xc
                adsorbed = make_adsorbed_surface(name, adsname)
                adsorbed.make_surface(maxmole=4, mindist=2.5)
                adsorbed.write_trajectory()
