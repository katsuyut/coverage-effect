from MAInit import *
from MACalc import *

elements = ['Cu', 'Pt', 'Ag', 'Pd', 'Au', 'Ni', 'Al', 'Rh']
# xcs = ['RPBE', 'RPBE-D2', 'vdW-DF', 'optB88-vdW', 'vdW-DF2', 'BEEF-vdW']
xcs = ['RPBE']
faces = ['100', '111', '110', '211']
unitlengths = [2, 3]
adsorbate = 'CO'
# maxmole = []

for ele in elements:
    bare = make_baresurface(ele)
    for xc in xcs:
        bare.calc_LC(xc)
        for face in faces:
            for unit in unitlengths:
                bare.make_surface_pymatgen(face, unit, 4)

### other than 221
faces = ['100', '111', '110']
for ele in elements:
    for xc in xcs:
        for face in faces:
            for unit in unitlengths:
                name = ele + '_' + face + '_u' + str(unit) + '_' + xc
                adsname = adsorbate + '_' + xc
                adsorbed = make_adsorbed_surface(name, adsname)
                if unit == 2:
                    molenum = 4
                else:
                    molenum = 1
                adsorbed.make_surface(maxmole=molenum, mindist=2.5)
                adsorbed.write_trajectory()

### for 211
faces = ['211']
for ele in elements:
    for xc in xcs:
        for face in ['211']:
            name = ele + '_' + face + '_u' + str(2) + '_' + xc
            adsname = adsorbate + '_' + xc
            adsorbed = make_adsorbed_surface(name, adsname)
            if unit == 2:
                molenum = 4
            else:
                molenum = 1
            adsorbed.make_surface(maxmole=molenum, mindist=2.5)
            adsorbed.write_trajectory()