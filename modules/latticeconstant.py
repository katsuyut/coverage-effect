from MACalc import *

mpids = ['mp-30', # Cu
         'mp-81', # Au
         'mp-124', # Ag
         'mp-23', # Ni
         'mp-2', # Pd
         'mp-126', # Pt
         'mp-79', # Zn
         'mp-134', # Al
         'mp-74', # Rh
         'mp-33', # Ru
         'mp-1941', # NiGa
         'mp-1219908', # PdPt
         'mp-922', # CoPt3
         ]

xcs = ['RPBE']
for mpid in mpids:
    for xc in xcs:
        get_equiblium_bulk(mpid, xc)

xcs = ['RPBE-D2', 'vdW-DF', 'optB88-vdW', 'vdW-DF2', 'BEEF-vdW']
for mpid in ['mp-30']:
    for xc in xcs:
        get_equiblium_bulk(mpid, xc)