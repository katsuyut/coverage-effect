from MAUtil import *

env = 'spacom'
files = os.listdir('./')
filenames = []

for i in range(len(files)):
    if '.traj' in files[i]:
        if '__' not in files[i]:
            if 'all' not in files[i]:
                atoms = query(files[i])
                ene = atoms.get_potential_energy()
                force = np.max(np.abs(atoms.get_forces()))
                print(files[i], ene, force, force < 0.03)



