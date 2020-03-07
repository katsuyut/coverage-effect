import warnings
from functools import reduce
import math
import re
import pickle
import numpy as np
from scipy.spatial.qhull import QhullError
from ase import Atoms
from ase.build import rotate
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.local_env import VoronoiNN

from collections import OrderedDict, Iterable, Mapping


def make_slabs_from_bulk_atoms(atoms, miller_indices,
                               slab_generator_settings, get_slab_settings):
    '''
    Use pymatgen to enumerate the slabs from a bulk.
    Args:
        atoms                   The `ase.Atoms` object of the bulk that you
                                want to make slabs out of
        miller_indices          A 3-tuple of integers containing the three
                                Miller indices of the slab[s] you want to
                                make.
        slab_generator_settings A dictionary containing the settings to be
                                passed to pymatgen's `SpaceGroupAnalyzer`
                                class.
        get_slab_settings       A dictionary containing the settings to be
                                ppassed to the `get_slab` method of
                                pymatgen's `SpaceGroupAnalyzer` class.
    Returns:
        slabs   A list of the slabs in the form of pymatgen.Structure
                objects. Note that there may be multiple slabs because
                of different shifts/terminations.
    '''
    # Get rid of the `miller_index` argument, which is superceded by the
    # `miller_indices` argument.
    try:
        slab_generator_settings = unfreeze_dict(slab_generator_settings)
        slab_generator_settings.pop('miller_index')
        warnings.warn('You passed a `miller_index` object into the '
                      '`slab_generator_settings` argument for the '
                      '`make_slabs_from_bulk_atoms` function. By design, '
                      'this function will instead use the explicit '
                      'argument, `miller_indices`.', SyntaxWarning)
    except KeyError:
        pass

    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=0.1)
    struct_stdrd = sga.get_conventional_standard_structure()
    slab_gen = SlabGenerator(initial_structure=struct_stdrd,
                             miller_index=miller_indices,
                             **slab_generator_settings)
    slabs = slab_gen.get_slabs(**get_slab_settings)
    return slabs


def unfreeze_dict(frozen_dict):
    '''
    Recursive function to turn a Luigi frozen dictionary into an ordered dictionary,
    along with all of the branches.
    Arg:
        frozen_dict     Instance of a luigi.parameter._FrozenOrderedDict
    Returns:
        dict_   Ordered dictionary
    '''
    # If the argument is a dictionary, then unfreeze it
    if isinstance(frozen_dict, Mapping):
        unfrozen_dict = OrderedDict(frozen_dict)

        # Recur
        for key, value in unfrozen_dict.items():
            unfrozen_dict[key] = unfreeze_dict(value)

    # Recur on the object if it's a tuple
    elif isinstance(frozen_dict, tuple):
        unfrozen_dict = tuple(unfreeze_dict(element)
                              for element in frozen_dict)

    # Recur on the object if it's a mutable iterable
    elif isinstance(frozen_dict, Iterable) and not isinstance(frozen_dict, str):
        unfrozen_dict = frozen_dict
        for i, element in enumerate(unfrozen_dict):
            unfrozen_dict[i] = unfreeze_dict(element)

    # If the argument is neither mappable nor iterable, we'rpe probably at a leaf
    else:
        unfrozen_dict = frozen_dict

    return unfrozen_dict


def slab_settings():
    '''
    The default settings we use to enumerate slabs, along with the subsequent
    DFT settings. The 'slab_generator_settings' are passed to the
    `SlabGenerator` class in pymatgen, and the `get_slab_settings` are passed
    to the `get_slab` method of that class.
    '''
    slab_settings = OrderedDict(max_miller=2,
                                max_atoms=80,
                                vasp=OrderedDict(ibrion=2,
                                                 nsw=100,
                                                 isif=0,
                                                 isym=0,
                                                 kpts=(4, 4, 1),
                                                 lreal='Auto',
                                                 ediffg=-0.03,
                                                 encut=350.,
                                                 pp_version=pp_version(),
                                                 **xc_settings('pbesol')),
                                slab_generator_settings=OrderedDict(min_slab_size=7.,
                                                                    min_vacuum_size=20.,
                                                                    lll_reduce=False,
                                                                    center_slab=True,
                                                                    primitive=True,
                                                                    max_normal_search=1),
                                get_slab_settings=OrderedDict(tol=0.3,
                                                              bonds=None,
                                                              max_broken_bonds=0,
                                                              symmetrize=False))
    return slab_settings
