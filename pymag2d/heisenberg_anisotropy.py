from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPSOCSet
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator, CollinearMagneticStructureAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Chgcar, Oszicar, Outcar, Potcar
from pymatgen.command_line.bader_caller import bader_analysis_from_objects, bader_analysis_from_path
from pymatgen.io.ase import AseAtomsAdaptor
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, FrozenJobErrorHandler,\
    MeshSymmetryErrorHandler, PositiveEnergyErrorHandler, StdErrHandler, NonConvergingErrorHandler, PotimErrorHandler 
from custodian.vasp.jobs import VaspJob
from custodian.vasp.validators import VasprunXMLValidator
import sys
import os
from shutil import copyfile
import datetime
from time import time, sleep
from ase.io import read, write
from ase.build import make_supercell, sort
import numpy as np
from sympy import Symbol, linsolve
import math
from pickle import load, dump
from misc import *


magnetic_list_gl = [Element('Co'), Element('Cr'), Element('Fe'), Element('Mn'), Element('Mo'), Element('Ni'), Element('V'), Element('W')]

LDAUJ_dict = {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0}
LDAUU_dict = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}
LDAUL_dict = {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2}

relx_dict_gl = {'ALGO': 'Fast', 'ISMEAR': 0, 'SIGMA': 0.01, 'EDIFF': 1E-4, 'EDIFFG': -0.01, 
'KPAR': 2,  'NCORE': 1, 'NSIM': 4, 'LCHARG': False, 'ICHARG': 2, 'LREAL': False,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'LWAVE': False,
'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4,
'ISIF': 3, 'IBRION': 3, 'POTIM': 0, 'IOPT': 3, 'LTWODIM': True}

relx_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
    FrozenJobErrorHandler(timeout=900), MeshSymmetryErrorHandler(), PositiveEnergyErrorHandler(),
    StdErrHandler(), NonConvergingErrorHandler(nionic_steps=5), PotimErrorHandler(dE_threshold=0.5)]

stat_dict_gl = {'ISMEAR': 0, 'EDIFF': 1E-6, 'KPAR': 2, 'NCORE': 1, 'NSIM': 4, 'LORBMOM': True, 'LAECHG': True, 'LREAL': False,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'NELMIN': 6, 'NELM': 500, 'LVHAR': False, 'SIGMA': 0.01,
'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4, 'LCHARG': True, 'LWAVE': True, 'ISYM': -1, 'LVTOT': False}

stat_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
    FrozenJobErrorHandler(timeout=3600), MeshSymmetryErrorHandler(), PositiveEnergyErrorHandler(), StdErrHandler()]

validator = [VasprunXMLValidator()]

dipole_dict = {'DIPOL': [0.5, 0.5, 0.5], 'IDIPOL': 3, 'LDIPOL': True}



def relx_gen(vasp_cmd_std,struct_file,randomise_cmd=False,user_incar_settings_relx={},LDAUJs={},LDAUUs={},LDAULs={},
    dipole_correction=False,kpt_den_relx=72,xc='PBE',pot='PBE_54',potcar_provided={},mag_species=[],max_errors=20):

    cmd = vasp_cmd_std.split()[:]

    root_path = os.getcwd()

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    if LDAUJs:
        LDAUJ_dict.update(LDAUJs)
    
    if LDAUUs:
        LDAUU_dict.update(LDAUUs)

    if LDAULs:
        LDAUL_dict.update(LDAULs)

    relx_dict = relx_dict_gl.copy()
    if dipole_correction:
        relx_dict.update(dipole_dict)

    if xc=='PBE':
        pot = 'PBE_54'
    elif xc=='LDA':
        pot = 'LDA_54'
    elif xc=='SCAN':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
    elif xc=='R2SCAN':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'R2SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
    elif xc=='SCAN+RVV10':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        relx_dict['LUSE_VDW'] = True
        relx_dict['BPARAM'] = 6.3
        relx_dict['CPARAM'] = 0.0093
    elif xc=='R2SCAN+RVV10':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'R2SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        relx_dict['LUSE_VDW'] = True
        relx_dict['BPARAM'] = 6.3
        relx_dict['CPARAM'] = 0.0093
    elif xc=='PBEsol':
        pot = 'PBE_54'
        relx_dict['GGA'] = 'PS'

    relx_dict.update(user_incar_settings_relx)

    cell = read(struct_file)
    c = cell.cell.cellpar()[2]

    for i in range(len(cell)):
        if cell[i].z > c*0.75:
            cell[i].z = cell[i].z - c

    cell.center(12.5,2)
    ase_adopt = AseAtomsAdaptor()
    struct = ase_adopt.get_structure(sort(cell))  

    if not potcar_provided:
        potcar_provided = None

    relx_path = root_path+'/initial_relx'

    if os.path.exists(relx_path+'/running'):
        msg = 'the calculation at '+relx_path+' is being already handled by another stream, moving on'
        log(msg)
        return None

    clean = sanitize(relx_path)

    if not clean:

        relx = MPRelaxSet(struct,user_incar_settings=relx_dict,user_kpoints_settings={'reciprocal_density':kpt_den_relx},
            force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
        relx.write_input(relx_path)

        with open(relx_path+'/running','w+') as f:
            f.write('JOB RUNNING!')
        msg = 'raised the running flag at '+relx_path
        log(msg)

        if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
            copyfile(root_path+'/vdw_kernel.bindat',relx_path+'/vdw_kernel.bindat')
        try:
            try_struct = Structure.from_file(relx_path+'/CONTCAR.bk')
            try_struct.to(filename=relx_path+'/POSCAR')
            msg = 'copied backed up CONTCAR to POSCAR'
        except Exception as e:
            print(e)
            msg = 'no backed up CONTCAR found'
        log(msg)
        kpts = Kpoints.from_file(relx_path+'/KPOINTS')
        kpts.kpts[0][2] = 1
        kpts.write_file(relx_path+'/KPOINTS')

        if randomise_cmd:
            cmd_rand = cmd[:]
            cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
            job = [VaspJob(cmd_rand)]
        else:
            job = [VaspJob(cmd)]
        cust = Custodian(relx_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
            gzipped_output=False,checkpoint=False)
        msg = 'running relaxtion'
        log(msg)
        done = 0

        os.chdir(relx_path)
        for j in range(3):
            try:
                cust.run()
                done = 1
                sleep(10)
                break
            except:
                sleep(10)
                continue
        os.chdir(root_path)
        if os.path.exists(relx_path+'/running'):
            os.remove(relx_path+'/running')
        
        if done == 1:
            msg = 'relaxation job finished successfully'
            log(msg)
        else:
            msg = 'relaxation failed after several attempts, exiting, you might want to manually handle this one,'
            msg += 'and then restart this code'
            log(msg)
            return None

    
    return 0



def stat_gen(vasp_cmd_std,struct_file,randomise_cmd=False,user_incar_settings_stat={},LDAUJs={},LDAUUs={},LDAULs={},
    dipole_correction=False,kpt_den_stat=300,xc='PBE',pot='PBE_54',potcar_provided={},mag_species=[],max_errors=10):

    cmd = vasp_cmd_std.split()[:]

    root_path = os.getcwd()

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    if LDAUJs:
        LDAUJ_dict.update(LDAUJs)
    
    if LDAUUs:
        LDAUU_dict.update(LDAUUs)

    if LDAULs:
        LDAUL_dict.update(LDAULs)

    stat_dict = stat_dict_gl.copy()
    if dipole_correction:
        stat_dict.update(dipole_dict)

    if xc=='PBE':
        pot = 'PBE_54'
    elif xc=='LDA':
        pot = 'LDA_54'
    elif xc=='SCAN':
        pot = 'PBE_54'
        stat_dict['METAGGA'] = 'SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
    elif xc=='R2SCAN':
        pot = 'PBE_54'
        stat_dict['METAGGA'] = 'R2SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
    elif xc=='SCAN+RVV10':
        pot = 'PBE_54'
        stat_dict['METAGGA'] = 'SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
        stat_dict['LUSE_VDW'] = True
        stat_dict['BPARAM'] = 6.3
        stat_dict['CPARAM'] = 0.0093
    elif xc=='R2SCAN+RVV10':
        pot = 'PBE_54'
        stat_dict['METAGGA'] = 'R2SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
        stat_dict['LUSE_VDW'] = True
        stat_dict['BPARAM'] = 6.3
        stat_dict['CPARAM'] = 0.0093
    elif xc=='PBEsol':
        pot = 'PBE_54'
        stat_dict['GGA'] = 'PS'

    stat_dict.update(user_incar_settings_stat)

    cell = read(struct_file)
    c = cell.cell.cellpar()[2]

    for i in range(len(cell)):
        if cell[i].z > c*0.75:
            cell[i].z = cell[i].z - c

    cell.center(12.5,2)
    ase_adopt = AseAtomsAdaptor()
    struct = ase_adopt.get_structure(sort(cell))  

    if not potcar_provided:
        potcar_provided = None

    stat_path = root_path+'/initial_stat'

    if os.path.exists(stat_path+'/running'):
        msg = 'the calculation at '+stat_path+' is being already handled by another stream, moving on'
        log(msg)
        return None

    clean = sanitize(stat_path)

    if not clean:

        stat = MPStaticSet(struct,user_incar_settings=stat_dict,reciprocal_density=kpt_den_stat,
            force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
        stat.write_input(stat_path)

        with open(stat_path+'/running','w+') as f:
            f.write('JOB RUNNING!')
        msg = 'raised the running flag at '+stat_path
        log(msg)

        if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
            copyfile(root_path+'/vdw_kernel.bindat',relx_path+'/vdw_kernel.bindat')

        kpts = Kpoints.from_file(stat_path+'/KPOINTS')
        kpts.kpts[0][2] = 1
        kpts.write_file(stat_path+'/KPOINTS')

        if randomise_cmd:
            cmd_rand = cmd[:]
            cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
            job = [VaspJob(cmd_rand)]
        else:
            job = [VaspJob(cmd)]
        cust = Custodian(stat_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
            gzipped_output=False,checkpoint=False)
        msg = 'running static calculation'
        log(msg)
        done = 0

        os.chdir(stat_path)
        for j in range(3):
            try:
                cust.run()
                done = 1
                sleep(10)
                break
            except:
                sleep(10)
                continue
        os.chdir(root_path)
        if os.path.exists(stat_path+'/running'):
            os.remove(stat_path+'/running')
        
        if done == 1:
            msg = 'static job finished successfully'
            log(msg)
        else:
            msg = 'static run failed after several attempts, exiting, you might want to manually handle this one,'
            msg += 'and then restart this code'
            log(msg)
            return None


    return 0



def run_heisenberg(vasp_cmd_std,struct_file,rep_DFT=[2,2,1],max_neigh=3,strain=[],skip=False,exit_if_afm = False,
    randomise_cmd=False,relx=True,user_incar_settings_relx={},user_incar_settings_stat={},LDAUJs={},LDAUUs={},LDAULs={},
    dipole_correction=False,kpt_den_relx=72,kpt_den_stat=300,xc='PBE',pot='PBE_54',potcar_provided={},mag_species=[],default_magmoms=None,
    mag_prec = 0.1,enum_prec = 0.001,ltol=0.4,stol=0.6,atol=5,max_errors_relx=20,max_errors_stat=10):

    cmd = vasp_cmd_std.split()[:]

    root_path = os.getcwd()

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    if LDAUJs:
        LDAUJ_dict.update(LDAUJs)
    
    if LDAUUs:
        LDAUU_dict.update(LDAUUs)

    if LDAULs:
        LDAUL_dict.update(LDAULs)

    relx_dict = relx_dict_gl.copy()
    stat_dict = stat_dict_gl.copy()
    if dipole_correction:
        relx_dict.update(dipole_dict)
        stat_dict.update(dipole_dict)

    if xc=='PBE':
        pot = 'PBE_54'
    elif xc=='LDA':
        pot = 'LDA_54'
    elif xc=='SCAN':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        stat_dict['METAGGA'] = 'SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
    elif xc=='R2SCAN':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'R2SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        stat_dict['METAGGA'] = 'R2SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
    elif xc=='SCAN+RVV10':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        relx_dict['LUSE_VDW'] = True
        relx_dict['BPARAM'] = 6.3
        relx_dict['CPARAM'] = 0.0093
        stat_dict['METAGGA'] = 'SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
        stat_dict['LUSE_VDW'] = True
        stat_dict['BPARAM'] = 6.3
        stat_dict['CPARAM'] = 0.0093
    elif xc=='R2SCAN+RVV10':
        pot = 'PBE_54'
        relx_dict['METAGGA'] = 'R2SCAN'
        relx_dict['LMIXTAU'] = True
        relx_dict['LDAU'] = False
        relx_dict['ALGO'] = 'All'
        relx_dict['LUSE_VDW'] = True
        relx_dict['BPARAM'] = 6.3
        relx_dict['CPARAM'] = 0.0093
        stat_dict['METAGGA'] = 'R2SCAN'
        stat_dict['LMIXTAU'] = True
        stat_dict['LDAU'] = False
        stat_dict['ALGO'] = 'All'
        stat_dict['LUSE_VDW'] = True
        stat_dict['BPARAM'] = 6.3
        stat_dict['CPARAM'] = 0.0093
    elif xc=='PBEsol':
        pot = 'PBE_54'
        relx_dict['GGA'] = 'PS'
        stat_dict['GGA'] = 'PS'

    relx_dict.update(user_incar_settings_relx)
    stat_dict.update(user_incar_settings_stat)

    if not potcar_provided:
        potcar_provided = None

    cell = read(struct_file)
    c = cell.cell.cellpar()[2]

    for i in range(len(cell)):
        if cell[i].z > c*0.75:
            cell[i].z = cell[i].z - c

    cell.center(2.5,2)
    ase_adopt = AseAtomsAdaptor()
    struct = ase_adopt.get_structure(sort(cell))
    if strain:
        struct.apply_strain(strain)
        msg = 'the structure is being starined with '+str(strain)+', will set ISIF = 2'
        log(msg)

    mag_enum = MagneticStructureEnumerator(struct,default_magmoms=default_magmoms,
        transformation_kwargs={'symm_prec':mag_prec,'enum_precision_parameter':enum_prec})
    mag_enum_structs = []
    for mag_struct in mag_enum.ordered_structures:
        n = len(mag_struct)
        spins = [0]*n
        uneven_spins = False
        for j in range(n):
            try:
                spins[j] = mag_struct.species[j].spin
            except Exception:
                element = mag_struct[j].specie.element
                if element in magnetic_list:
                    uneven_spins = True
                    break
                else:
                    spins[j] = 0.0
        if uneven_spins:
            msg = '** a config has uneven spins, continuing without it'
            log(msg)
            continue
        mag_struct.add_site_property('magmom',spins)
        mag_struct.remove_spin()
        mag_struct.sort()
        mag_enum_structs.append(mag_struct)

    s1 = mag_enum_structs[0].copy()
    s1.make_supercell(rep_DFT)
    matcher = StructureMatcher(primitive_cell=False,attempt_supercell=True)

    mag_structs = []
    mag_structs_super = []
    spins_configs = []
    spins_configs_super = []
    count = 0

    for i in range(len(mag_enum_structs)):
        s_mag = mag_enum_structs[i].copy()
        if matcher.fit(s1,s_mag):
            mag_struct = matcher.get_s2_like_s1(s1,s_mag)
            spins = mag_struct.site_properties['magmom']
            mag_tot = np.sum(spins)
            if i>0 and mag_tot!=0:
                msg = '** a config has uneven spins, continuing without it'
                log(msg)
                continue
            mag_cell = ase_adopt.get_atoms(mag_struct,magmoms=spins)
            mag_cell.center(12.75,2)
            mag_struct = ase_adopt.get_structure(mag_cell)
            mag_struct.add_spin_by_site(spins)
            mag_struct.to(filename='POSCAR.config_'+str(count)+'.supercell.vasp')
            mag_structs_super.append(mag_struct)
            spins_configs_super.append(spins)
            mag_struct.remove_spin()
            mag_struct.add_site_property('magmom',spins)
            mag_struct_prim = mag_struct.get_primitive_structure(use_site_props=True)
            spins_prim = mag_struct_prim.site_properties['magmom']
            mag_struct_prim.remove_site_property('magmom')
            mag_struct_prim.add_spin_by_site(spins_prim)
            mag_struct_prim.to(filename='POSCAR.config_'+str(count)+'.vasp')
            mag_structs.append(mag_struct_prim)
            spins_configs.append(spins_prim)
            count += 1

    if skip:
        skip2 = [int(ind) for ind in skip]
        for i in range(len(skip)):
            ind = skip2[i]
            mag_structs.pop(ind)
            spins_configs.pop(ind)
            mag_structs_super.pop(ind)
            spins_configs_super.pop(ind)
            skip2 = [ind-1 for ind in skip2]
            msg = 'skipping config_'+str(ind)+' on user request, the remaining configs would be renumbered'
            log(msg)
            
    num_struct = len(mag_structs)
    if num_struct == 1:
        msg = '*** only one config could be generated, can not fit Hamiltonian, exiting,'
        msg +=' play with enum_prec and mag_prec and DFT_supercell_size to generate more configs or try out a new material'
        log(msg)
        return None
    elif num_struct == 2:
        msg = '** only two configs could be generated, only first nearest neighbor interaction can be included,'
        msg +=' play with enum_prec and mag_prec and DFT_supercell_size to generate more configs'
        log(msg)

    msg = 'total '+str(num_struct)+' configs generated'
    log(msg)

    num_atoms = []
    for struct in mag_structs:
        num_atoms.append(len(struct))
    lcm_atoms = np.lcm.reduce(num_atoms)   

    ortho_ab= (mag_enum.ordered_structures[0].lattice.gamma>88 and mag_enum.ordered_structures[0].lattice.gamma<92) and (mag_enum.ordered_structures[0].lattice.a/mag_enum.ordered_structures[0].lattice.b<0.9 or
        mag_enum.ordered_structures[0].lattice.a/mag_enum.ordered_structures[0].lattice.b>1.1)

    if xc=='LDA':
        pot = 'LDA_54'
    else:
        pot = 'PBE_54'

    if not relx:
        relx_dict['EDIFFG'] = -10.0
        msg = 'command detected for no relaxation, structures wont be relaxed, only a fake and fast relaxation will be performed'
        log(msg)

    if strain:
        relx_dict['ISIF'] = 2


    start_time_dft = time()
    energies_relx = []


    # relax the enumerated structures, no supercell 

    for i in range(num_struct):

        spins = spins_configs[i]
        struct_current = mag_structs[i].copy()
        factor = float(lcm_atoms)/len(struct_current)
        
        if factor!=int(factor):
            msg = '*** factor is float, '+str(factor)+', exiting'
            log(msg)
            return None

        relx_path = root_path+'/relaxations'+'/config_'+str(i)

        if os.path.exists(relx_path+'/running'):
            msg = 'the calculation at '+relx_path+' is being already handled by another stream, moving on'
            log(msg)
            continue

        clean = sanitize(relx_path)

        if not clean:

            relx = MPRelaxSet(struct_current,user_incar_settings=relx_dict,user_kpoints_settings={'reciprocal_density':kpt_den_relx},
                force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
            relx.write_input(relx_path)

            with open(relx_path+'/running','w+') as f:
                f.write('JOB RUNNING!')
            msg = 'raised the running flag at '+relx_path
            log(msg)

            if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
                copyfile(root_path+'/vdw_kernel.bindat',relx_path+'/vdw_kernel.bindat')
            try:
                try_struct = Structure.from_file(relx_path+'/CONTCAR.bk')
                try_struct.to(filename=relx_path+'/POSCAR')
                msg = 'copied backed up CONTCAR to POSCAR'
            except Exception as e:
                print(e)
                msg = 'no backed up CONTCAR found'
            log(msg)
            kpts = Kpoints.from_file(relx_path+'/KPOINTS')
            kpts.kpts[0][2] = 1
            kpts.write_file(relx_path+'/KPOINTS')

            if randomise_cmd:
                cmd_rand = cmd[:]
                cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                job = [VaspJob(cmd_rand)]
            else:
                job = [VaspJob(cmd)]
            cust = Custodian(relx_handlers,job,validator,max_errors=max_errors_relx,polling_time_step=5,monitor_freq=10,
                gzipped_output=False,checkpoint=False)
            msg = 'running relaxtion for config '+str(i)
            log(msg)
            done = 0

            os.chdir(relx_path)
            for j in range(3):
                try:
                    cust.run()
                    done = 1
                    sleep(10)
                    break
                except:
                    sleep(10)
                    continue
            os.chdir(root_path)
            if os.path.exists(relx_path+'/running'):
                os.remove(relx_path+'/running')
            
            if done == 1:
                msg = 'relaxation job finished successfully for config '+str(i)
                log(msg)
            else:
                msg = 'relaxation failed for config '+str(i)+' after several attempts, exiting, you might want to manually handle this one,'
                msg += 'and then restart this code'
                log(msg)
                return None

        run = Vasprun(relx_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        energy = float(run.final_energy)
        energy = energy*factor
        energies_relx.append(energy)

    msg = 'all relaxations have finished gracefully'
    log(msg)
    msg = 'the configuration wise relaxation energies are: '+str(energies_relx)
    log(msg)
    most_stable = np.argmin(energies_relx)
    msg = '### The most stable config = config_'+str(most_stable)
    log(msg)
    if exit_if_afm and most_stable!=0:
        msg = 'the ground state is AFM, not proceeding further'
        log(msg)
        return 'AFM'

    s_mag = Structure.from_file(root_path+'/relaxations'+'/config_'+str(most_stable)+'/CONTCAR')
    s1 = Structure.from_file(root_path+'/POSCAR.config_0.supercell.vasp')
    matcher = StructureMatcher(primitive_cell=False,attempt_supercell=True,ltol=ltol,stol=stol,angle_tol=atol)
    struct_ground_super = matcher.get_s2_like_s1(s1,s_mag)
    if struct_ground_super==None:
        msg = 'can not make supercell with the most stable relaxed structure, '
        msg += 'carefully check the relaxation results and play with ltol, stol and angle_tol'
        msg += ' or relax manually and run this code without relaxations, exiting'
        log(msg)
        return None

    mag_structs = []
    for i in range(num_struct):
        mag_struct = struct_ground_super.copy()
        mag_struct.add_spin_by_site(spins_configs_super[i])
        mag_structs.append(mag_struct)

    if most_stable<=max_neigh:
        mag_structs = mag_structs[:max_neigh+1]

    num_struct = len(mag_structs)


    # static run of the enumerated structures, relaxed with ground state mag config, with supercell

    for i in range(num_struct):

        spins = spins_configs[i]
        stat_struct = mag_structs[i].copy()

        stat_path = root_path+'/static_runs'+'/config_'+str(i)

        if os.path.exists(stat_path+'/running'):
            msg = 'the calculation at '+stat_path+' is being already handled by another stream, moving on'
            log(msg)
            continue

        clean = sanitize(stat_path)

        if not clean:

            stat = MPStaticSet(stat_struct,user_incar_settings=stat_dict,reciprocal_density=kpt_den_stat,
                force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
            stat.write_input(stat_path)

            with open(stat_path+'/running','w+') as f:
                f.write('JOB RUNNING!')
            msg = 'raised the running flag at '+stat_path
            log(msg)

            if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
                copyfile(root_path+'/vdw_kernel.bindat',stat_path+'/vdw_kernel.bindat')
            kpts = Kpoints.from_file(stat_path+'/KPOINTS')
            kpts.kpts[0][2] = 1
            kpts.write_file(stat_path+'/KPOINTS')

            if randomise_cmd:
                cmd_rand = cmd[:]
                cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                job = [VaspJob(cmd_rand)]
            else:
                job = [VaspJob(cmd)]
            cust = Custodian(stat_handlers,job,validator,max_errors=max_errors_stat,polling_time_step=5,monitor_freq=10,
                gzipped_output=False,checkpoint=False)
            msg = 'running static run for config '+str(i)
            log(msg)
            done = 0

            os.chdir(stat_path)
            for j in range(3):
                try:
                    cust.run()
                    done = 1
                    sleep(10)
                    break
                except:
                    sleep(10)
                    continue
            os.chdir(root_path)
            if os.path.exists(stat_path+'/running'):
                os.remove(stat_path+'/running')
        
            if done == 1:
                msg = 'static run finished successfully for config '+str(i)
                log(msg)
            else:
                msg = 'static run failed for config '+str(i)
                msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                log(msg)
                return None


    msg = 'all collinear static runs have finished gracefully'
    log(msg)
    end_time_dft = time()
    time_dft = np.around(end_time_dft - start_time_dft, 2)

    msg = 'collinear DFT energy calculations/check of all possible configurations took total '+str(time_dft)+' s'
    log(msg)

    return 0



def run_anisotropy(vasp_cmd_ncl,randomise_cmd=False,kpt_den_stat=300,xc='PBE',pot='PBE_54',potcar_provided={},mag_species=[],
    ediff=1E-6,icharg_scan=1,nelm=500,kpar=1,ncore=1,nsim=4,lcharg=False,lwave=False,rm_chgcar=True,rm_wavecar=True,max_errors=10):

    cmd_ncl = vasp_cmd_ncl.split()[:]

    root_path = os.getcwd()

    num_struct = len(os.listdir(root_path+'/static_runs'))

    struct_base = Structure.from_file(root_path+'/static_runs/config_0/POSCAR')

    ortho_ab= (struct_base.lattice.gamma>88 and struct_base.lattice.gamma<92) and (struct_base.lattice.a/struct_base.lattice.b<0.9 or
    struct_base.lattice.a/struct_base.lattice.b>1.1)

    if ortho_ab:
        saxes = [(1,0,0),(0,1,0),(0,0,1)]
        msg = 'found orthogonal a and b vectors, will perform noncollinear calculations for '+str(saxes)
        log(msg)
    else:
        saxes = [(1,0,0),(0,0,1)]
        msg = 'found non-orthogonal or orthogonal but equal a and b vectors, will perform noncollinear calculations for '+str(saxes)+ ' only'
        log(msg)

    start_time_dft = time()


    for i in range(num_struct):

        stat_path = root_path+'/static_runs'+'/config_'+str(i)

        for axis in saxes:
            
            mae_path = root_path+'/MAE/config_'+str(i)+'/'+str(axis).replace(' ','')

            if os.path.exists(mae_path+'/running'):
                msg = 'the calculation at '+mae_path+' is being already handled by another stream, moving on'
                log(msg)
                continue

            clean = sanitize(mae_path)
            
            if not clean:

                soc = MPSOCSet.from_prev_calc(stat_path,saxis=axis,nbands_factor=2,reciprocal_density=kpt_den_stat,
                    force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)                     
                soc.write_input(mae_path)

                with open(mae_path+'/running','w+') as f:
                    f.write('JOB RUNNING!')
                msg = 'raised the running flag at '+mae_path
                log(msg)

                inc = Incar.from_file(mae_path+'/INCAR')

                if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
                    copyfile(root_path+'/vdw_kernel.bindat',mae_path+'/vdw_kernel.bindat')
                replace_text(mae_path+'/INCAR','NELM = '+str(inc.as_dict()['NELM']),'NELM = '+str(nelm))
                replace_text(mae_path+'/INCAR','LCHARG = '+str(inc.as_dict()['LCHARG']),'LCHARG = '+str(lcharg))
                replace_text(mae_path+'/INCAR','LWAVE = '+str(inc.as_dict()['LWAVE']),'LWAVE = '+str(lwave))
                replace_text(mae_path+'/INCAR','LAECHG = '+str(inc.as_dict()['LAECHG']),'LAECHG = False')
                replace_text(mae_path+'/INCAR','NSIM = '+str(inc.as_dict()['NSIM']),'NSIM = '+str(nsim))
                replace_text(mae_path+'/INCAR','EDIFF = 1e-06','EDIFF = '+str(ediff))
                if 'SCAN' in xc:
                    replace_text(mae_path+'/INCAR','ICHARG = 11','ICHARG = '+str(icharg_scan))
                with open(mae_path+'/INCAR','a') as inc:
                    inc.write('\nKPAR = '+str(kpar)+'\nNCORE = '+str(ncore)+'\nNSIM = '+str(nsim))
                kpts = Kpoints.from_file(mae_path+'/KPOINTS')
                kpts.kpts[0][2] = 1
                kpts.write_file(mae_path+'/KPOINTS')

                try:
                    copyfile(stat_path+'/WAVECAR',mae_path+'/WAVECAR')
                except:
                    msg = '** no collinear WAVECAR found, still continuing'
                    log(msg)
                
                if randomise_cmd:
                    cmd_rand = cmd_ncl[:]
                    cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                    job = [VaspJob(cmd_rand)]
                else:
                    job = [VaspJob(cmd_ncl)]
                cust = Custodian(stat_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
                    gzipped_output=False,checkpoint=False)
                msg = 'running non-collinear run for config '+str(i)+' and direction '+str(axis)
                log(msg)
                done = 0

                os.chdir(mae_path)
                for j in range(3):
                    try:
                        cust.run()
                        done = 1
                        sleep(10)
                        break
                    except:
                        sleep(10)
                        continue
                os.chdir(root_path)
                if os.path.exists(mae_path+'/running'):
                    os.remove(mae_path+'/running')
            
                if done == 1:
                    msg = 'non-collinear run finished successfully for config '+str(i)+' and direction '+str(axis)
                    log(msg)
                else:
                    msg = 'non-collinear run failed for config '+str(i)+' and direction '+str(axis)
                    msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                    log(msg)
                    return None
                if os.path.exists(mae_path+'/CHGCAR') and rm_chgcar:
                    os.remove(mae_path+'/CHGCAR')
                if os.path.exists(mae_path+'/WAVECAR') and rm_wavecar:
                    os.remove(mae_path+'/WAVECAR')


    end_time_dft = time()
    time_dft = np.around(end_time_dft - start_time_dft, 2)
    msg = 'all non-collinear runs for anisotopies have finished gracefully'
    log(msg)

    msg = 'DFT energy calculations/check of all possible configurations took total '+str(time_dft)+' s'
    log(msg)

    return 0



def fit_heisenberg(with_anisotropy=True,max_neigh=3,mag_from='OSZICAR',mag_species=[],d_thresh=0.05):

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    root_path = os.getcwd()

    num_struct = len(os.listdir(root_path+'/static_runs'))

    struct_base = Structure.from_file(root_path+'/static_runs/config_0/POSCAR')

    ortho_ab= (struct_base.lattice.gamma>88 and struct_base.lattice.gamma<92) and (struct_base.lattice.a/struct_base.lattice.b<0.9 or
    struct_base.lattice.a/struct_base.lattice.b>1.1)

    if ortho_ab:
        saxes = [(1,0,0),(0,1,0),(0,0,1)]
        msg = 'found orthogonal a and b vectors, will perform noncollinear calculations for '+str(saxes)
        log(msg)
    else:
        saxes = [(1,0,0),(0,0,1)]
        msg = 'found non-orthogonal or orthogonal but equal a and b vectors, will perform noncollinear calculations for '+str(saxes)+ ' only'
        log(msg)

    msg = 'attempting to collect data and fit the Hamiltonian now'
    log(msg)


    num_neigh = min([max_neigh, num_struct-1])
    msg = 'total '+str(num_struct)+' valid FM/AFM configs have been detected, including '
    msg += str(num_neigh)+' nearest-neighbors in the fitting'
    log(msg)

    semifinal_list = []

    for i in range(num_struct):
        
        msg = 'checking vasp run status of config_'+str(i)+' static and non-collinear runs'
        log(msg)
        config_info = []
        stat_path = root_path+'/static_runs/config_'+str(i)
        struct = Structure.from_file(root_path+'/static_runs/config_'+str(i)+'/POSCAR')
        inc = Incar.from_file(root_path+'/static_runs/config_'+str(i)+'/INCAR')
        struct.add_spin_by_site(inc.as_dict()['MAGMOM'])
        run = Vasprun(stat_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        if not run.converged_electronic:
            msg = '*** static run have not converged for config_'+str(i)+', exiting'
            log(msg)
            if with_anisotropy:
                return [0]*20
            else:
                return [0]*5
        else:
            msg = 'found converged static run'
            log(msg)
            
        energy = float(run.final_energy)  
            
        config_info.append(i)
        config_info.append(struct)
        config_info.append(energy)

        if with_anisotropy:

            for axis in saxes:

                mae_path = root_path+'/MAE/config_'+str(i)+'/'+str(axis).replace(' ','')
                run = Vasprun(mae_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
                struct = Structure.from_file(mae_path+'/POSCAR')
                if not run.converged_electronic:
                    msg = '*** non-collinear run have not converged for config_'+str(i)+' and axis '+str(axis)+', exiting'
                    log(msg)
                    return [0]*20
                else:
                    msg = 'found converged non-collinear run'
                    log(msg)
                energy = float(run.final_energy)
                config_info.append(energy)
                if not ortho_ab and axis==(1,0,0):
                    config_info.append(energy)
        
        semifinal_list.append(config_info)

    semifinal_list = sorted(semifinal_list, key = lambda x : x[2])
    most_stable = semifinal_list[0][0]

    msg = '### The most stable config = config_'+str(most_stable)
    log(msg)

    if with_anisotropy:

        energies_ncl = semifinal_list[0][3:]
        EMA = np.argmin(energies_ncl)
        saxes = [(1,0,0),(0,1,0),(0,0,1)]

        msg = '### The easy magnetization axis (EMA) = '+str(saxes[EMA])
        log(msg)

        # analyzer = CollinearMagneticStructureAnalyzer(semifinal_list[0][1],overwrite_magmom_mode='replace_all_if_undefined',
        #     make_primitive=False)
        # num_mag_atoms = analyzer.number_of_magnetic_sites
        num_mag_atoms = count_mag_atoms(semifinal_list[0][1],magnetic_list)
        E_100_001 = (energies_ncl[0] - energies_ncl[2])/(num_mag_atoms)
        E_010_001 = (energies_ncl[1] - energies_ncl[2])/(num_mag_atoms)
        msg = '### magnetocrystalline anisotropic energies (MAE) are:'
        log(msg)
        msg = 'E[100]-E[001] = '+str(E_100_001*1e6)+' ueV/magnetic_atom'
        log(msg)
        msg = 'E[010]-E[001] = '+str(E_010_001*1e6)+' ueV/magnetic_atom'
        log(msg)


    for i in range(len(semifinal_list)):

        config = semifinal_list[i][0]
        stat_path = root_path+'/static_runs'+'/config_'+str(config)
        if mag_from=='Bader' and config==most_stable:
            if not os.path.exists(stat_path+'/bader.dat'):
                msg = 'starting bader analysis for config_'+str(config)
                log(msg)
                ba = bader_analysis_from_path(stat_path)
                msg = 'finished bader analysis successfully'
                log(msg)
                f = open(stat_path+'/bader.dat','wb')
                dump(ba,f)
                f.close()                         
            else:
                f = open(stat_path+'/bader.dat','rb')
                ba = load(f)
                f.close()
                msg = 'reading magmoms from bader file'
                log(msg)
            magmom_stable = max(ba['magmom'])
            S_stable = magmom_stable/2.0

        elif mag_from=='OSZICAR' and config==0:
            # osz = Oszicar(stat_path+'/OSZICAR')
            # config_magmom = float(osz.ionic_steps[-1]['mag'])
            config_magmom = read_mag_oszi(stat_path)
            # analyzer = CollinearMagneticStructureAnalyzer(semifinal_list[i][1],overwrite_magmom_mode='replace_all_if_undefined',
            #     make_primitive=False)
            # num_mag_atoms = analyzer.number_of_magnetic_sites
            num_mag_atoms = count_mag_atoms(semifinal_list[i][1],magnetic_list)
            magmom_stable = config_magmom/num_mag_atoms
            S_stable = magmom_stable/2.0

    E0 = Symbol('E0')
    J1 = Symbol('J1')
    J2 = Symbol('J2')
    J3 = Symbol('J3')
    J4 = Symbol('J4')
    if with_anisotropy:
        K1x = Symbol('K1x')
        K1y = Symbol('K1y')
        K1z = Symbol('K1z')
        K2x = Symbol('K2x')
        K2y = Symbol('K2y')
        K2z = Symbol('K2z')
        K3x = Symbol('K3x')
        K3y = Symbol('K3y')
        K3z = Symbol('K3z')
        K4x = Symbol('K4x')
        K4y = Symbol('K4y')
        K4z = Symbol('K4z')
        Ax = Symbol('Ax')
        Ay = Symbol('Ay')
        Az = Symbol('Az')

    kB = np.double(8.6173303e-5)

    fitted = False

    while num_neigh>0:
        
        final_list = semifinal_list[:(num_neigh+1)]

        num_config = len(final_list)
        eqn_set_iso = [0]*num_config
        if with_anisotropy:
            eqn_set_x = [0]*num_config
            eqn_set_y = [0]*num_config
            eqn_set_z = [0]*num_config
        CN1s = []
        CN2s = []
        CN3s = []
        CN4s = []

        for i in range(num_config):
            
            config = final_list[i][0]
            struct = final_list[i][1]
            energy_iso = final_list[i][2]
            if with_anisotropy:
                energies_ncl = final_list[i][3:]
            stat_path = root_path+'/static_runs'+'/config_'+str(config)
                   
            out = Outcar(stat_path+'/OUTCAR')
                
            sites_mag = []
            magmoms_mag = []
            magmoms_out = []
            for j in range(len(struct)):
                element = struct[j].specie.element
                if element in magnetic_list:
                    sign_magmom = np.sign(struct[j].specie.spin)
                    magmom = sign_magmom*magmom_stable
                    magmoms_mag.append(magmom)
                    sites_mag.append(struct[j])
                    magmoms_out.append(out.magnetization[j]['tot'])
            struct_mag = Structure.from_sites(sites_mag)
            struct_mag_out = Structure.from_sites(sites_mag)
            struct_mag.remove_spin()
            struct_mag.add_site_property('magmom',magmoms_mag)
            struct_mag_out.add_site_property('magmom',magmoms_out)
            N = len(struct_mag)
            msg = 'config_'+str(config)+' (only magnetic atoms) = '
            log(msg)
            log(struct_mag)
            msg = 'same config with magmoms from OUTCAR is printed below, make sure this does not deviate too much from above'
            log(msg)
            log(struct_mag_out)

            if np.linalg.norm(np.sign(magmoms_mag)-np.sign(magmoms_out))>1e-6:
                msg = 'seems the sign of magnetism has changed after DFT, exiting'
                log(msg)
                if with_anisotropy:
                    return [0]*20
                else:
                    return [0]*5
            
            ds = dist_neighbors(struct_mag,d_thresh=d_thresh)
            dr = ds[0]

            eqn_iso = E0 - energy_iso
            if with_anisotropy:
                eqn_x, eqn_y, eqn_z = energy_iso - energies_ncl[0], energy_iso - energies_ncl[1], energy_iso - energies_ncl[2]

            N1s = []
            N2s = []
            N3s = []
            N4s = []
            
            for j in range(N):
                site = j
                S_site = struct_mag.site_properties['magmom'][j]/2.0
                if num_config==2:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                elif num_config==3:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                elif num_config==4:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                    N3s = Nfinder(struct_mag,site,ds[3],dr)[0]
                elif num_config==5:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                    N3s = Nfinder(struct_mag,site,ds[3],dr)[0]
                    N4s = Nfinder(struct_mag,site,ds[4],dr)[0]
                
                for N1 in N1s:
                    S_N1 = struct_mag.site_properties['magmom'][N1]/2.0
                    eqn_iso += -0.5*J1*S_site*S_N1
                    if with_anisotropy:
                        eqn_x += -0.5*K1x*S_site*S_N1
                        eqn_y += -0.5*K1y*S_site*S_N1
                        eqn_z += -0.5*K1z*S_site*S_N1
                if N2s:
                    for N2 in N2s:
                        S_N2 = struct_mag.site_properties['magmom'][N2]/2.0
                        eqn_iso += -0.5*J2*S_site*S_N2
                        if with_anisotropy:
                            eqn_x += -0.5*K2x*S_site*S_N2
                            eqn_y += -0.5*K2y*S_site*S_N2
                            eqn_z += -0.5*K2z*S_site*S_N2
                if N3s:
                    for N3 in N3s:
                        S_N3 = struct_mag.site_properties['magmom'][N3]/2.0
                        eqn_iso += -0.5*J3*S_site*S_N3
                        if with_anisotropy:
                            eqn_x += -0.5*K3x*S_site*S_N3
                            eqn_y += -0.5*K3y*S_site*S_N3
                            eqn_z += -0.5*K3z*S_site*S_N3
                if N4s:
                    for N4 in N4s:
                        S_N4 = struct_mag.site_properties['magmom'][N4]/2.0
                        eqn_iso += -0.5*J4*S_site*S_N4
                        if with_anisotropy:
                            eqn_x += -0.5*K4x*S_site*S_N4
                            eqn_y += -0.5*K4y*S_site*S_N4
                            eqn_z += -0.5*K4z*S_site*S_N4
                if with_anisotropy:
                    eqn_x += -Ax*np.square(S_site)
                    eqn_y += -Ay*np.square(S_site)
                    eqn_z += -Az*np.square(S_site)
                CN1s.append(len(N1s))
                CN2s.append(len(N2s))
                CN3s.append(len(N3s))
                CN4s.append(len(N4s))

            eqn_set_iso[i] = eqn_iso
            if with_anisotropy:
                eqn_set_x[i] = eqn_x
                eqn_set_y[i] = eqn_y
                eqn_set_z[i] = eqn_z

            if config==most_stable:
                struct_mag_stable = struct_mag
                ds_stable = ds
                struct_stable = struct_mag

        msg = '### mu = '+str(magmom_stable)+' bohr magnetron/magnetic atom'
        log(msg)
                
        msg = 'eqns are:'
        log(msg)
        
        for eqn in eqn_set_iso:
            msg = str(eqn)+' = 0'
            log(msg)
        if with_anisotropy:
            for eqn in eqn_set_x:
                msg = str(eqn)+' = 0'
                log(msg)
            if ortho_ab:
                for eqn in eqn_set_y:
                    msg = str(eqn)+' = 0'
                    log(msg)        
            for eqn in eqn_set_z:
                msg = str(eqn)+' = 0'
                log(msg)        

        if num_config==2:
            soln_iso = linsolve(eqn_set_iso, E0, J1)
            if with_anisotropy:
                soln_x = linsolve(eqn_set_x, K1x, Ax)
                if ortho_ab:
                    soln_y = linsolve(eqn_set_y, K1y, Ay)
                soln_z = linsolve(eqn_set_z, K1z, Az)
        elif num_config==3:
            soln_iso = linsolve(eqn_set_iso, E0, J1, J2)
            if with_anisotropy:
                soln_x = linsolve(eqn_set_x, K1x, K2x, Ax)
                if ortho_ab:
                    soln_y= linsolve(eqn_set_y, K1y, K2y, Ay)
                soln_z = linsolve(eqn_set_z, K1z, K2z, Az)
        elif num_config==4:
            soln_iso = linsolve(eqn_set_iso, E0, J1, J2, J3)
            if with_anisotropy:
                soln_x = linsolve(eqn_set_x, K1x, K2x, K3x, Ax)
                if ortho_ab:
                    soln_y = linsolve(eqn_set_y, K1y, K2y, K3y, Ay)
                soln_z = linsolve(eqn_set_z, K1z, K2z, K3z, Az)
        elif num_config==5:
            soln_iso = linsolve(eqn_set_iso, E0, J1, J2, J3, J4)
            if with_anisotropy:
                soln_x = linsolve(eqn_set_x, K1x, K2x, K3x, K4x, Ax)
                if ortho_ab:
                    soln_y = linsolve(eqn_set_y, K1y, K2y, K3y, K4y, Ay)
                soln_z = linsolve(eqn_set_z, K1z, K2z, K3z, K4z, Az)
        soln_iso = list(soln_iso)
        if with_anisotropy:
            soln_x = list(soln_x)
            if ortho_ab:
                soln_y = list(soln_y)
            else:
                soln_y = [0]
            soln_z = list(soln_z)
        msg = 'the solutions are:'
        log(msg)
        log(soln_iso)
        if with_anisotropy:
            log(soln_x)
            if ortho_ab:
                log(soln_y)
            log(soln_z)

        try:
            if with_anisotropy:
                quant = (soln_iso and soln_x and soln_y and soln_z and np.max(np.abs(soln_iso[0]))<5e3 and np.max(np.abs(soln_x[0]))<5e3 and
                    np.max(np.abs(soln_y[0]))<5e3 and np.max(np.abs(soln_z[0]))<5e3)
            else:
                quant = (soln_iso and np.max(np.abs(soln_iso[0]))<5e3)

            if quant:
                fitted = True
                break
        except Exception as e:
            log(e)
            fitted = False

        if not fitted:
            num_neigh -= 1
            msg = 'looks like these set of equations are either not solvable or yielding unphysical values'
            log(msg)
            msg = 'reducing the number of included NNs to '+str(num_neigh)
            log(msg)

            
    if not fitted:
        msg = '*** could not fit the Hamiltonian after several tries, exiting'
        log(msg)
        if with_anisotropy:
            return [0]*20
        else:
            return [0]*5


    CN1 = np.mean(CN1s)
    CN2 = np.mean(CN2s)
    CN3 = np.mean(CN3s)
    CN4 = np.mean(CN4s)

    if with_anisotropy:
        if ortho_ab:
            msg = 'orthogonal a and b vectors found for the lattice, using the XYZ model'
            log(msg)
        else:
            soln_y = soln_x
            msg = 'non-orthogonal a and b vectors found for the lattice, using XXZ model'
            log(msg)

    if num_config==2:
        E0, J1 = soln_iso[0][0], soln_iso[0][1]
        J2, J3, J4 = 0, 0, 0
        if with_anisotropy:
            K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
            K2, K3, K4 = np.zeros(3), np.zeros(3), np.zeros(3)
            A = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        if with_anisotropy:
            msg = 'K1 = '+str(K1*1e3)+' meV/link'
            log(msg)
            msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
            log(msg)
        
    elif num_config==3:
        E0, J1, J2 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2]
        J3, J4 = 0, 0
        if with_anisotropy:
            K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
            K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
            K3, K4 = np.zeros(3), np.zeros(3)
            A =np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        if with_anisotropy:
            msg = 'K1 = '+str(K1*1e3)+' meV/link'
            log(msg)
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        if with_anisotropy:
            msg = 'K2 = '+str(K2*1e3)+' meV/link'
            log(msg)
            msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
            log(msg)
        
    elif num_config==4:
        E0, J1, J2, J3 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2], soln_iso[0][3]
        J4 = 0
        if with_anisotropy:
            K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
            K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
            K3 = np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
            K4 = np.zeros(3)
            A = np.array([soln_x[0][3], soln_y[0][3], soln_z[0][3]])
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        if with_anisotropy:
            msg = 'K1 = '+str(K1*1e3)+' meV/link'
            log(msg)
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        if with_anisotropy:
            msg = 'K2 = '+str(K2*1e3)+' meV/link'
            log(msg)
        msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
        log(msg)
        if with_anisotropy:
            msg = 'K3 = '+str(K3*1e3)+' meV/link'
            log(msg)
            msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
            log(msg)
        
    elif num_config==5:
        E0, J1, J2, J3, J4 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2], soln_iso[0][3], soln_iso[0][4]
        if with_anisotropy:
            K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
            K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
            K3 = np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
            K4 = np.array([soln_x[0][3], soln_y[0][3], soln_z[0][3]])
            A = np.array([soln_x[0][4], soln_y[0][4], soln_z[0][4]])
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
        log(msg)
        msg = 'the NNNNN corordinations for all configs and sites are: '+str(CN4s)
        log(msg)
        msg = 'the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        if with_anisotropy:
            msg = 'K1 = '+str(K1*1e3)+' meV/link'
            log(msg)    
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        if with_anisotropy:
            msg = 'K2 = '+str(K2*1e3)+' meV/link'
            log(msg)
        msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
        log(msg)
        if with_anisotropy:
            msg = 'K3 = '+str(K3*1e3)+' meV/link'
            log(msg)   
        msg = 'J4 = '+str(J4*1e3)+' meV/link with d4 = '+str(ds_stable[4])+' ang and avg. NNNNN coordination = '+str(CN4)
        log(msg)
        if with_anisotropy:
            msg = 'K4 = '+str(K4*1e3)+' meV/link'
            log(msg)
            msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
            log(msg)
        

    if ds_stable[1]/ds_stable[2] >= 0.8:
        msg = '** d1/d2 is greater than 0.8, consider adding the 2nd neighbor for accurate results'
        log(msg)
        
    elif ds_stable[1]/ds_stable[3] >= 0.7:
        msg = '** d1/d3 is greater than 0.7, consider adding the 3rd neighbor for accurate results'
        log(msg)

    msg = 'the Hamiltonian fitting procedure finished successfullly'
    log(msg)

    if with_anisotropy:
        return [S_stable,J1,J2,J3,J4,K1x,K1y,K1z,K2x,K2y,K2z,K3x,K3y,K3z,K4x,K4y,K4z,Ax,Ay,Az]
    else:
        return [S_stable,J1,J2,J3,J4]



def fit_heisenberg_anisotropic(max_neigh=3,mag_from='OSZICAR',mag_species=[],d_thresh=0.05):

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    root_path = os.getcwd()

    num_struct = len(os.listdir(root_path+'/static_runs'))

    struct_base = Structure.from_file(root_path+'/static_runs/config_0/POSCAR')

    ortho_ab = (struct_base.lattice.gamma>88 and struct_base.lattice.gamma<92) and (struct_base.lattice.a/struct_base.lattice.b<0.9 or
    struct_base.lattice.a/struct_base.lattice.b>1.1)

    if ortho_ab:
        msg = '** WARNING: The structure seems to have orthogonal and unequal a and b vectors, proceed with caution!'
        log(msg)

    saxes = [(1,0,0),(0,0,1)]

    msg = 'attempting to collect data and fit the Hamiltonian now'
    log(msg)


    num_neigh = min([max_neigh, num_struct-1])
    msg = 'total '+str(num_struct)+' valid FM/AFM configs have been detected, including '
    msg += str(num_neigh)+' nearest-neighbors in the fitting'
    log(msg)

    semifinal_list = []

    for i in range(num_struct):
        
        msg = 'checking vasp run status of config_'+str(i)+' static and non-collinear runs'
        log(msg)
        config_info = []
        stat_path = root_path+'/static_runs/config_'+str(i)
        struct = Structure.from_file(root_path+'/static_runs/config_'+str(i)+'/POSCAR')
        inc = Incar.from_file(root_path+'/static_runs/config_'+str(i)+'/INCAR')
        struct.add_spin_by_site(inc.as_dict()['MAGMOM'])
        run = Vasprun(stat_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        if not run.converged_electronic:
            msg = '*** static run have not converged for config_'+str(i)+', exiting'
            log(msg)
            return [0]*10
        else:
            msg = 'found converged static run'
            log(msg)
            
        energy = float(run.final_energy)  
            
        config_info.append(i)
        config_info.append(struct)
        config_info.append(energy)

        for axis in saxes:

            mae_path = root_path+'/MAE/config_'+str(i)+'/'+str(axis).replace(' ','')
            run = Vasprun(mae_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
            struct = Structure.from_file(mae_path+'/POSCAR')
            if not run.converged_electronic:
                msg = '*** non-collinear run have not converged for config_'+str(i)+' and axis '+str(axis)+', exiting'
                log(msg)
                return [0]*10
            else:
                msg = 'found converged non-collinear run'
                log(msg)
            energy = float(run.final_energy)
            config_info.append(energy)
        
        semifinal_list.append(config_info)

    semifinal_list = sorted(semifinal_list, key = lambda x : x[2])
    most_stable = semifinal_list[0][0]

    msg = '### The most stable config = config_'+str(most_stable)
    log(msg)

    energies_ncl = semifinal_list[0][3:]
    EMA = np.argmin(energies_ncl)

    msg = '### The easy magnetization axis (EMA) = '+str(saxes[EMA])
    log(msg)

    num_mag_atoms = count_mag_atoms(semifinal_list[0][1],magnetic_list)
    E_100_001 = (energies_ncl[0] - energies_ncl[1])/(num_mag_atoms)
    msg = '### magnetocrystalline anisotropic energies (MAE) are:'
    log(msg)
    msg = 'E[100]-E[001] = '+str(E_100_001*1e6)+' ueV/magnetic_atom'
    log(msg)


    for i in range(len(semifinal_list)):

        config = semifinal_list[i][0]
        stat_path = root_path+'/static_runs'+'/config_'+str(config)
        if mag_from=='Bader' and config==most_stable:
            if not os.path.exists(stat_path+'/bader.dat'):
                msg = 'starting bader analysis for config_'+str(config)
                log(msg)
                ba = bader_analysis_from_path(stat_path)
                msg = 'finished bader analysis successfully'
                log(msg)
                f = open(stat_path+'/bader.dat','wb')
                dump(ba,f)
                f.close()                         
            else:
                f = open(stat_path+'/bader.dat','rb')
                ba = load(f)
                f.close()
                msg = 'reading magmoms from bader file'
                log(msg)
            magmom_stable = max(ba['magmom'])
            S_stable = magmom_stable/2.0

        elif mag_from=='OSZICAR' and config==0:
            config_magmom = read_mag_oszi(stat_path)
            num_mag_atoms = count_mag_atoms(semifinal_list[i][1],magnetic_list)
            magmom_stable = config_magmom/num_mag_atoms
            S_stable = magmom_stable/2.0

    E0 = Symbol('E0')
    J1 = Symbol('J1')
    J2 = Symbol('J2')
    J3 = Symbol('J3')
    J4 = Symbol('J4')
    K1z = Symbol('K1z')
    K2z = Symbol('K2z')
    K3z = Symbol('K3z')
    K4z = Symbol('K4z')
    Az = Symbol('Az')

    kB = np.double(8.6173303e-5)

    fitted = False

    while num_neigh>0:
        
        final_list = semifinal_list[:(num_neigh+1)]

        num_config = len(final_list)
        eqn_set_x = [0]*num_config
        eqn_set_z = [0]*num_config
        
        CN1s = []
        CN2s = []
        CN3s = []
        CN4s = []

        for i in range(num_config):
            
            config = final_list[i][0]
            struct = final_list[i][1]
            energy_iso = final_list[i][2]
            energies_ncl = final_list[i][3:]
            stat_path = root_path+'/static_runs'+'/config_'+str(config)
                   
            out = Outcar(stat_path+'/OUTCAR')
                
            sites_mag = []
            magmoms_mag = []
            magmoms_out = []
            for j in range(len(struct)):
                element = struct[j].specie.element
                if element in magnetic_list:
                    sign_magmom = np.sign(struct[j].specie.spin)
                    magmom = sign_magmom*magmom_stable
                    magmoms_mag.append(magmom)
                    sites_mag.append(struct[j])
                    magmoms_out.append(out.magnetization[j]['tot'])
            struct_mag = Structure.from_sites(sites_mag)
            struct_mag_out = Structure.from_sites(sites_mag)
            struct_mag.remove_spin()
            struct_mag.add_site_property('magmom',magmoms_mag)
            struct_mag_out.add_site_property('magmom',magmoms_out)
            N = len(struct_mag)
            msg = 'config_'+str(config)+' (only magnetic atoms) = '
            log(msg)
            log(struct_mag)
            msg = 'same config with magmoms from OUTCAR is printed below, make sure this does not deviate too much from above'
            log(msg)
            log(struct_mag_out)

            if np.linalg.norm(np.sign(magmoms_mag)-np.sign(magmoms_out))>1e-6:
                msg = 'seems the sign of magnetism has changed after DFT, exiting'
                log(msg)
                return [0]*10
            
            ds = dist_neighbors(struct_mag,d_thresh=d_thresh)
            dr = ds[0]

            eqn_x, eqn_z = E0 - energies_ncl[0], E0 - energies_ncl[1]

            N1s = []
            N2s = []
            N3s = []
            N4s = []
            
            for j in range(N):
                site = j
                S_site = struct_mag.site_properties['magmom'][j]/2.0
                if num_config==2:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                elif num_config==3:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                elif num_config==4:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                    N3s = Nfinder(struct_mag,site,ds[3],dr)[0]
                elif num_config==5:
                    N1s = Nfinder(struct_mag,site,ds[1],dr)[0]
                    N2s = Nfinder(struct_mag,site,ds[2],dr)[0]
                    N3s = Nfinder(struct_mag,site,ds[3],dr)[0]
                    N4s = Nfinder(struct_mag,site,ds[4],dr)[0]
                
                for N1 in N1s:
                    S_N1 = struct_mag.site_properties['magmom'][N1]/2.0
                    eqn_x += -0.5*J1*S_site*S_N1
                    eqn_z += (-0.5*J1*S_site*S_N1) + (-0.5*K1z*S_site*S_N1)
                if N2s:
                    for N2 in N2s:
                        S_N2 = struct_mag.site_properties['magmom'][N2]/2.0
                        eqn_x += -0.5*J2*S_site*S_N2
                        eqn_z += (-0.5*J2*S_site*S_N2) + (-0.5*K2z*S_site*S_N2)
                if N3s:
                    for N3 in N3s:
                        S_N3 = struct_mag.site_properties['magmom'][N3]/2.0
                        eqn_x += -0.5*J3*S_site*S_N3
                        eqn_z += (-0.5*J3*S_site*S_N3) + (-0.5*K3z*S_site*S_N3)
                if N4s:
                    for N4 in N4s:
                        S_N4 = struct_mag.site_properties['magmom'][N4]/2.0
                        eqn_x += -0.5*J4*S_site*S_N4
                        eqn_z += (-0.5*J4*S_site*S_N4) + (-0.5*K4z*S_site*S_N4)
                eqn_z += -Az*np.square(S_site)

                CN1s.append(len(N1s))
                CN2s.append(len(N2s))
                CN3s.append(len(N3s))
                CN4s.append(len(N4s))

            eqn_set_x[i] = eqn_x
            eqn_set_z[i] = eqn_z

            if config==most_stable:
                struct_mag_stable = struct_mag
                ds_stable = ds
                struct_stable = struct_mag

        msg = '### mu = '+str(magmom_stable)+' bohr magnetron/magnetic atom'
        log(msg)
                
        msg = 'eqns are:'
        log(msg)
        
        for eqn in eqn_set_x:
            msg = str(eqn)+' = 0'
            log(msg)        
        for eqn in eqn_set_z:
            msg = str(eqn)+' = 0'
            log(msg)        

        if num_config==2:
            soln = linsolve(eqn_set_x+eqn_set_z, E0, J1, K1z, Az)
        elif num_config==3:
            soln = linsolve(eqn_set_x+eqn_set_z, E0, J1, J2, K1z, K2z, Az)
        elif num_config==4:
            soln = linsolve(eqn_set_x+eqn_set_z, E0, J1, J2, J3, K1z, K2z, K3z, Az)
        elif num_config==5:
            soln = linsolve(eqn_set_x+eqn_set_z, E0, J1, J2, J3, J4, K1z, K2z, K3z, K4z, Az)

        soln = list(soln)
        msg = 'the solutions are:'
        log(msg)
        log(soln)

        try:
            quant = (soln and np.max(np.abs(soln[0]))<5e3)
            if quant:
                fitted = True
                break
        except Exception as e:
            log(e)
            fitted = False

        if not fitted:
            num_neigh -= 1
            msg = 'looks like these set of equations are either not solvable or yielding unphysical values'
            log(msg)
            msg = 'reducing the number of included NNs to '+str(num_neigh)
            log(msg)

            
    if not fitted:
        msg = '*** could not fit the Hamiltonian after several tries, exiting'
        log(msg)
        return [0]*10

    CN1 = np.mean(CN1s)
    CN2 = np.mean(CN2s)
    CN3 = np.mean(CN3s)
    CN4 = np.mean(CN4s)


    if num_config==2:
        E0, J1, K1z, Az = soln[0] 
        J2, J3, J4, K2z, K3z, K4z = 6*[0]
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        msg = 'K1z = '+str(K1z*1e3)+' meV/link'
        log(msg)
        msg = 'Az = '+str(Az*1e3)+' meV/magnetic_atom'
        log(msg)
        
    elif num_config==3:
        E0, J1, J2, K1z, K2z, Az = soln[0]
        J3, J4, K3z, K4z = 4*[0]
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        msg = 'K1z = '+str(K1z*1e3)+' meV/link'
        log(msg)
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        msg = 'K2z = '+str(K2z*1e3)+' meV/link'
        log(msg)
        msg = 'Az = '+str(Az*1e3)+' meV/magnetic_atom'
        log(msg)
        
    elif num_config==4:
        E0, J1, J2, J3, K1z, K2z, K3z, Az = soln[0]
        J4, K4z = 0, 0
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
        log(msg)
        msg = '### the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        msg = 'K1z = '+str(K1z*1e3)+' meV/link'
        log(msg)
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        msg = 'K2z = '+str(K2z*1e3)+' meV/link'
        log(msg)
        msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
        log(msg)
        msg = 'K3z = '+str(K3z*1e3)+' meV/link'
        log(msg)
        msg = 'Az = '+str(Az*1e3)+' meV/magnetic_atom'
        log(msg)
        
    elif num_config==5:
        E0, J1, J2, J3, J4, K1z, K2z, K3z, K4z, Az = soln[0]
        msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
        log(msg)
        msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
        log(msg)
        msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
        log(msg)
        msg = 'the NNNNN corordinations for all configs and sites are: '+str(CN4s)
        log(msg)
        msg = 'the solutions are:'
        log(msg)
        msg = 'E0 = '+str(E0)+' eV'
        log(msg)
        msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
        log(msg)
        msg = 'K1z = '+str(K1z*1e3)+' meV/link'
        log(msg)    
        msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
        log(msg)
        msg = 'K2z = '+str(K2z*1e3)+' meV/link'
        log(msg)
        msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
        log(msg)
        msg = 'K3z = '+str(K3z*1e3)+' meV/link'
        log(msg)   
        msg = 'J4 = '+str(J4*1e3)+' meV/link with d4 = '+str(ds_stable[4])+' ang and avg. NNNNN coordination = '+str(CN4)
        log(msg)
        msg = 'K4z = '+str(K4z*1e3)+' meV/link'
        log(msg)
        msg = 'Az = '+str(Az*1e3)+' meV/magnetic_atom'
        log(msg)
        

    if ds_stable[1]/ds_stable[2] >= 0.8:
        msg = '** d1/d2 is greater than 0.8, consider adding the 2nd neighbor for accurate results'
        log(msg)
        
    elif ds_stable[1]/ds_stable[3] >= 0.7:
        msg = '** d1/d3 is greater than 0.7, consider adding the 3rd neighbor for accurate results'
        log(msg)

    msg = 'the Hamiltonian fitting procedure finished successfullly'
    log(msg)

    
    return [S_stable,J1,J2,J3,J4,K1z,K2z,K3z,K4z,Az]


