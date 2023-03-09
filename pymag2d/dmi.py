from pymatgen.io.vasp.sets import MPStaticSet, MPSOCSet
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Kpoints, Potcar
from pymatgen.io.vasp.outputs import Vasprun, Outcar, Oszicar
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, FrozenJobErrorHandler,\
    MeshSymmetryErrorHandler, PositiveEnergyErrorHandler, StdErrHandler, NonConvergingErrorHandler, PotimErrorHandler 
from custodian.vasp.jobs import VaspJob
from custodian.vasp.validators import VasprunXMLValidator
import sys
import os
from shutil import copyfile
import fileinput
import datetime
from time import time, sleep
from pickle import dump
import numpy as np
from misc import *



magnetic_list = [Element('Co'), Element('Cr'), Element('Fe'), Element('Mn'), Element('Mo'), Element('Ni'), Element('V'), Element('W')]

LDAUJ_dict = {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0}
LDAUU_dict = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}
LDAUL_dict = {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2}

stat_dict_gl = {'ALGO': 'Normal', 'ISMEAR': 0, 'EDIFF': 1E-6, 'KPAR': 2, 'NCORE': 1, 'NSIM': 4, 'LORBMOM': True, 'LAECHG': True,
'LREAL': False, 'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'NELMIN': 6, 'NELM': 250, 'LVHAR': False,
'SIGMA': 0.01, 'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4, 'LCHARG': True, 'LWAVE': True, 'LVTOT': False, 'ISYM': -1}

stat_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
FrozenJobErrorHandler(timeout=900), MeshSymmetryErrorHandler(), PositiveEnergyErrorHandler(), StdErrHandler()]

validator = [VasprunXMLValidator()]

dipole_dict = {'DIPOL': [0.5, 0.5, 0.5], 'IDIPOL': 3, 'LDIPOL': True}



def run_dmi(vasp_cmd_ncl,prev_path='./relaxations/config_0',randomise_cmd=False,supercell=[4,4,1],max_neigh=3,
    user_incar_settings={},LDAUJs={},LDAUUs={},LDAULs={},dipole_correction=False,
    kpt_den=300,xc='PBE',pot='PBE_54',mag_species=[],d_thresh=0.05,user_potcar_settings={},max_errors=10,lamb=50.0,rm_wavecar=True):

    cmd = vasp_cmd_ncl.split()[:]

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

    stat_dict.update(user_incar_settings)

    soc_dict = stat_dict.copy()
    soc_changes = {'LCHARG': False, 'LWAVE': True, 'LAECHG': False, 'ICHARG': 1,
    'I_CONSTRAINED_M': 1, 'LAMBDA': lamb}
    soc_dict.update(soc_changes)
    del soc_dict['LDAUJ']
    del soc_dict['LDAUL']
    del soc_dict['LDAUU']
    potcar = Potcar.from_file(prev_path+'/POTCAR')
    symbols = potcar.symbols
    rwigs = []
    for symbol in symbols:
        rwigs.append(CovalentRadius.radius[symbol.split('_')[0]])
    soc_dict['RWIGS'] = rwigs
    soc_dict.update(user_incar_settings)

    struct_prev = Structure.from_file(prev_path+'/CONTCAR')
    out = Outcar(prev_path+'/OUTCAR')
    magmoms_prev = []
    for j in range(len(struct_prev)):
        magmoms_prev.append(out.magnetization[j]['tot'])
    struct_prev.add_site_property('magmom',magmoms_prev)
    struct = struct_prev.copy()
    struct.make_supercell(supercell)
        
    sites_mag = []
    magmoms_out = []
    for j in range(len(struct)):
        element = struct[j].specie
        if element in magnetic_list:
            sites_mag.append(struct[j])
            magmoms_out.append(struct.site_properties['magmom'][j])
            mag_species = element
    struct_mag = Structure.from_sites(sites_mag)
    struct_mag.add_site_property('magmom',magmoms_out)


    ds = dist_neighbors(struct_mag,d_thresh)
    dr = ds[0]

    for n in range(max_neigh):

        site1_coord = struct_mag.cart_coords[0]
        d = ds[n+1]
        neighbors, neigh_coords = Nfinder(struct_mag,0,d,dr)

        for neighbor in neighbors:

            site2_coord = struct_mag.cart_coords[neighbor] 
            
            for i in range(3):
                directions = [0,1,2]
                directions.pop(i)

                for k in range(4):
                    magmoms3D = []

                    for j in range(len(struct)):
                        site_coord = struct.cart_coords[j]
                        magmom = struct.site_properties['magmom'][j]
                        magmom3D = [0,0,0]

                        if np.linalg.norm(site1_coord-site_coord)<1e-2:
                            if k==0 or k==1:
                                magmom3D[directions[0]] = magmom
                            else:
                                magmom3D[directions[0]] = -magmom

                        elif np.linalg.norm(site2_coord-site_coord)<1e-2:
                            if k==0 or k==2:
                                magmom3D[directions[1]] = magmom
                            else:
                                magmom3D[directions[1]] = -magmom

                        else:
                            magmom3D[2] = magmom
                        
                        magmoms3D.append(magmom3D)

                    directions3D = np.sign(np.array(magmoms3D))
                    directions3D = directions3D.astype(int).flatten().tolist()
                    dmi_path = root_path+'/DMI/shell-'+str(n)+'_neigh-'+str(neighbor)+'_dir-'+str(i)+'_calc-'+str(k)

                    if os.path.exists(dmi_path+'/running'):
                        msg = 'the calculation at '+dmi_path+' is being already handled by another stream, moving on'
                        log(msg)
                        continue

                    clean = sanitize(dmi_path)
                    if not clean:
                        struct_soc = struct.copy(site_properties={'magmom': magmoms3D})
                        soc = MPSOCSet(struct_soc,saxis=(0,0,1),copy_chgcar=False,reciprocal_density=kpt_den,
                            force_gamma=True,user_potcar_functional=pot,sort_structure=False,magmom=magmoms3D,
                            user_incar_settings=soc_dict,user_potcar_settings=user_potcar_settings)
                        soc.write_input(dmi_path)
                        with open(dmi_path+'/running','w+') as f:
                            f.write('JOB RUNNING!')
                        msg = 'raised the running flag at '+dmi_path
                        log(msg)
                        incar = soc.incar
                        incar.update(soc_dict)
                        del incar['SAXIS']
                        incar['M_CONSTR'] = directions3D
                        incar.write_file(dmi_path+'/INCAR')
                        kpts = Kpoints.from_file(dmi_path+'/KPOINTS')
                        kpts.kpts[0][2] = 1
                        kpts.write_file(dmi_path+'/KPOINTS')

                        if randomise_cmd:
                            cmd_rand = cmd[:]
                            cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                            job = [VaspJob(cmd_rand)]
                        else:
                            job = [VaspJob(cmd)]
                        cust = Custodian(stat_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
                            gzipped_output=False,checkpoint=False)
                        msg = 'running non-collinear run for directory '+dmi_path 
                        log(msg)
                        done = 0

                        os.chdir(dmi_path)
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
                        if os.path.exists(dmi_path+'/running'):
                            os.remove(dmi_path+'/running')
                    
                        if done == 1:
                            msg = 'non-collinear run finished successfully for directory '+dmi_path
                            log(msg)
                        else:
                            msg = 'non-collinear run failed for directory '+dmi_path
                            msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                            log(msg)
                            return None

                        if os.path.exists(dmi_path+'/WAVECAR') and rm_wavecar:
                            os.remove(dmi_path+'/WAVECAR')    

    msg = 'all done from this stream'
    log(msg)

    return 0



def calc_dmi(prev_path='./relaxations/config_0',supercell=[4,4,1],S=None,max_neigh=3,mag_species=[],d_thresh=0.05,
    exclude_penalty_energy=False,rm_wavecar=True):

    root_path = os.getcwd()

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    struct_prev = Structure.from_file(prev_path+'/CONTCAR')
    out = Outcar(prev_path+'/OUTCAR')
    magmoms_prev = []
    for j in range(len(struct_prev)):
        magmoms_prev.append(out.magnetization[j]['tot'])
    struct_prev.add_site_property('magmom',magmoms_prev)
    struct = struct_prev.copy()
    struct.make_supercell(supercell)
        
    sites_mag = []
    magmoms_out = []
    for j in range(len(struct)):
        element = struct[j].specie
        if element in magnetic_list:
            sites_mag.append(struct[j])
            magmoms_out.append(struct.site_properties['magmom'][j])
            mag_species = element
    struct_mag = Structure.from_sites(sites_mag)
    struct_mag.add_site_property('magmom',magmoms_out)
    
    if S==None:
        S = read_mag_oszi(prev_path)/(2*len(struct_mag)/(supercell[0]*supercell[1]*supercell[2]))

    ds = dist_neighbors(struct_mag,d_thresh)
    dr = ds[0]

    Ds = [0]*max_neigh
    Ds_form = [0]*max_neigh
    us = [0]*max_neigh
    zxus = [0]*max_neigh

    for n in range(max_neigh):

        site1_coord = struct_mag.cart_coords[0]
        d = ds[n+1]
        neighbors, neigh_coords = Nfinder(struct_mag,0,d,dr)
        Dshell = []
        Dshell_form = []
        ushell = []
        zxushell = []
        neigh_idx = 0

        for neighbor in neighbors:
            site2_coord = struct_mag.cart_coords[neighbor]
            site2_coord_unwarpped = neigh_coords[neigh_idx]
            D = [0,0,0]
            D_form = [0,0,0]

            for i in range(3):
                directions = [0,1,2]
                directions.pop(i)
                energies = [0,0,0,0]

                for k in range(4): 
                    magmoms3D = []

                    for j in range(len(struct)):
                        site_coord = struct.cart_coords[j]
                        magmom = struct.site_properties['magmom'][j]
                        magmom3D = [0,0,0]

                        if np.linalg.norm(site1_coord-site_coord)<1e-2:
                            if k==0 or k==1:
                                magmom3D[directions[0]] = magmom
                            else:
                                magmom3D[directions[0]] = -magmom

                        elif np.linalg.norm(site2_coord-site_coord)<1e-2:
                            if k==0 or k==2:
                                magmom3D[directions[1]] = magmom
                            else:
                                magmom3D[directions[1]] = -magmom

                        else:
                            magmom3D[2] = magmom
                        
                        magmoms3D.append(magmom3D)

                    directions3D = np.sign(np.array(magmoms3D))
                    directions3D = directions3D.astype(int).flatten().tolist()
                    dmi_path = root_path+'/DMI/shell-'+str(n)+'_neigh-'+str(neighbor)+'_dir-'+str(i)+'_calc-'+str(k)
                    msg = 'current directory is '+dmi_path
                    log(msg)
                    out_dmi = Outcar(dmi_path+'/OUTCAR')
                    for j in range(len(magmoms3D)):
                        diff = np.linalg.norm(np.array(magmoms3D[j])-np.array(list(out_dmi.magnetization[0]['tot'])))
                        if diff>7.0:
                            msg = 'the norm of difference between input and output 3D magmoms is '+str(diff)+', exiting, check OUTCAR!'
                            log(msg)
                            return None
                    run = Vasprun(dmi_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
                    with open(dmi_path+'/OSZICAR','r') as f:
                        for line in f:
                            if 'E_p =' in line:
                                E_p = float(line.split()[2])

                    if exclude_penalty_energy:
                        energies[k] = run.final_energy-E_p
                        msg = 'the energy from the penalty functional is '+str(E_p)+' eV and this is being subtracted from total energy'
                        log(msg)
                    else:
                        energies[k] = run.final_energy
                        msg = 'the energy from the penalty functional is '+str(E_p)+' eV and this is included in total energy'
                        log(msg)
                
                D[i] = (energies[0]+energies[3]-energies[1]-energies[2])/(4*S**2)

            u = (site2_coord_unwarpped - site1_coord)/np.linalg.norm(site2_coord_unwarpped - site1_coord)
            z = np.array([0,0,1])
            zxu = np.cross(z,u)

            if np.abs(zxu[1])>1e-10:
                dplane = D[1]/zxu[1]
            else:
                dplane = D[0]/zxu[0]
            D_form = [dplane]

            Dshell.append(D)
            Dshell_form.append(D_form)
            ushell.append(u)
            zxushell.append(zxu)

            neigh_idx += 1

        Ds[n] = Dshell
        Ds_form[n] = Dshell_form
        us[n] = ushell
        zxus[n] = zxushell  
    
    msg = 'the calculated shell-neighborwise D vectors are '+str(Ds)
    log(msg)
    msg = 'the calculated shell-neighborwise in-plane d factors are '+str(Ds_form)
    log(msg)
    msg = 'the calculated in-plane u vectors are '+str(us)
    log(msg)
    msg = 'the calculated z cross u vectors are '+str(zxus)
    log(msg)

    return Ds, Ds_form



def run_dmi_hex(vasp_cmd_ncl,prev_path='./relaxations/config_0',randomise_cmd=False,supercell=[4,4,1],max_neigh=1,
    user_incar_settings={},LDAUJs={},LDAUUs={},LDAULs={},dipole_correction=False,
    kpt_den=300,xc='PBE',pot='PBE_54',mag_species=[],d_thresh=0.05,user_potcar_settings={},max_errors=10,lamb=50.0):

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

    stat_dict.update(user_incar_settings)

    soc_dict = stat_dict.copy()
    soc_changes = {'LCHARG': False, 'LWAVE': True, 'LAECHG': False, 'ICHARG': 1,
    'I_CONSTRAINED_M': 1, 'LAMBDA': lamb}
    soc_dict.update(soc_changes)
    potcar = Potcar.from_file(prev_path+'/POTCAR')
    symbols = potcar.symbols
    rwigs = []
    LDAUJ = []
    LDAUU = []
    LDAUL = []
    for symbol in symbols:
        element = symbol.split('_')[0]
        rwigs.append(CovalentRadius.radius[element])
        if element in LDAUJ_dict.keys():
            LDAUJ.append(LDAUJ_dict[element])
        else:
            LDAUJ.append(0)
        if element in LDAUU_dict.keys():
            LDAUU.append(LDAUU_dict[element])
        else:
            LDAUU.append(0)
        if element in LDAUL_dict.keys():
            LDAUL.append(LDAUL_dict[element])
        else:
            LDAUL.append(-1)
    soc_dict['RWIGS'] = rwigs
    soc_dict.update(user_incar_settings)

    struct_prev = Structure.from_file(prev_path+'/CONTCAR')
    out = Outcar(prev_path+'/OUTCAR')
    magmoms_prev = []
    for j in range(len(struct_prev)):
        magmoms_prev.append(out.magnetization[j]['tot'])
    struct_prev.add_site_property('magmom',magmoms_prev)
    struct = struct_prev.copy()
    struct.make_supercell(supercell)
        
    sites_mag = []
    magmoms_out = []
    for j in range(len(struct)):
        element = struct[j].specie
        if element in magnetic_list:
            sites_mag.append(struct[j])
            magmoms_out.append(struct.site_properties['magmom'][j])
            mag_species = element
    struct_mag = Structure.from_sites(sites_mag)
    struct_mag.add_site_property('magmom',magmoms_out)

    if len(struct_mag)>16:
        msg = 'point to the directory consisting the collinear FM calculation of an unit cell using prev_path, exiting'
        log(msg)
        return None
    if np.sqrt((struct_mag[0].frac_coords[0])**2+(struct_mag[0].frac_coords[1])**2)>1e-4:
        msg = 'make sure the magnetic atom is at (0,0) in the unit cell, exiting'
        log(msg)
        return None

    ds = dist_neighbors(struct_mag,d_thresh)
    dr = ds[0]

    for n in range(max_neigh):

        site1_coord = struct_mag.cart_coords[0]
        d = ds[n+1]
        neighbors, neigh_coords = Nfinder(struct_mag,0,d,dr)
        neighbor = None
        for l in range(len(neighbors)):
            site2_coord_unwrapped = neigh_coords[l]
            vec = (site2_coord_unwrapped - site1_coord)/np.linalg.norm(site2_coord_unwrapped - site1_coord)
            if n==0 or n==2 and np.linalg.norm(vec-np.array([-0.5,-0.866,0]))<1e-2:
                neighbor = neighbors[l]
                break
            # TODO: figure out vector for neighbor shell 1

        if neighbor==None:
            msg = 'could not find the desired neighbor, make sure the magnetic atom is at (0,0) in the unit cell, exiting'
            log(msg)
            return None

        site2_coord = struct_mag.cart_coords[neighbor]
        
        for i in [1,2]:
            directions = [0,1,2]
            directions.pop(i)

            for k in range(4):
                magmoms3D = []

                for j in range(len(struct)):
                    site_coord = struct.cart_coords[j]
                    magmom = struct.site_properties['magmom'][j]
                    magmom3D = [0,0,0]

                    if np.linalg.norm(site1_coord-site_coord)<1e-2:
                        if k==0 or k==1:
                            magmom3D[directions[0]] = np.abs(magmom)
                        else:
                            magmom3D[directions[0]] = -np.abs(magmom)

                    elif np.linalg.norm(site2_coord-site_coord)<1e-2:
                        if k==0 or k==2:
                            magmom3D[directions[1]] = np.abs(magmom)
                        else:
                            magmom3D[directions[1]] = -np.abs(magmom)

                    else:
                        magmom3D[2] = magmom
                    
                    magmoms3D.append(magmom3D)

                directions3D = np.sign(np.array(magmoms3D))
                directions3D = directions3D.astype(int).flatten().tolist()
                dmi_path = root_path+'/DMI_hex/shell-'+str(n)+'_neigh-'+str(neighbor)+'_dir-'+str(i)+'_calc-'+str(k)

                if os.path.exists(dmi_path+'/running'):
                    msg = 'the calculation at '+dmi_path+' is being already handled by another stream, moving on'
                    log(msg)
                    continue

                clean = sanitize(dmi_path)
                if not clean:
                    struct_soc = struct.copy(site_properties={'magmom': magmoms3D})
                    soc = MPSOCSet(struct_soc,saxis=(0,0,1),copy_chgcar=False,reciprocal_density=kpt_den,
                        force_gamma=True,user_potcar_functional=pot,sort_structure=False,magmom=magmoms3D,
                        user_incar_settings=soc_dict,user_potcar_settings=user_potcar_settings)
                    soc.write_input(dmi_path)
                    with open(dmi_path+'/running','w+') as f:
                        f.write('JOB RUNNING!')
                    msg = 'raised the running flag at '+dmi_path
                    log(msg)
                    incar = soc.incar
                    incar.update(soc_dict)
                    incar['LDAUJ'] = LDAUJ
                    incar['LDAUU'] = LDAUU
                    incar['LDAUL'] = LDAUL
                    del incar['SAXIS']
                    incar['M_CONSTR'] = directions3D
                    incar.write_file(dmi_path+'/INCAR')
                    kpts = Kpoints.from_file(dmi_path+'/KPOINTS')
                    kpts.kpts[0][2] = 1
                    kpts.write_file(dmi_path+'/KPOINTS')

                    if randomise_cmd:
                        cmd_rand = cmd[:]
                        cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                        job = [VaspJob(cmd_rand)]
                    else:
                        job = [VaspJob(cmd)]
                    cust = Custodian(stat_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
                        gzipped_output=False,checkpoint=False)
                    msg = 'running non-collinear run for directory '+dmi_path 
                    log(msg)
                    done = 0

                    os.chdir(dmi_path)
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
                    if os.path.exists(dmi_path+'/running'):
                        os.remove(dmi_path+'/running')
                
                    if done == 1:
                        msg = 'non-collinear run finished successfully for directory '+dmi_path
                        log(msg)
                    else:
                        msg = 'non-collinear run failed for directory '+dmi_path
                        msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                        log(msg)
                        return None

                    if os.path.exists(dmi_path+'/WAVECAR') and rm_wavecar:
                        os.remove(dmi_path+'/WAVECAR')    

    msg = 'all done from this stream'
    log(msg)

    return 0



def calc_dmi_hex(prev_path='./relaxations/config_0',S=None,supercell=[4,4,1],max_neigh=1,
    mag_species=[],d_thresh=0.05,exclude_penalty_energy=False,rm_wavecar=True):

    root_path = os.getcwd()

    if mag_species:
        magnetic_list = []
        for s in mag_species:
            magnetic_list.append(Element(s))
    else:
        magnetic_list = magnetic_list_gl[:]

    struct_prev = Structure.from_file(prev_path+'/CONTCAR')
    out = Outcar(prev_path+'/OUTCAR')
    magmoms_prev = []
    for j in range(len(struct_prev)):
        magmoms_prev.append(out.magnetization[j]['tot'])
    struct_prev.add_site_property('magmom',magmoms_prev)
    struct = struct_prev.copy()
    struct.make_supercell(supercell)
        
    sites_mag = []
    magmoms_out = []
    for j in range(len(struct)):
        element = struct[j].specie
        if element in magnetic_list:
            sites_mag.append(struct[j])
            magmoms_out.append(struct.site_properties['magmom'][j])
            mag_species = element
    struct_mag = Structure.from_sites(sites_mag)
    struct_mag.add_site_property('magmom',magmoms_out)

    if S==None:
        S = read_mag_oszi(prev_path)/(2*len(struct_mag)/(supercell[0]*supercell[1]*supercell[2]))

    ds = dist_neighbors(struct_mag,d_thresh)
    dr = ds[0]

    Ds = []
    dplanes = []

    for n in range(max_neigh):

        site1_coord = struct_mag.cart_coords[0]
        d = ds[n+1]
        neighbors, neigh_coords = Nfinder(struct_mag,0,d,dr)
        neighbor = None
        for l in range(len(neighbors)):
            site2_coord_unwrapped = neigh_coords[l]
            vec = (site2_coord_unwrapped - site1_coord)/np.linalg.norm(site2_coord_unwrapped - site1_coord)
            if n==0 or n==2 and np.linalg.norm(vec-np.array([-0.5,-0.866,0]))<1e-2:
                neighbor = neighbors[l]
                break
            # TODO: figure out vector for neighbor shell 1

        if neighbor==None:
            msg = 'could not find the desired neighbor, make sure the magnetic atom is at (0,0) in the unit cell, exiting'
            log(msg)
            return None

        site2_coord = struct_mag.cart_coords[neighbor]
        D_dict = {}
        D = [0,0,0]

        for i in [1,2]:
            directions = [0,1,2]
            directions.pop(i)
            energies = [0,0,0,0]

            for k in range(4): 
                magmoms3D = []

                for j in range(len(struct)):
                    site_coord = struct.cart_coords[j]
                    magmom = struct.site_properties['magmom'][j]
                    magmom3D = [0,0,0]

                    if np.linalg.norm(site1_coord-site_coord)<1e-2:
                        if k==0 or k==1:
                            magmom3D[directions[0]] = np.abs(magmom)
                        else:
                            magmom3D[directions[0]] = -np.abs(magmom)

                    elif np.linalg.norm(site2_coord-site_coord)<1e-2:
                        if k==0 or k==2:
                            magmom3D[directions[1]] = np.abs(magmom)
                        else:
                            magmom3D[directions[1]] = -np.abs(magmom)

                    else:
                        magmom3D[2] = magmom
                    
                    magmoms3D.append(magmom3D)

                directions3D = np.sign(np.array(magmoms3D))
                directions3D = directions3D.astype(int).flatten().tolist()
                dmi_path = root_path+'/DMI_hex/shell-'+str(n)+'_neigh-'+str(neighbor)+'_dir-'+str(i)+'_calc-'+str(k)
                msg = 'current directory is '+dmi_path
                log(msg)
                out_dmi = Outcar(dmi_path+'/OUTCAR')
                for j in range(len(magmoms3D)):
                    diff = np.linalg.norm(np.array(magmoms3D[j])-np.array(list(out_dmi.magnetization[j]['tot'])))
                    if diff>0.3:
                        msg = 'the norm of difference between input and output 3D magmoms is '+str(diff)+', exiting, check OUTCAR!'
                        log(msg)
                        msg = 'input = '+str(magmoms3D[j])+', output = '+str(list(out_dmi.magnetization[j]['tot']))
                        log(msg)
                        return None
                run = Vasprun(dmi_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
                with open(dmi_path+'/OSZICAR','r') as f:
                    for line in f:
                        if 'E_p =' in line:
                            E_p = float(line.split()[2])

                if exclude_penalty_energy:
                    energies[k] = float(run.final_energy)-E_p
                    msg = 'the energy from the penalty functional is '+str(E_p)+' eV and this is being subtracted from total energy'
                    log(msg)
                else:
                    energies[k] = float(run.final_energy)
                    msg = 'the energy from the penalty functional is '+str(E_p)+' eV and this is included in total energy'
                    log(msg)
            
            D[i] = (energies[0]+energies[3]-energies[1]-energies[2])/(4*S**2)

        u = (site2_coord_unwrapped - site1_coord)/np.linalg.norm(site2_coord_unwrapped - site1_coord)
        z = np.array([0,0,1])
        zxu = np.cross(z,u)
        msg = 'the positional unit vector is '+str(u)
        log(msg)
        msg = 'z cross unit vector is '+str(zxu)
        log(msg)

        dplane1 = D[1]/zxu[1]
        dz = D[2]-dplane1*zxu[2]
        msg = 'the dplane is '+str(1e3*np.abs(dplane1))+' meV'

        dplane0 = -dplane1
        if n==0 or n==2 and np.linalg.norm(vec-np.array([-0.5,-0.866,0]))<1e-2:
            D[0] = dplane0*zxu[0]
            D_dict[(-0.5,-0.866,0)] = np.array(D)
            D_dict[(-0.5,0.866,0)] = np.array([-D[0],D[1],D[2]])
            D_dict[(-1,0,0)] = np.array([0,np.sqrt(D[0]**2 + D[1]**2),-D[2]])
            D_dict[(0.5,-0.866,0)] = np.array([D[0],-D[1],-D[2]])
            D_dict[(0.5,0.866,0)] = np.array([-D[0],-D[1],-D[2]])
            D_dict[(1,0,0)] = np.array([0,-np.sqrt(D[0]**2 + D[1]**2),D[2]])
        #TODO: for 2nd nearest neighbor

        Ds.append(D_dict)
        dplanes.append(np.abs(dplane1))
        with open('N'+str(n+1)+'vecs','wb') as f:
            dump(np.array(list(D_dict.keys())),f)
        with open('D'+str(n+1)+'vecs','wb') as f:
            dump(np.array(list(D_dict.values())),f)            

    msg = 'the calculated shell-neighborwise D vectors are '+str(Ds)
    log(msg)
    msg = 'the calculated shellwise in-plane d costants are '+str(dplanes)
    log(msg)

    return Ds, dplanes



def run_calc_dmi_hex_NN_inplane(vasp_cmd_ncl,prev_path='./relaxations/config_0',randomise_cmd=False,S=None,
    user_incar_settings={},LDAUJs={},LDAUUs={},LDAULs={},dipole_correction=False,icharg_scan=1,
    kpt_den=300,nbands_factor=1.0,constrain=False,constrain_only_magsites=True,exclude_penalty_energy=False,lamb=50.0,
    xc='PBE',pot='PBE_54',mag_species=[],d_thresh=0.05,user_potcar_settings={},max_errors=10,rm_wavecar=True):

    cmd = vasp_cmd_ncl.split()[:]

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

    stat_dict.update(user_incar_settings)

    soc_dict = stat_dict.copy()
    soc_changes = {'LCHARG': False, 'LWAVE': True, 'LAECHG': False}
    if constrain:
        soc_changes.update({'I_CONSTRAINED_M': 1, 'LAMBDA': lamb})
    if 'SCAN' in xc:
        soc_changes.update({'ICHARG': icharg_scan})
    soc_dict.update(soc_changes)
    potcar = Potcar.from_file(prev_path+'/POTCAR')
    symbols = potcar.symbols
    rwigs = []
    LDAUJ = []
    LDAUU = []
    LDAUL = []
    for symbol in symbols:
        element = symbol.split('_')[0]
        rwigs.append(CovalentRadius.radius[element])
        if element in LDAUJ_dict.keys():
            LDAUJ.append(LDAUJ_dict[element])
        else:
            LDAUJ.append(0)
        if element in LDAUU_dict.keys():
            LDAUU.append(LDAUU_dict[element])
        else:
            LDAUU.append(0)
        if element in LDAUL_dict.keys():
            LDAUL.append(LDAUL_dict[element])
        else:
            LDAUL.append(-1)
    if constrain:
        soc_dict['RWIGS'] = rwigs
    soc_dict.update(user_incar_settings)

    struct_prev = Structure.from_file(prev_path+'/CONTCAR')
    out = Outcar(prev_path+'/OUTCAR')
    magmoms_prev = []
    for j in range(len(struct_prev)):
        magmoms_prev.append(out.magnetization[j]['tot'])
    struct_prev.add_site_property('magmom',magmoms_prev)
    struct = struct_prev.copy()
    struct.make_supercell([4,1,1])

    if S==None:
        S = read_mag_oszi(prev_path)/2
 
    sites_mag = []
    magmoms_out = []   

    for j in range(len(struct)):
        element = struct[j].specie
        if element in magnetic_list:
            sites_mag.append(struct[j])
            magmoms_out.append(struct.site_properties['magmom'][j])
            mag_species = element
    struct_mag = Structure.from_sites(sites_mag)
    struct_mag.add_site_property('magmom',magmoms_out)

    if len(struct_mag)>4:
        msg = 'point to the directory consisting the collinear calculation of an unit cell using prev_path, exiting'
        log(msg)
        return None

    energies = [0,0]
    
    for i in range(2):

        magmoms3D = []
        mag_site_count=-1

        for j in range(len(struct)):

            magmom = struct.site_properties['magmom'][j]
            magmom3D = [0,0,0]
            element = struct[j].specie
            if element in magnetic_list:
                mag_site_count += 1

            if i==0:
                config = 'anticlockwise-spin-spiral'
                if mag_site_count==0 :
                    magmom3D[2] = np.abs(magmom)
                elif mag_site_count==1:
                    magmom3D[0] = -np.abs(magmom)
                elif mag_site_count==2:
                    magmom3D[2] = -np.abs(magmom)
                elif mag_site_count==3:
                    magmom3D[0] = np.abs(magmom)
                    mag_site_count = -1e6
                else:
                    #magmom3D = np.around(0.8*np.random.random_sample(3)-0.4,2).tolist()
                    magmom3D = [0,0,magmom]

            elif i==1:
                config = 'clockwise-spin-spiral'
                if mag_site_count==0:
                    magmom3D[2] = np.abs(magmom)
                elif mag_site_count==1:
                    magmom3D[0] = np.abs(magmom)
                elif mag_site_count==2:
                    magmom3D[2] = -np.abs(magmom)
                elif mag_site_count==3:
                    magmom3D[0] = -np.abs(magmom)
                    mag_site_count = -1e6
                else:
                    #magmom3D = np.around(0.8*np.random.random_sample(3)-0.4,2).tolist()
                    magmom3D = [0,0,magmom]
            
            magmoms3D.append(magmom3D)

        directions3D = np.sign(magmoms3D)
        directions3D = directions3D.astype(int).flatten().tolist()
        magmoms3D_ideal = np.array(magmoms3D)
        thresh_inds = np.abs(magmoms3D_ideal)<0.5
        magmoms3D_ideal[thresh_inds] = 0
        directions3D_ideal = np.sign(magmoms3D_ideal)
        directions3D_ideal = directions3D_ideal.astype(int).flatten().tolist()

        # magmoms3D_str = ''
        # directions3D_str = ''
        # for magmom3D in magmoms3D:
        #     if magmom3D==[0, 0, 0]:
        #         magmoms3D_str += '3*0  '
        #         directions3D_str += '3*0  '
        #     else:
        #         for magmom in magmom3D:
        #             magmoms3D_str += str(magmom)+' '
        #             directions3D_str += str(np.sign(magmom).astype(int))+' '
        #         magmoms3D_str += ' '
        #         directions3D_str += ' '

        dmi_path = root_path+'/DMI_hex_NN_inplane/'+config

        if os.path.exists(dmi_path+'/running'):
            msg = 'the calculation at '+dmi_path+' is being already handled by another stream, moving on'
            log(msg)
            continue

        clean = sanitize(dmi_path)
        if not clean:
            struct_soc = struct.copy(site_properties={'magmom': magmoms3D})
            soc = MPSOCSet(struct_soc,saxis=(0,0,1),copy_chgcar=False,
                nbands_factor=nbands_factor,reciprocal_density=kpt_den,force_gamma=True,
                user_potcar_functional=pot,sort_structure=False,user_incar_settings=soc_dict,
                user_potcar_settings=user_potcar_settings)
            soc.write_input(dmi_path)
            with open(dmi_path+'/running','w+') as f:
                f.write('JOB RUNNING!')
            msg = 'raised the running flag at '+dmi_path
            log(msg)
            incar = soc.incar
            incar.update(soc_dict)
            if 'SCAN' in xc:
                if 'ISIF' in incar.keys():
                    incar.pop('ISIF')
                if 'LDAU' in incar.keys():
                    incar.pop('LDAU')
                if 'LDAUJ' in incar.keys():
                    incar.pop('LDAUJ')
                if 'LDAUL' in incar.keys():
                    incar.pop('LDAUL')
                if 'LDAUPRINT' in incar.keys():
                    incar.pop('LDAUPRINT')
                if 'LDAUTYPE' in incar.keys():
                    incar.pop('LDAUTYPE')
                if 'LDAUU' in incar.keys():
                    incar.pop('LDAUU')
            else:
                incar['LDAUJ'] = LDAUJ
                incar['LDAUU'] = LDAUU
                incar['LDAUL'] = LDAUL

            del incar['SAXIS']
            if constrain:
                if constrain_only_magsites:
                    incar['M_CONSTR'] = directions3D_ideal
                else:
                    incar['M_CONSTR'] = directions3D

            incar.write_file(dmi_path+'/INCAR')
            # for line in fileinput.input([dmi_path+'/INCAR'], inplace=True):
            #     if line.strip().startswith('MAGMOM = '):
            #         line = 'MAGMOM = '+magmoms3D_str+'\n'
            #     sys.stdout.write(line)
            kpts = Kpoints.from_file(dmi_path+'/KPOINTS')
            kpts.kpts[0][2] = 1
            kpts.write_file(dmi_path+'/KPOINTS')

            if randomise_cmd:
                cmd_rand = cmd[:]
                cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                job = [VaspJob(cmd_rand)]
            else:
                job = [VaspJob(cmd)]
            cust = Custodian(stat_handlers,job,validator,max_errors=max_errors,polling_time_step=5,monitor_freq=10,
                gzipped_output=False,checkpoint=False)
            msg = 'running non-collinear run for directory '+dmi_path 
            log(msg)
            done = 0

            os.chdir(dmi_path)
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
            if os.path.exists(dmi_path+'/running'):
                os.remove(dmi_path+'/running')
        
            if done == 1:
                msg = 'non-collinear run finished successfully for directory '+dmi_path
                log(msg)
            else:
                msg = 'non-collinear run failed for directory '+dmi_path
                msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                log(msg)
                return None

            if os.path.exists(dmi_path+'/WAVECAR') and rm_wavecar:
                os.remove(dmi_path+'/WAVECAR')

        out = Outcar(dmi_path+'/OUTCAR')
        mags_resolved = out.magnetization
        magmoms3D_out = []
        for entry_resolved in mags_resolved:
            mag3D_out = list(entry_resolved['tot'])
            mag3D_out_ideal = []
            for m in mag3D_out:
                if np.abs(m)>0.5:
                    mag3D_out_ideal.append(m)
                else:
                    mag3D_out_ideal.append(0)
            magmoms3D_out.append(mag3D_out_ideal)
        magmoms3D_out = np.array(magmoms3D_out)

        if np.linalg.norm(np.sign(magmoms3D_ideal)-np.sign(magmoms3D_out))>1e-6:
            msg = 'seems the sign of magnetism has changed after DMI calculation, exiting'
            log(msg)
            try:
                os.rename(dmi_path+'/OSZICAR',dmi_path+'/OSZICAR.flipped')
                os.rename(dmi_path+'/OUTCAR',dmi_path+'/OUTCAR.flipped')
                if os.path.exists(dmi_path+'/vasprun.xml'):
                    os.rename(dmi_path+'/vasprun.xml',dmi_path+'/vasprun.xml.flipped')
                if os.path.exists(dmi_path+'/WAVECAR'):
                    os.rename(dmi_path+'/WAVECAR',dmi_path+'/WAVECAR.flipped')
            except:
                pass
            return None

        run = Vasprun(dmi_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        if constrain:
            with open(dmi_path+'/OSZICAR','r') as f:
                E_p = 0
                for line in f:
                    if 'E_p =' in line:
                        E_p = float(line.split()[2])
            if exclude_penalty_energy:
                energies[i] = float(run.final_energy)-E_p
                msg = 'the energy from the penalty functional is '+str(E_p)+' eV and this is being subtracted from total energy'
                log(msg)
            else:
                energies[i] = float(run.final_energy)
                msg = 'the energy from the penalty functional is '+str(E_p)+' eV'
                log(msg)
        else:
            energies[i] = float(run.final_energy)


    dplane = (energies[1]-energies[0])/(12*S**2)
    msg = 'the in plane d constant is '+str(1e3*dplane)+' meV'
    log(msg)
    msg = 'all done from this stream'
    log(msg)
    return dplane
