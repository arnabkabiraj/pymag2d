#!/nlsasfs/home/skyrmions/akabiraj/pyenv3810/bin/python

import sys
import os
import re
from shutil import copyfile
from pymatgen.io.vasp.outputs import Vasprun, Outcar
sys.path.insert(0,'/nlsasfs/home/skyrmions/akabiraj/tools/scripts/pymag2d')
from heisenberg_anisotropy import *
from dmi import *


def calc_form_energy(struct,tot_energy,ref_dir):
  form_energy = tot_energy
  ref_contents = os.listdir(ref_dir)
  compos = struct.composition.as_dict()
  count = 0
  for species in compos.keys():
      for item in ref_contents:
        if re.split('(\d+)',item.split('_')[0])[0]==species and '_dir' in item:
          #print(re.split('(\d+)',item.split('_')[0])[0])
          #print(item)
          run = Vasprun(ref_dir+'/'+item+'/initial_stat/vasprun.xml',parse_dos=False,parse_eigen=False)
          form_energy -= compos[species]*float(run.final_energy)/len(run.structures[0])
          count += 1
          break
  if count != len(compos.keys()):
    log('**energies of all elements unavailable for '+str(struct))
    raise ValueError

  return form_energy/len(struct)



relx_settings = {'IOPT':3,'EDIFF':1e-5,'EDIFFG':-0.02,'NSW':400,'KPAR':4}
relx_settings_light = {'EDIFF':1e-5,'NELM':400,'KPAR':3}
stat_settings = {'KPAR':3,'NSIM':4}
stat_settings_light = stat_settings.copy()
stat_settings_light.update({'ISYM':2,'KPAR':4})
dmi_settings = {'KPAR':2,'NSIM':4,'NELM':500}
magnetic_list = ['Mn','Cr','V']
default_magmoms = {'Mn':5,'Cr':5,'V':5}

everything = os.listdir()
root_path = os.getcwd()



for file in everything:
  
  if '.cif' in file:

    path = root_path+'/'+file[:-4]+'_dir'

    msg = 'current path is '+path
    log(msg)

    if not os.path.exists(path):
      os.mkdir(path)
      copyfile(root_path+'/'+file,path+'/'+file)

    if os.path.exists(path+'/running'):
        msg = 'the calculation at '+path+' is being already handled by another stream, moving on'
        log(msg)
        continue
    elif not os.path.exists(root_path+'/'+file):
        os.chdir(root_path)
        if os.path.exists(path+'/running'):
          os.remove(path+'/running')
        continue
    else:
        with open(path+'/running','w+') as f:
            f.write('JOB RUNNING! node, jobID = '+sys.argv[1]+', '+sys.argv[2])
        msg = 'raised the running flag at '+path
        msg += '\nnode, jobID = '+sys.argv[1]+', '+sys.argv[2]
        log(msg)

    os.chdir(path)

    x = relx_gen('mpirun -np '+str(sys.argv[3])+' vasp_std_631_acc',file,user_incar_settings_relx=relx_settings,xc='R2SCAN',
      dipole_correction=True,randomise_cmd=False,potcar_provided={'W':'W_sv'},mag_species=magnetic_list,max_errors=10)

    if x==None:
      os.chdir(root_path)
      msg = 'calculation failed, renaming '+file+' to '+file[:-4]+'.failed'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.failed')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    x = stat_gen('mpirun -np '+str(sys.argv[4])+' vasp_std_631_acc','./initial_relx/CONTCAR',user_incar_settings_stat=stat_settings_light,
      xc='R2SCAN',randomise_cmd=False,potcar_provided={'W':'W_sv'},mag_species=magnetic_list,max_errors=5)

    if x==None:
      os.chdir(root_path)
      msg = 'calculation failed, renaming '+file+' to '+file[:-4]+'.failed'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.failed')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    run = Vasprun('./initial_stat/vasprun.xml',parse_dos=False,parse_eigen=False)
    tot_energy = float(run.final_energy)
    struct = run.structures[0]
    form_energy = calc_form_energy(struct,tot_energy,'/nlsasfs/home/skyrmions/akabiraj/work/MXene_skyrmion/formation_energies')
    msg = 'the formation energy of the material '+file[:-4]+' is '+str(form_energy)+' eV/atom'
    log(msg)

    if form_energy>=0:
      os.chdir(root_path)
      msg = 'positive formaion energy, renaming '+file+' to '+file[:-4]+'.posenergy'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.posenergy')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue
    if np.abs(read_mag_oszi('./initial_stat'))<0.5:
      os.chdir(root_path)
      msg = 'very low magnetism, renaming '+file+' to '+file[:-4]+'.nomag'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.nomag')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue
    out = Outcar('./initial_stat/OUTCAR')
    mag_site_number = 0
    for j in range(len(struct)):
      if np.abs(out.magnetization[j]['tot'])>0.5:
        mag_site_number += 1
    if mag_site_number>1:
      os.chdir(root_path)
      msg = 'multiple sites with high magnetism, renaming '+file+' to '+file[:-4]+'.multimag'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.multimag')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    x = run_heisenberg('mpirun -np '+str(sys.argv[5])+' vasp_std_631_acc','./initial_stat/POSCAR',[2,4,1],max_neigh=1,relx=False,
      user_incar_settings_relx=relx_settings_light,user_incar_settings_stat=stat_settings,xc='R2SCAN',randomise_cmd=False,
      potcar_provided={'W':'W_sv'},exit_if_afm=True,mag_species=magnetic_list,default_magmoms=default_magmoms,
      max_errors_relx=10,max_errors_stat=5)

    if x=='AFM':
      os.chdir(root_path)
      msg = 'AFM ground state, renaming '+file+' to '+file[:-4]+'.afm'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.afm')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue
    elif x==None:
      os.chdir(root_path)
      msg = 'calculation failed, renaming '+file+' to '+file[:-4]+'.failed'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.failed')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    [S,J1,J2,J3,J4] = fit_heisenberg(with_anisotropy=False,max_neigh=1,mag_species=magnetic_list)

    if J1<0.1e-3 and S!=0:
      os.chdir(root_path)
      msg = 'material likely AFM, renaming '+file+' to '+file[:-4]+'.afm'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.afm')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue
    elif J1==0 and S==0:
      os.chdir(root_path)
      msg = 'calculation failed, renaming '+file+' to '+file[:-4]+'.failed'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.failed')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    d1plane = run_calc_dmi_hex_NN_inplane('mpirun -np '+str(sys.argv[6])+' vasp_ncl_631_acc',user_incar_settings=dmi_settings,xc='R2SCAN',
      constrain=True,lamb=50,icharg_scan=2,randomise_cmd=False,user_potcar_settings={'W':'W_sv'},mag_species=magnetic_list,max_errors=3)

    if d1plane==None:
      msg = 'seems the DMI calculations failed, retrying with lambda=10 instead of the regular 50'
      log(msg)
      if os.path.exists(path+'/DMI_hex_NN_inplane') and not os.path.exists(path+'/DMI_hex_NN_inplane_flipped'):
        os.rename(path+'/DMI_hex_NN_inplane',path+'/DMI_hex_NN_inplane_flipped')
      dmi_settings.update({'NELM':750})
      d1plane = run_calc_dmi_hex_NN_inplane('mpirun -np 36 vasp_ncl_631_acc',user_incar_settings=dmi_settings,xc='R2SCAN',constrain=True,
        lamb=10,icharg_scan=2,randomise_cmd=False,user_potcar_settings={'W':'W_sv'},mag_species=magnetic_list,max_errors=2)

    if d1plane==None:
      os.chdir(root_path)
      msg = 'calculation failed despite multiple attempts, renaming '+file+' to '+file[:-4]+'.failed'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.failed')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    ratio = np.abs(d1plane)/np.abs(J1)

    msg = 'the d1plane/J1 ratio is '+str(ratio)
    log(msg)

    if ratio<0.09:
      os.chdir(root_path)
      msg = 'low d1plane/J1 ratio, renaming '+file+' to '+file[:-4]+'.lowratio'
      log(msg)
      if os.path.exists(root_path+'/'+file):
        os.rename(file,file[:-4]+'.lowratio')
      if os.path.exists(path+'/running'):
        os.remove(path+'/running')
      msg = 'moving on to the next material'
      log(msg)
      continue

    os.chdir(root_path)
    msg = 'all initial calculations finished for material '+file[:-4]+' which is a potentially good material for skyrmion formation'
    log(msg)
    msg = 'renaming '+file+' to '+file[:-4]+'.skyrmion'
    log(msg)
    if os.path.exists(root_path+'/'+file):
      os.rename(file,file[:-4]+'.skyrmion')
    if os.path.exists(path+'/running'):
      os.remove(path+'/running')
    msg = 'on to the next material!'
    log(msg)

