from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
import os
from shutil import copyfile
import datetime
import numpy as np
import numba
from numba.typed import List



def replace_text(fileName,toFind,replaceWith):
    s = open(fileName).read()
    s = s.replace(toFind, replaceWith)
    f = open(fileName, 'w')
    f.write(s)
    f.close()


def log(string,log_file='log'):
    string = str(string)
    f = open(log_file,'a+')
    time = datetime.datetime.now()
    f.write('>>> '+str(time)+'    '+string+'\n')
    f.close()
    print('>>> '+string)


def sanitize(path):

    try:
        run = Vasprun(path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        if run.converged:
            msg = 'found converged vasp run in '+path+', no sanitization required'
            log(msg)
            return True
        else:
            raise ValueError
    except Exception as e:
        msg = str(e)
        log(msg)
        msg = 'found unconverged, nonexistent or damaged vasp run in '+path+', starting sanitization'
        log(msg)
        try:
            try_struct = Structure.from_file(path+'/CONTCAR')
            copyfile(path+'/CONTCAR',path+'/CONTCAR.bk')
            msg = 'backed up CONTCAR'
            log(msg)
        except Exception as e:
            msg = str(e)
            log(msg)
            msg = 'no valid CONTCAR found in '+ path
            log(msg)
        try:
            os.remove(path+'/INCAR')
            os.remove(path+'/INCAR.orig')
            os.remove(path+'/KPOINTS')
            os.remove(path+'/KPOINTS.orig')
            os.remove(path+'/POTCAR')
            os.remove(path+'/POTCAR.orig')
            msg = 'removed old INCAR, KPOINTS and POTCAR'
            log(msg)
        except Exception as e:
            msg = str(e)
            log(msg)
            msg = 'no INCAR or KPOINTS or POTCAR found in '+ path
            log(msg)
        return False


def read_mag_oszi(path,dim=1):
    mag = 0
    with open(path+'/OSZICAR') as f:
        for line in f:
            row = line.split()
            if 'mag=' in line and dim==3:
                mag = np.array([float(row[-3]),float(row[-2]),float(row[-1])])
            elif 'mag=' in line and dim==1:
                mag = float(row[-1])

    return mag


def count_mag_atoms(struct,magnetic_list):
    num_mag_atoms = 0
    for site in struct:
        element = site.specie.element
        if element in magnetic_list:
            num_mag_atoms += 1
    return num_mag_atoms


def dist_neighbors(struct,d_thresh=0.05,d_buff=0.01):
    struct_l = struct.copy()
    struct_l.make_supercell([20,20,1])
    distances = np.unique(np.sort(np.around(struct_l.distance_matrix[1],2)))[0:15]
    dr_max = 0.01
    for i in range(len(distances)):
        for j in range(len(distances)):
            dr = np.abs(distances[i]-distances[j])
            if distances[j]<distances[i] and dr<d_thresh:
                distances[i]=distances[j]
                if dr>dr_max:
                    dr_max = dr
    distances = np.unique(distances)
    msg = 'neighbor distances are: '+str(distances)+' ang'
    log(msg)
    msg = 'treating '+str(dr_max)+' ang separated atoms as same neighbors'
    log(msg)
    distances[0]=dr_max+d_buff
    return distances


def Nfinder(struct_mag,site,d_N,dr):
    N = len(struct_mag)
    coord_site = struct_mag.cart_coords[site]
    Ns = struct_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    coords_N = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        coords_N[i] = Ns[i].coords
        for j in range(N):
            if struct_mag[j].distance(Ns_wrapped[i])<0.01:
                candidates[i] = j
                break
    return candidates, coords_N


@numba.jit((numba.int64,numba.float64,numba.float64[:],numba.int64,numba.int64,
    numba.int64[:,:],numba.int64[:,:],numba.int64[:,:],numba.int64[:,:]),nopython=True)
def MC_func(N,T,input_set,trange,threshold,N1list,N2list,N3list,N4list):

    kB = np.double(8.6173303e-5)
    [S, EMA, J1, J2, J3, J4, K1x, K1y, K1z, K2x, K2y, K2z, K3x, K3y, K3z, K4x, K4y, K4z, Ax, Ay, Az,
    CN1, CN2, CN3, CN4, temp1, temp2, temp3] = input_set
    [J1, J2, J3, J4, K1x, K1y, K1z, K2x, K2y, K2z, K3x, K3y, K3z, K4x, K4y, K4z, Ax, Ay, 
    Az] = 1e-3*np.array([J1, J2, J3, J4, K1x, K1y, K1z, K2x, K2y, K2z, K3x, K3y, K3z, K4x, K4y, K4z, Ax, Ay, Az])

    if CN1!=len(N1list[0]) or CN2!=len(N2list[0]) or CN3!=len(N3list[0]) or CN4!=len(N4list[0]):
        raise ValueError 

    spins_init = S*np.ones(N)
    spins_abs = np.abs(np.copy(spins_init))
    if EMA==0:
        spins_x = np.copy(spins_init)
        spins_y = np.zeros(N)
        spins_z = np.zeros(N)
    elif EMA==1:
        spins_x = np.zeros(N)
        spins_y = np.copy(spins_init)
        spins_z = np.zeros(N)
    elif EMA==2:
        spins_x = np.zeros(N)
        spins_y = np.zeros(N)
        spins_z = np.copy(spins_init)   

    mag_ups, mag_up_sqs, mag_downs, mag_down_sqs, mag_tots, mag_tot_sqs = np.zeros(trange-threshold), np.zeros(trange-threshold), np.zeros(
        trange-threshold), np.zeros(trange-threshold), np.zeros(trange-threshold), np.zeros(trange-threshold)

    for t in range(trange):
        
        mag_up = 0
        mag_down = 0
        mag_tot = 0
        prec_err = False
        fallen = False
        
        for i in range(N):

            site = np.random.randint(0,N)
            N1s = N1list[site]
            N2s = N2list[site]
            N3s = N3list[site]
            N4s = N4list[site]                   
            S_current = np.array([spins_x[site],spins_y[site],spins_z[site]])
            if np.linalg.norm(S_current)>spins_abs[site]:
                S_current = spins_abs[site]*S_current/(np.linalg.norm(S_current)+1e-10)
                [spins_x[site],spins_y[site],spins_z[site]] = S_current
            
            u, v = np.random.random(),np.random.random()
            phi = 2*np.pi*u
            theta = np.arccos(2*v-1)
            S_x = spins_abs[site]*np.sin(theta)*np.cos(phi)
            S_y = spins_abs[site]*np.sin(theta)*np.sin(phi)
            S_z = spins_abs[site]*np.cos(theta)
            S_after = np.array([S_x,S_y,S_z])

            E_current = 0
            E_after = 0
            
            for N1 in N1s:
                if N1!=100000 or N1!=-5:
                    S_N1 = np.array([spins_x[N1],spins_y[N1],spins_z[N1]])
                    if K1x!=0 and K1y!=0:
                        E_current += -J1*np.dot(S_current,S_N1) + (-K1x*S_current[0]*S_N1[0]) + (-K1y*S_current[1]*S_N1[1]) + (
                            -K1z*S_current[2]*S_N1[2])
                        E_after += -J1*np.dot(S_after,S_N1) + (-K1x*S_after[0]*S_N1[0]) + (-K1y*S_after[1]*S_N1[1]) + (
                            -K1z*S_after[2]*S_N1[2])
                    else:
                        E_current += -J1*np.dot(S_current,S_N1) + (-K1z*S_current[2]*S_N1[2])
                        E_after += -J1*np.dot(S_after,S_N1) + (-K1z*S_after[2]*S_N1[2])
            if J2!=0:
                for N2 in N2s:
                    if N2!=100000 or N2!=-5:
                        S_N2 = np.array([spins_x[N2],spins_y[N2],spins_z[N2]])
                        if K2x!=0 and K2y!=0:
                            E_current += -J2*np.dot(S_current,S_N2) + (-K2x*S_current[0]*S_N2[0]) + (-K2y*S_current[1]*S_N2[1]) + (
                            -K2z*S_current[2]*S_N2[2])
                            E_after += -J2*np.dot(S_after,S_N2) + (-K2x*S_after[0]*S_N2[0]) + (-K2y*S_after[1]*S_N2[1]) + (
                            -K2z*S_after[2]*S_N2[2])
                        else:
                            E_current += -J2*np.dot(S_current,S_N2) + (-K2z*S_current[2]*S_N2[2])
                            E_after += -J2*np.dot(S_after,S_N2) + (-K2z*S_after[2]*S_N2[2])                            
            if J3!=0: 
                for N3 in N3s:
                    if N3!=100000 or N3!=-5:
                        S_N3 = np.array([spins_x[N3],spins_y[N3],spins_z[N3]])
                        if K3x!=0 and K3y!=0:
                            E_current += -J3*np.dot(S_current,S_N3) + (-K3x*S_current[0]*S_N3[0]) + (-K3y*S_current[1]*S_N3[1]) + (
                            -K3z*S_current[2]*S_N3[2])
                            E_after += -J3*np.dot(S_after,S_N3) + (-K3x*S_after[0]*S_N3[0]) + (-K3y*S_after[1]*S_N3[1]) + (
                            -K3z*S_after[2]*S_N3[2])
                        else:
                            E_current += -J3*np.dot(S_current,S_N3) + (-K3z*S_current[2]*S_N3[2])
                            E_after += -J3*np.dot(S_after,S_N3) + (-K3z*S_after[2]*S_N3[2])
            if J4!=0: 
                for N4 in N4s:
                    if N4!= 100000 or N4!=-5:
                        S_N4 = np.array([spins_x[N4],spins_y[N4],spins_z[N4]])
                        if K4x!=0 and K4y!=0:
                            E_current += -J4*np.dot(S_current,S_N4) + (-K4x*S_current[0]*S_N4[0]) + (-K4y*S_current[1]*S_N4[1]) + (
                            -K4z*S_current[2]*S_N4[2])
                            E_after += -J4*np.dot(S_after,S_N4) + (-K4x*S_after[0]*S_N4[0]) + (-K4y*S_after[1]*S_N4[1]) + (
                            -K4z*S_after[2]*S_N4[2])
                        else:
                            E_current += -J4*np.dot(S_current,S_N4) + (-K4z*S_current[2]*S_N4[2])
                            E_after += -J4*np.dot(S_after,S_N4) + (-K4z*S_after[2]*S_N4[2])                                    

            if Ax!=0 and Ay!=0:
                E_current += -Ax*np.square(S_current[0]) + (-Ay*np.square(S_current[1])) + (-Az*np.square(S_current[2])) 
                E_after += -Ax*np.square(S_after[0]) + (-Ay*np.square(S_after[1])) + (-Az*np.square(S_after[2]))
            else:
                E_current += -Az*np.square(S_current[2]) 
                E_after += -Az*np.square(S_after[2])                
            
            del_E = E_after-E_current
                    
            if del_E<0:
                spins_x[site],spins_y[site],spins_z[site] = S_x,S_y,S_z 
            else:
                samp = np.random.random()
                if samp<=np.exp(-del_E/(kB*T)):
                    spins_x[site],spins_y[site],spins_z[site] = S_x,S_y,S_z


        if t>=threshold:

            mag_vec_up = np.zeros(3)
            mag_vec_down = np.zeros(3)

            for i in range(N):
                if spins_init[i]>0:
                    mag_vec_up += 2*np.array([spins_x[i], spins_y[i], spins_z[i]])
                else:
                    mag_vec_down += 2*np.array([spins_x[i], spins_y[i], spins_z[i]])

            mag_up = np.linalg.norm(mag_vec_up)
            mag_ups[t-threshold] = np.abs(mag_up)
            mag_up_sqs[t-threshold] = np.square(mag_up)
            mag_down = np.linalg.norm(mag_vec_down)
            mag_downs[t-threshold] = np.abs(mag_down)
            mag_down_sqs[t-threshold] = np.square(mag_down)

            mag_vec_tot = 2*np.array([np.sum(spins_x),np.sum(spins_y),np.sum(spins_z)])
            mag_tot = np.linalg.norm(mag_vec_tot)
            mag_tots[t-threshold] = np.abs(mag_tot)
            mag_tot_sqs[t-threshold] = np.square(mag_tot)
            

    (M_up,M_up_sq,M_down,M_down_sq,M_tot,M_tot_sq) = (np.mean(mag_ups),np.mean(mag_up_sqs),np.mean(mag_downs),np.mean(mag_down_sqs),
    np.mean(mag_tots),np.mean(mag_tot_sqs))

    X_up = (M_up_sq-np.square(M_up))/(N*kB*T)
    X_down = (M_down_sq-np.square(M_down))/(N*kB*T)
    X_tot = (M_tot_sq-np.square(M_tot))/(N*kB*T)
    M_up = M_up/(2*S*N)
    M_down = M_down/(2*S*N)
    M_tot = M_tot/(2*S*N)

    return M_up,X_up,M_down,X_down,M_tot,X_tot,spins_x,spins_y,spins_z
