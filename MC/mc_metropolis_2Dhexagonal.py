#!/home/arnab/atomate/pyenv395/bin/python

from pymatgen.core import Structure
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
import sys
import os
from shutil import copyfile
from subprocess import Popen
import datetime
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
import math
from numba import jit, prange, cuda
from numba.core import types
from numba.typed import Dict
from numba_progress import ProgressBar
from pickle import load, dump



__author__ = "Arnab Kabiraj"
__copyright__ = "Copyright 2023, NSDRL, IISc Bengaluru"
__credits__ = ["Arnab Kabiraj"]


root_path = os.getcwd()


def log(string):
    string = str(string)
    f = open(root_path+'/log','a+')
    time = datetime.datetime.now()
    f.write('>>> '+str(time)+'    '+string+'\n')
    f.close()
    print('>>> '+string)


def dist_neighbors(struct,d_buff=0.01):
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
    # coords_N = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        for j in range(N):
            if struct_mag[j].distance(Ns_wrapped[i])<0.1:
                candidates[i] = j
                break
        # coords_N[i] = Ns[i].coords
    return candidates


@cuda.jit
def Nfinder_kernel(all_coords,coord_N,index):
    """
    Code for kernel.
    """
    pos = cuda.grid(1)
    if pos <= all_coords.size:
        if math.sqrt((all_coords[pos][0]-coord_N[0])**2 + (all_coords[pos][1]-coord_N[1])**2 +
            (all_coords[pos][2]-coord_N[2])**2) < 0.1:
            index[0] = pos


def Nfinder_GPU(struc_mag,site,d_N,dr):
    N = len(struc_mag)
    coord_site = struc_mag.cart_coords[site]
    Ns = struc_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    #print(Ns)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        coord_N = np.array([Ns_wrapped[i].x,Ns_wrapped[i].y,Ns_wrapped[i].z],dtype='float32')
        index = np.array([-5])
        threadsperblock = 480
        blockspergrid = math.ceil(all_coords.shape[0] / threadsperblock)
        Nfinder_kernel[blockspergrid,threadsperblock](all_coords,coord_N,index)
        candidates[i]=index[0]
        if index[0]==-5:
            for j in range(N):
                if struc_mag[j].distance(Ns_wrapped[i])<0.01:
                    candidates[i] = j
                    break
    return candidates


@jit(nopython=True,parallel=True)
def spins_rand_init(N):
    spins_init = np.zeros((N,3))
    for i in prange(N):
        u, v = np.random.random(),np.random.random()
        phi = 2*np.pi*u
        theta = np.arccos(2*v-1)
        S_x = np.sin(theta)*np.cos(phi)
        S_y = np.sin(theta)*np.sin(phi)
        S_z = np.cos(theta)
        S_3D = np.array([S_x,S_y,S_z])
        spins_init[i] = S_3D
    return spins_init


@jit(nopython=True,parallel=True)
def spins_updown_init(N,neighs):
    spins_init = np.zeros((N,3))
    for i in prange(N):
        if i in neighs:
            spins_init[i][EMA] = -1.0
        else:
            spins_init[i][EMA] = 1.0
    return spins_init


@jit(nopython=True,parallel=True)
def spins_circle_init(N,radius):
    spins_init = np.zeros((N,3))
    point1 = all_coords[int(N/3)]
    for i in prange(N):
        point2 = all_coords[i]
        dist12 = np.linalg.norm(point2-point1)
        if dist12 <= radius:
            spins_init[i][EMA] = -1.0
        else:
            spins_init[i][EMA] = 1.0
    return spins_init


def find_max_len(lst): 
    maxList = max(lst, key = lambda i: len(i)) 
    maxLength = len(maxList)   
    return maxLength 


@jit(nopython=True,nogil=True)
def MC_func(spins_init,T,num_iterations,progress_proxy):

    spins = spins_init
    kB = np.double(8.6173303e-5)

    for itr in range(num_iterations):
        
        for i in range(N):

            site = np.random.randint(0,N)
            N1s = N1list[site]
            N2s = N2list[site]
            N3s = N3list[site]
            N4s = N4list[site] 
            
            S_current = S*spins[site]
            u, v = np.random.random(),np.random.random()
            phi = 2*np.pi*u
            theta = np.arccos(2*v-1)
            S_x = np.sin(theta)*np.cos(phi)
            S_y = np.sin(theta)*np.sin(phi)
            S_z = np.cos(theta)
            S_after = S*np.array([S_x,S_y,S_z])
            E_current = 0
            E_after = 0
            N1idx, N2idx, N3idx, N4idx = 0,0,0,0
            
            for N1 in N1s:
                if N1!=1e9 or N1!=-5:
                    S_N1 = spins[N1]
                    D1 = D1vecs[N1idx]
                    E_current += -J1*np.dot(S_current,S_N1) + (-K1z*S_current[2]*S_N1[2]) + (-np.dot(D1,np.cross(S_current,S_N1)))
                    E_after += -J1*np.dot(S_after,S_N1) + (-K1z*S_after[2]*S_N1[2]) + (-np.dot(D1,np.cross(S_after,S_N1)))
                    N1idx += 1
            if J2!=0:
                for N2 in N2s:
                    if N2!=1e9 or N2!=-5:
                        S_N2 = spins[N2]
                        D2 = D2vecs[N2idx]
                        E_current += -J2*np.dot(S_current,S_N2) + (-K2z*S_current[2]*S_N2[2])
                        E_after += -J2*np.dot(S_after,S_N2) + (-K2z*S_after[2]*S_N2[2]) 
                        if D2plane!=0:
                            D2 = D2vecs[N2idx]
                            E_current += (-np.dot(D2,np.cross(S_current,S_N2)))
                            E_after += (-np.dot(D2,np.cross(S_after,S_N2)))
                        N2idx += 1
            if J3!=0: 
                for N3 in N3s:
                    if N3!=1e9 or N3!=-5:
                        S_N3 = spins[N3]
                        D3 = D3vecs[N3idx]
                        E_current += -J3*np.dot(S_current,S_N3) + (-K3z*S_current[2]*S_N3[2])
                        E_after += -J3*np.dot(S_after,S_N3) + (-K3z*S_after[2]*S_N3[2])
                        if D3plane!=0:
                            D3 = D3vecs[N3idx]
                            E_current += (-np.dot(D3,np.cross(S_current,S_N3)))
                            E_after += (-np.dot(D3,np.cross(S_after,S_N3)))
                        N3idx += 1
            if J4!=0: 
                for N4 in N4s:
                    if N4!= 1e9 or N4!=-5:
                        S_N4 = spins[N4]
                        D4 = D4vecs[N4idx]
                        E_current += -J4*np.dot(S_current,S_N4) + (-K4z*S_current[2]*S_N4[2])
                        E_after += -J4*np.dot(S_after,S_N4) +(-K4z*S_after[2]*S_N4[2])
                        if D4plane!=0:
                            D4 = D4vecs[N4idx]
                            E_current += (-np.dot(D4,np.cross(S_current,S_N4)))
                            E_after += (-np.dot(D4,np.cross(S_after,S_N4)))
                        N4idx += 1           

            E_current +=  (-Az*np.square(S_current[2]))
            E_after += (-Az*np.square(S_after[2]))
            if np.linalg.norm(B)>1e-10:
                E_current += (-2*S*5.788e-5*np.dot(B,S_current))/S
                E_after += (-2*S*5.788e-5*np.dot(B,S_after))/S
            
            del_E = E_after-E_current
                    
            if del_E < 0:
                spins[site] = np.array([S_x,S_y,S_z])
            else:
                samp = np.random.random()
                if samp <= np.exp(-del_E/(kB*T)):
                    spins[site] = np.array([S_x,S_y,S_z])

            
        # print('t = '+str(t)+'/'+str(trange))
        progress_proxy.update(1)        

    return spins




# main code

msg = '*'*150
log(msg)
msg = '*** this code have been developed by Arnab Kabiraj at Nano-Scale Device Research Laboratory (NSDRL), IISc, Bengaluru, India ***\n'
msg += '*** for any queries please contact the authors at kabiraj@iisc.ac.in or santanu@iisc.ac.in ***'
log(msg)
msg = '*'*150
log(msg)

if not os.path.exists(root_path+'/input_MC'):
    msg = 'no input_MC file detected, exiting, write this and restart the code'
    log(msg)
else:
    msg = 'input_MC detected, will try to run the MC based on this'
    log(msg)
    sleep(3)


with open('input_MC') as f: 
    for line in f:
        row = line.split()
        if 'GPU_accel' in line:
            GPU_accel = row[-1]=='True'
        elif 'GPU_threads_per_block' in line:
            threadsperblock = int(row[-1])
        elif 'same_neighbor_thresh' in line:
            d_thresh = float(row[-1])
        elif 'structure_file' in line:
            struct_file = row[-1]
        elif 'directory' in line:
            path = root_path+'/'+row[-1]
        elif 'read_neighbors' in line:
            restart = row[-1]=='True'
        elif 'spins_filename' in line:
            spins_file = row[-1]
        elif 'circle_radius' in line:
            radius = np.double(row[-1])
        elif 'repeat' in line:
            rep_z = int(row[-1])
            rep_y = int(row[-2])
            rep_x = int(row[-3])
        elif 'J1' in line:
            J1 = np.double(row[-1])
        elif 'J2' in line:
            J2 = np.double(row[-1])
        elif 'J3' in line:
            J3 = np.double(row[-1])
        elif 'J4' in line:
            J4 = np.double(row[-1])
        elif 'K1z' in line:
            K1z = np.double(row[-1])
        elif 'K2z' in line:
            K2z = np.double(row[-1])
        elif 'K3z' in line:
            K3z = np.double(row[-1])
        elif 'K4z' in line:
            K4z = np.double(row[-1])
        elif 'Az' in line:
            Az = np.double(row[-1])
        elif 'D1plane' in line:
            D1plane = np.double(row[-1])
        elif 'D1z' in line:
            D1z = np.double(row[-1])
        elif 'D2plane' in line:
            D2plane = np.double(row[-1])
        elif 'D2z' in line:
            D2z = np.double(row[-1])
        elif 'D3plane' in line:
            D3plane = np.double(row[-1])
        elif 'D3z' in line:
            D3z = np.double(row[-1])
        elif 'D4plane' in line:
            D4plane = np.double(row[-1])
        elif 'D4z' in line:
            D4z = np.double(row[-1])
        elif 'B (T) = ' in line:
            B = np.array([np.double(row[-3]),np.double(row[-2]),np.double(row[-1])])
        elif 'EMA' in line:
            EMA = int(row[-1])
        elif 'T_start' in line:
            Tstart = float(row[-1])
        elif 'T_end' in line:
            Trange = float(row[-1])
        elif 'div_T' in line:
            div_T = int(row[-1])
        elif 'magmom' in line:
            magmom = np.double(row[-1])
            magmom = np.array([magmom])
        elif 'MCS' in line:
            num_iterations = int(float(row[-1]))
        elif 'dump_interval' in line:
            dump_intv = int(float(row[-1]))

msg = 'the read inputs are:'
msg += '\nGPU_accel = '+str(GPU_accel)
msg += '\nGPU_threads_per_block = '+str(threadsperblock)
msg += '\nsame_neighbor_thresh (angstrom) = '+str(d_thresh)
msg += '\nstructure_file = '+str(struct_file)
msg += '\ndirectory = '+str(path)
msg += '\nread_neighbors = '+str(restart)
msg += '\nspins_filename = '+str(spins_file)
msg += '\ncircle_radius (angstrom) = '+str(radius)
msg += '\nrepeat = '+str(rep_x)+' '+str(rep_y)+' '+str(rep_z)
msg += '\nJ1 (eV/link) = '+str(J1)
msg += '\nK1z (eV/link) = '+str(K1z)
msg += '\nD1plane (eV/link) = '+str(D1plane)
msg += '\nD1z (eV/link) = '+str(D1z)
msg += '\nJ2 (eV/link) = '+str(J2)
msg += '\nK2z (eV/link) = '+str(K2z)
msg += '\nD2plane (eV/link) = '+str(D2plane)
msg += '\nD2z (eV/link) = '+str(D2z)
msg += '\nJ3 (eV/link) = '+str(J3)
msg += '\nK3z (eV/link) = '+str(K3z)
msg += '\nD3plane (eV/link) = '+str(D3plane)
msg += '\nD3z (eV/link) = '+str(D3z)
msg += '\nJ4 (eV/link) = '+str(J4)
msg += '\nK4z (eV/link) = '+str(K4z)
msg += '\nD3plane (eV/link) = '+str(D4plane)
msg += '\nD3z (eV/link) = '+str(D4z)
msg += '\nAz (eV) = '+str(Az)
msg += '\nB (T) = '+str(B)
msg += '\nEMA = '+str(EMA)
msg += '\nmagmoms (mu_B/mag_atom) = '+str(magmom)
msg += '\nT_start (K) = '+str(Tstart)
msg += '\nT_end (K) = '+str(Trange)
msg += '\ndiv_T = '+str(div_T)
msg += '\nMCS = '+str(num_iterations)
msg += '\ndump_interval = '+str(dump_intv)
log(msg)



if GPU_accel:
    nf = Nfinder_GPU
    msg = 'neighbor mapping will try to use GPU acceleration'
    log(msg)
else:
    nf = Nfinder
    msg = 'neighbor mapping will be sequentially done in CPU, can be quite slow'
    log(msg)

if os.path.exists(path):
    new_name = path+'_'+str(time())
    os.rename(path,new_name)
    msg = 'found an old MC directory, renaming it to '+new_name
    log(msg)

os.makedirs(path)
try:
    copyfile(new_name+'/N1list',path+'/N1list')
    copyfile(new_name+'/N2list',path+'/N2list')
    copyfile(new_name+'/N3list',path+'/N3list')
    copyfile(new_name+'/N4list',path+'/N4list')
    copyfile(new_name+'/'+spins_file,path+'/'+spins_file)
except:
    pass


struct = Structure.from_file(filename=struct_file)
analyzer = CollinearMagneticStructureAnalyzer(struct,overwrite_magmom_mode='replace_all_if_undefined',make_primitive=False)
struct_mag_stable = analyzer.get_structure_with_only_magnetic_atoms(make_primitive=False)
S = np.abs(magmom[0]/2.0)
magmom = magmom/(2*S)
struct_mag_stable.add_site_property('magmom',magmom)
ds_stable = dist_neighbors(struct_mag_stable)
dr_max = ds_stable[0]
d_N1 = ds_stable[1]
d_N2 = ds_stable[2]
d_N3 = ds_stable[3]
d_N4 = ds_stable[4]
        
repeat = [rep_x,rep_y,rep_z]
suff = '_'+str(rep_x)+'x'+str(rep_y)+'x'+str(rep_z)
os.chdir(path)

struct_mag_stable.make_supercell(repeat)
N = len(struct_mag_stable)
all_coords = struct_mag_stable.cart_coords
with open('all_coords'+suff,'wb') as f:
    dump(all_coords,f)
copyfile('all_coords'+suff,'../all_coords'+suff)

if os.path.exists('../'+spins_file) and spins_file!='False':
    copyfile('../'+spins_file,spins_file)
    with open(spins_file, 'rb') as f:
        spins_init = load(f)        
    msg = 'successfully read initial spins'
    log(msg)
elif spins_file=='circle':
    msg = 'generating circled spin texture'
    log(msg)
    circle_neighs = nf(struct_mag_stable,int(N/3),0.1,radius)
    circle_neighs = np.array(circle_neighs)
    spins_init = spins_updown_init(N,circle_neighs)
    # spins_init = spins_circle_init(N,180)
    with open('spins_circled_'+str(radius)+suff, 'wb') as f:
        dump(spins_init, f)
    copyfile('spins_circled_'+str(radius)+suff,'../spins_circled_'+str(radius)+suff)
    msg = 'circled up-down intial spin generation is finished and dumped'
    log(msg)    
else:
    msg = 'no intial spins found, using random initial spins'
    log(msg)
    spins_init = spins_rand_init(N)
    with open('spins_rand'+suff, 'wb') as f:
        dump(spins_init, f)
    copyfile('spins_rand'+suff,'../spins_rand'+suff)
    msg = 'random intial spin generation is finished and dumped'
    log(msg)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins_init[:,0],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
plt.savefig('spins_init_X.png')
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins_init[:,1],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
plt.savefig('spins_init_Y.png')
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins_init[:,2],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
plt.savefig('spins_init_Z.png')
plt.close()


if not restart:

    N1list = [[1,2]]*N
    N2list, N3list, N4list = N1list[:], N1list[:], N1list[:]
        
    start_time_map = time()
    for i in range(N):
        N1list[i] = nf(struct_mag_stable,i,d_N1,dr_max)
        if J2!=0:
            N2list[i] = nf(struct_mag_stable,i,d_N2,dr_max)
        if J3!=0:
            N3list[i] = nf(struct_mag_stable,i,d_N3,dr_max)
        if J4!=0:
            N4list[i] = nf(struct_mag_stable,i,d_N4,dr_max)
        print(str(i)+' / '+str(N-1)+ ' mapped')


    N1list = np.array(N1list)
    N2list = np.array(N2list)
    N3list = np.array(N3list)
    N4list = np.array(N4list)

    end_time_map = time()
    time_map = np.around(end_time_map - start_time_map, 2)
    with open('N1list'+suff, 'wb') as f:
        dump(N1list, f)
    with open('N2list'+suff, 'wb') as f:
        dump(N2list, f)
    with open('N3list'+suff, 'wb') as f:
        dump(N3list, f)
    with open('N4list'+suff, 'wb') as f:
        dump(N4list, f)
    copyfile('N1list'+suff,'../N1list'+suff)
    copyfile('N2list'+suff,'../N2list'+suff)
    copyfile('N3list'+suff,'../N3list'+suff)
    copyfile('N4list'+suff,'../N4list'+suff)
    msg = 'neighbor mapping finished and dumped'
    log(msg)
    msg = 'the neighbor mapping process for a '+str(N)+' site lattice took '+str(time_map)+' s'
    log(msg) 

else:
    copyfile('../N1list'+suff,'N1list'+suff)
    copyfile('../N2list'+suff,'N2list'+suff)
    copyfile('../N3list'+suff,'N3list'+suff)
    copyfile('../N4list'+suff,'N4list'+suff)
    with open('N1list'+suff, 'rb') as f:
        N1list = load(f)
    with open('N2list'+suff, 'rb') as f:
        N2list = load(f)
    with open('N3list'+suff, 'rb') as f:
        N3list = load(f)
    with open('N4list'+suff, 'rb') as f:
        N4list = load(f)
    N = len(N1list)
    msg = 'neighbor mapping successfully read'
    log(msg) 


msg = '### CN1 = '+str(len(N1list[0]))
log(msg)
msg = '### CN2 = '+str(len(N2list[0]))
log(msg)
msg = '### CN3 = '+str(len(N3list[0]))
log(msg)
msg = '### CN4 = '+str(len(N4list[0]))
log(msg)

D1vecs, D2vecs, D3vecs, D4vecs= [[1.0,2.0,3.0]]*len(N1list[0]), [[1.0,2.0,3.0]]*len(N2list[0]), [[1.0,2.0,3.0]]*len(N3list[0]), [[1.0,2.0,3.0]]*len(N4list[0])
D1vecs, D2vecs, D3vecs, D4vecs = np.array(D1vecs), np.array(D2vecs), np.array(D3vecs), np.array(D4vecs)

sitei_coord = all_coords[int(N/3)]
#print(int(N/3))
z = np.array([0,0,1])
for idx in range(len(N1list[0])):
    sitej_coord = all_coords[N1list[int(N/3)][idx]]
    #print(sitei_coord)
    #print(sitej_coord)
    uij = sitej_coord-sitei_coord
    uij = uij/np.linalg.norm(uij)
    #print(uij)
    D1vec = D1plane*np.cross(uij,z) #+ D1z*z
    #print(D1vec)
    #D1vec = D1plane*np.cross(z,uij)
    D1vecs[idx] = D1vec
for idx in range(len(N2list[0])):
    sitej_coord = all_coords[N2list[int(N/3)][idx]]
    uij = sitej_coord-sitei_coord
    uij = uij/np.linalg.norm(uij)
    D2vec = D2plane*np.cross(uij,z) #+ D2z*z
    D2vecs[idx] = D2vec
for idx in range(len(N3list[0])):
    sitej_coord = all_coords[N3list[int(N/3)][idx]]
    uij = sitej_coord-sitei_coord
    uij = uij/np.linalg.norm(uij)
    D3vec = D3plane*np.cross(uij,z) #+ D3z*z
    D3vecs[idx] = D3vec
for idx in range(len(N4list[0])):
    sitej_coord = all_coords[N4list[int(N/3)][idx]]
    uij = sitej_coord-sitei_coord
    uij = uij/np.linalg.norm(uij)
    D4vec = D4plane*np.cross(uij,z) #+ D4z*z
    D4vecs[idx] = D4vec
msg = "### D1vecs="+str(D1vecs)
log(msg)
msg = "### D2vecs="+str(D2vecs)
log(msg)
msg = "### D3vecs="+str(D3vecs)
log(msg)
msg = "### D4vecs="+str(D4vecs)
log(msg)


temp = N1list.flatten()
corrupt = np.count_nonzero(temp == -5)
msg = 'the amount of site corruption in N1s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
log(msg)
if J2!=0:
    temp = N2list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N2s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)    
if J3!=0:
    temp = N3list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N3s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)
if J4!=0:
    temp = N4list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N4s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)


Ts = np.linspace(Tstart,Trange,div_T)
material = struct.composition.reduced_formula

start_time_mc = time()

for T in Ts:

    for epoch in range(int(num_iterations/dump_intv)):

        with ProgressBar(total=dump_intv) as progress:
            spins = MC_func(spins_init,T,dump_intv,progress)

        dumpname = 'spins_'+material+'_'+str(T)+'K'+'_'+str(np.linalg.norm(B))+'T'+'_'+str(epoch)
        with open(dumpname, 'wb') as f:
            dump(spins, f)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins[:,0],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
        plt.savefig(dumpname+'_X.png')
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins[:,1],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
        plt.savefig(dumpname+'_Y.png')
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.scatter(all_coords[:,0],all_coords[:,1],s=5.0,c=spins[:,2],cmap='turbo',vmin=-1,vmax=1,linewidths=0.1)
        plt.savefig(dumpname+'_Z.png')
        plt.close()
        msg = 'spins are dumped and plotted for temp. '+str(T)+' K and epoch '+str(epoch)
        log(msg)

        spins_init = spins


end_time_mc = time()
time_mc = np.around(end_time_mc - start_time_mc, 2)
msg = 'MC simulation have finished, analyse the dumps to get insights into the noncollinear magnetic structures'
log(msg)
msg = 'the MC simulation took '+str(time_mc)+' s'
log(msg)