import numpy as np
from scipy import ndimage
from scipy.stats import poisson
import sys

nu_k  = float(sys.argv[1])   #lateral expansion speed normalized by kink-pair width, per second
J_pos = float(sys.argv[2])   #nucleate rate of positive kink pairs on the whole dislocation line, per second
J_neg = float(sys.argv[3])   #nucleate rate of negative kink pairs on the whole dislocation line, per second, need to be smaller than J_pos
l     =   int(sys.argv[4])   #dislocation segment length normalized by kink-pair width, 1
NUC   =   int(sys.argv[5])   #max number of positive nucleation events before cut off, 1
i     =   int(sys.argv[6])    #ID of replication

#simulation time interval set as the time required for kink to move a pixel, or multiple pixels but smaller than 1% of nucleation time, making sure the time step is still converged
delta_t = max(1./nu_k, (0.01/J_pos)//(1./nu_k)*(1./nu_k)) if nu_k != 0 else 0.01/J_pos
#pattern used for lateral expansion (ndimage.binary_dilation)
lateral = np.array([[0],[1],[0]])*np.ones(2*int(delta_t*nu_k)+1)

#initialization
plane = np.zeros((4,l), dtype=int) #initial simulation window of 4-row height
plane[0:2] = 1  #0, unslipped; >=1, slipped
timestep = 0
Nkp_pos = 0 #number of positive kink pair nucleation events
Nkp_neg = 0 #number of negative kink pair nucleation events
filledRow = -2 #number of (new) filled rows, accounting for the original two rows
slipped = 0 #slipped area in pixels

while Nkp_pos < NUC:
    #time increment
    timestep+=1

    #lateral expansion
    if np.any(plane != plane[:,[0]]): #otherwise, the dislocation is a straight line
        plane = ndimage.binary_dilation(plane, structure=lateral).astype(plane.dtype)

    #kp nucleation
    New_kp_pos = int(poisson.ppf(np.random.uniform(), J_pos*delta_t))
    New_kp_neg = int(poisson.ppf(np.random.uniform(), J_neg*delta_t))
    Nkp_pos += New_kp_pos
    Nkp_neg += New_kp_neg
    nuc_list = np.concatenate([np.ones(New_kp_pos,dtype=int), np.zeros(New_kp_neg,dtype=int)]) #all the nucleation events to happen in this time step
    np.random.shuffle(nuc_list) #randomize the event sequence
    for kp in nuc_list:
        pos_L = np.random.randint(0,l) #generate a random position along the segment
        pos_h = np.where(plane[:,pos_L]==(1-kp))[0][kp-1] #the opposite unit along that vertical line that's closest to the dislocation line
        plane[pos_h,pos_L] = kp #new kp
        if np.any(plane[-1]==1): #have reached the top    of the simulation window
            plane = np.concatenate((plane, np.zeros((1,l),dtype=int)), axis=0) #add a new row on top of the simulation window
        if np.any(plane[0] ==0): #have reached the bottom of the simulation window
            plane = np.concatenate((np.ones((1,l),dtype=int), plane),  axis=0) #add a new row at the bottom of the simulation window

    #remove filled rows for better simulation efficiency
    while np.all(plane[2]==1): #have completely filled a row of simulation window
        plane = np.delete(plane, 2, axis=0)
        filledRow+=1 #counter

#output
with open(f'{nu_k}_{J_pos:.2e}_{(1.-J_neg/J_pos):.2f}_{l}_{NUC:.0e}_{i}.txt', "w") as file:
    file.write(f'{J_pos} {((filledRow*l+len(np.where(plane==1)[0]))/l/timestep/delta_t):.4e}\n')
