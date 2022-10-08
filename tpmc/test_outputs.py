import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities import read_stl, KB

with open('particle_0p5e6_100_50part.pkl', 'rb') as f:
    particle = pickle.load(f)
    
# create grid for cylinder 
M = 1 # TODO replace with correct mass
n = 10
n_0 = 0.0
n_l = 0.02
dx = (n_l-n_0)/n
output_grids = np.vstack([np.linspace(n_0 + dx/2, n_l - dx/2, n), np.zeros(n), np.zeros(n)])
dof = np.shape(output_grids)[1]
particles_in_cell = [[] for x in np.arange(0,dof)]
temp_x = [] # verify position with histogram fcn
for p in particle:
    dist = []
    temp_x.append(p.posn_hist[-1][0])
    for g in np.arange(0,dof):
        # dist.append(np.linalg.norm(np.abs(p.posn_hist[-1] - output_grids[:,g])))
        dist.append(np.linalg.norm(np.abs(p.posn_hist[-1] - output_grids[:,g])))
    
    
    min_cell = np.argmin(dist)
    particles_in_cell[min_cell].append(p) # organize particle into cell

n_density = [0]*dof
blk_vel = [0]*dof
T = [0]*dof
i = 0
for c in particles_in_cell:
    n_density[i] = c.__len__() # add one to count density in the cell
    vels = np.zeros([3,c.__len__()])
    for p in np.arange(0,c.__len__()):
        vels[:,p] = c[p].vel
    vels_mean = np.mean(vels,1)
    vels_therm = np.mean(vels.transpose() - vels_mean, 0)

    T[i] = 1/3*M*np.sum(np.square(vels_therm))/KB

    i+=1

# confirm with number density histogram
plt.hist(temp_x, n, range=[n_0, n_l], edgecolor='black', )  # verify position with histogram fcn

# plot number density
plt.plot(output_grids[0], n_density, '-o')
plt.xlim([n_0, n_l])
plt.ylim([0, max(n_density)])
plt.show()

# plot temperature
plt.figure()
plt.plot(output_grids[0], T, '-o')
plt.xlim([n_0, n_l])
plt.title('Temperature')
plt.show()

# plot pressure vs. x
WALL_GRID_NAME   = r"../../geometry/cylinder_d2mm_l20mm_v1.stl"
wall_grid = read_stl(WALL_GRID_NAME)

with open('pressure_0p5e6_100_50part.pkl', 'rb') as f:
    pressure = pickle.load(f)

# calculate average pressure vector
t_len = pressure.__len__()
no_wall_elems = np.shape(wall_grid.centroids)[0]
pressure_avg = np.zeros([no_wall_elems])
window = 90 # width of pressure average
for t in np.arange(t_len - window, t_len):
    pressure_sum = [sum(x) for x in pressure[t]]
    pressure_avg = pressure_avg + np.array(pressure_sum) # TODO multiply by wp

pressure_avg = np.divide(pressure_avg,window)

# group wall elements with grid output points
# create grid for cylinder 
wallcells_in_cell = [[] for x in np.arange(0,dof)]
for c in np.arange(np.shape(wall_grid.centroids)[0]):
    dist = []
    for g in np.arange(0,dof):
        dist.append(np.linalg.norm(wall_grid.centroids[c] - output_grids[:,g]))

    min_cell = np.argmin(dist)
    wallcells_in_cell[min_cell].append(c) # organize particle into cell

pressure_cell = []
for c in wallcells_in_cell:
    pressure_cell.append(np.mean(pressure_avg[c]))

plt.figure()
plt.plot(output_grids[0], pressure_cell, '-o')
plt.show()

