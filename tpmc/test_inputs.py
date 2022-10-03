import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity, gen_posn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector
print('velocity reflection test')
particle = PARTICLE(1, 1, c, np.array([1,1,1]), 0, )
print(particle.vel)



# test random inlet generation
INLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_inlet_v1.stl"
inlet_grid = read_stl(INLET_GRID_NAME)

r = []
for i in np.arange(1,2000):
    b = gen_posn(inlet_grid)    
    r.append(b) 

y = [item[1] for item in r]
z = [item[2] for item in r]

pts = inlet_grid.points.reshape((inlet_grid.__len__()*3, 3))

# debugging plots
plt.scatter(y, z)
plt.scatter(pts[:,1],pts[:,2],c='red')
plt.xlim([-0.002, 0.002])
plt.ylim([-0.002, 0.002])
plt.grid()
plt.savefig(r"paper_aiaa/figs/inlet_verifiction.png")



# test random velocity generation
vx = []
vy = []
vz = []
c_m = 421.92
s_n = 0.237
FREESTREAM_VEL = np.array([100, 0, 0]) # m/s, x velocity

for i in np.arange(1,20000):
    b = gen_velocity(FREESTREAM_VEL, c_m, s_n)  
    vx.append(b[0])
    vy.append(b[1])
    vz.append(b[2])

plt.figure(1)
plt.subplot(311)
plt.hist(vx,100, density=True)
plt.ylabel('X Velocity')
plt.subplot(312)
plt.hist(vy,100, density=True)
plt.ylabel('Y Velocity')
plt.subplot(313)
plt.hist(vz,100, density=True)
plt.ylabel('Z Velocity')
plt.savefig(r"paper_aiaa/figs/velocity_verifiction.png")



# generate particles for each timestep
M = 1
particle = []
for n in np.arange(0,100):
    v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
    v = np.array([100, 30*np.cos(np.pi/9), 30*np.sin(np.pi/9)])
    r = gen_posn(inlet_grid)
    print(r)
    particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=0)) # fix t_init



