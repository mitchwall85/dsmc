import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity, gen_posn, KB

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector
print('velocity reflection test')
FREESTREAM_VEL = np.array([0, 0, 0]) # m/s, x velocity
M = 28.0134/1000/6.02e23 # mass of a N2 molecule [kg]
particle = PARTICLE(1, 1, c, np.array([1,1,1]), 0, FREESTREAM_VEL)
print(particle.vel)



# test random inlet generation
INLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_inlet_v1.stl"
inlet_grid = read_stl(INLET_GRID_NAME)

r = []
for i in np.arange(1,800):
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
plt.xlabel('Y Coordinate [m]')
plt.ylabel('Z Coordinate [m]')
plt.grid()
plt.savefig(r"figs/inlet_posn_verifiction.png")



# test random velocity generation
vx = []
vy = []
vz = []
FREESTREAM_TEMP = 300 # [k]
surf_normal = np.array([1, 0, 0])

c_m = np.sqrt(2*KB*FREESTREAM_TEMP/M)
v_bar = np.dot(FREESTREAM_VEL, surf_normal)
s_n = (np.dot(v_bar, surf_normal)/c_m)[0]# A.26

for i in np.arange(1,50000):
    b = gen_velocity(FREESTREAM_VEL, c_m, s_n)  
    vx.append(b[0])
    vy.append(b[1])
    vz.append(b[2])

# analitical maxwell-boltzmann
vi = np.linspace(-2000,2000,1000)
mb = np.sqrt(M/2/np.pi/KB/FREESTREAM_TEMP)*np.exp(-M*vi**2/2/KB/FREESTREAM_TEMP)
# vi = np.linspace(0,2000,1000)
# https://openstax.org/books/university-physics-volume-2/pages/2-4-distribution-of-molecular-speeds#:~:text=Maxwell%2DBoltzmann%20Distribution%20of%20Speeds&text=f%20(%20v%20)%20%3D%204%20%CF%80,2%20k%20B%20T%20)%20)%20.
# mb_speed = 4/np.sqrt(np.pi)*(M/2/KB/FREESTREAM_TEMP)**(3/2)*vi_speed**2*np.exp(-M*vi_speed**2/2/KB/FREESTREAM_TEMP)
mb_speed = 1/4*vi*mb
mb_speed = 0.002*mb_speed/np.max(mb_speed)

plt.figure(1)
plt.subplot(311)
plt.hist(vx,100, density=True)
plt.plot(vi,mb_speed)
plt.xlim([0, max(vi)])
plt.ylim([0, 1.1*max(mb_speed)])
plt.ylabel('vX PDF')
plt.subplot(312)
plt.hist(vy,100, density=True)
plt.plot(vi,mb)
plt.ylabel('vY PDF')
plt.subplot(313)
plt.hist(vz,100, density=True)
plt.plot(vi,mb)
plt.ylabel('vZ PDF')
plt.xlabel('Velocity [m/s]')
plt.savefig(r"figs/inlet_velocity_verifiction.png")



# generate particles for each timestep
particle = []
for n in np.arange(0,100):
    v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
    v = np.array([100, 30*np.cos(np.pi/9), 30*np.sin(np.pi/9)])
    r = gen_posn(inlet_grid)
    print(r)
    particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=0, bulk_vel= FREESTREAM_VEL)) # fix t_init



