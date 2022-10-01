import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity, gen_posn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector
print('velocity reflection test')
particle = PARTICLE(1, 1, c, np.array([1,1,1]), 0, )
print(particle.vel)



# test random inlet generation
INLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_inlet.stl"
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
plt.show()
