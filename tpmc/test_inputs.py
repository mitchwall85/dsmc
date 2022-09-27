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
TUBE_D = 0.002
INLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_inlet.stl"
inlet_grid = read_stl(INLET_GRID_NAME)
r = np.zeros(3)
for i in np.arange(1,2000):
    r = np.vstack([r,gen_posn(TUBE_D)])   

plt.scatter(r[:,1],r[:,2])
plt.ylim([-1.1*TUBE_D/2, 1.1*TUBE_D/2])
plt.xlim([-1.1*TUBE_D/2, 1.1*TUBE_D/2])
plt.show()



# TODO s
# create cartesian cells
