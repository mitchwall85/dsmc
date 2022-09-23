import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector
print('velocity reflection test')
particle = PARTICLE(1, 1, c, np.array([1,1,1]), 0, )
print(particle.vel)
particle.reflect_specular(wall)
print(particle.vel)

print(c)


# read in an STL file
print('read mesh test')
file_name = r"../../geometry/cylinder_d2mm_l20mm.stl"
your_mesh = read_stl(file_name)
points = np.reshape(your_mesh.points,(np.shape(your_mesh.points)[0]*3,3))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2])
plt.show()

c = gen_velocity(np.array([1e-8,2e-8,3e-8]), 100000, 1)
print('velocity generation test')
print(c)



# TODO s
# create cartesian cells
