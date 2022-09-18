import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector
print('velocity reflection test')
particle = PARTICLE(1, 1, c, np.array([1,1,1]), 0, )
print(particle.vel)
particle.reflect_spectular( wall)
print(particle.vel)

print(c)


# read in an STL file
print('read mesh test')
file_name = r"../../2-lego-pieces-1.snapshot.3/Lego.stl"
your_mesh = read_stl(file_name)
a  = 1

c = gen_velocity(np.array([1e-8,2e-8,3e-8]), 100000, 1)
print('velocity generation test')
print(c)



# TODO s
# create cartesian cells
