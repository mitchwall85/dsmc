import numpy as np
from stl import mesh

from particle import PARTICLE

from utilities import read_stl

# test velocity reflection function
c = np.array([1, 2, 3])  # velcity vector
wall = np.array([0, 0, 5])  # wall normal vector

particle = PARTICLE(1, c, np.array([1,1,1]), 0, )
print(particle.vel)
particle.reflect_spectular( wall)
print(particle.vel)

print(c)


# read in an STL file
file_name = r"../../2-lego-pieces-1.snapshot.3/Lego.stl"
your_mesh = read_stl(file_name)
a  = 1
