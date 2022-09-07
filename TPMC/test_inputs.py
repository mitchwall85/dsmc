from stl import mesh
from utilities import reflect_spectular
import numpy as np

# test velocity reflection function
c = np.array([1, 2, 3]) # velcity vector
wall = np.array([0, 0, 5]) # wall normal vector
c_prime = reflect_spectular(c, wall) # reflected velocity

print(c_prime)
print(c)


# read in an STL file
your_mesh = mesh.Mesh.from_file(r'../2-lego-pieces-1.snapshot.3/Lego.stl')