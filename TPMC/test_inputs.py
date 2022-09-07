from utilities import reflect_spectular
import numpy as np

c = np.array([1, 2, 3])
wall = np.array([0, 0, 5])

c_prime = reflect_spectular(c, wall)

print(c_prime)
print(c)
