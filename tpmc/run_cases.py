import sys
import numpy as np

# run cases for TPMC code



# specular neutral wall
DT = 1e-6 # TODO non-even timesteps? automatic timestep generation?
T_STEPS = 2000
PARTICLES_PER_TIMESTEP = 10 # choose a weighting factor such that only n particles are simulated per timestep
# freestream conditions
FREESTREAM_VEL = np.array([1000, 0, 0]) # m/s, x velocity
ALPHA = 1 # accomidation coeff
T_TW = 1 # wall temp ratio

import os

os.system(f"python3 tpmc.py 0.000001 2000 100 1000 0 1")  