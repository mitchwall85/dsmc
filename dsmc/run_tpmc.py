import numpy as np
from case_tpmc import CASE_TPMC
from utilities import AVOS_NUM
from tests import random_inlet_positions, random_inlet_velocity
import os


################################################################
################################################################
case_name = r'test_case_1'
# specular neutral wall
dt = 5e-6 
t_steps = 100
particles_per_timestep = 10 # choose a weighting factor such that only n particles are simulated per timestep
# freestream conditions
freestream_vel = np.array([1000, 0, 0]) # m/s, x velocity
alpha = 0 # accomidation coeff
t_tw = 1 # wall temp ratio

# geom
tube_d = 0.002
coll_cell_width = 0.002

# size of particle array
num_particles = 250

# freestream conditions
freestream_temp = 300 # k
kn = 100
m = 28.0134/1000/AVOS_NUM # mass of a N2 molecule [kg]
molecule_d = 364e-12 # [m]

# grids names ( must be continuous when assemebled)
geometry_dir = r"/home/mitch/odrive-agent-mount/OneDrive For Business/CUBoulder/ASEN6061/geometry"
wall_grid_name   = r"square_scaline_wall_v1.stl"
inlet_grid_name  = r"square_scaline_inlet_v1.stl"
outlet_grid_name = r"square_scaline_outlet_v1.stl"

# Post processing parameters
pct_window = 0.2 # check last n% of simulation
pp_tolerance = 0.1 # be within n% of inlet value to start post processing
cylinder_grids = 10 # number of points to extract from cylinder
output_dir = f"./cases/{case_name}"
average_window = 30 # average for removed particles
plot_freq = 10

case_1 = CASE_TPMC(case_name, dt, t_steps, particles_per_timestep, freestream_vel, \
            alpha, t_tw, tube_d, freestream_temp, kn, m, molecule_d, \
            wall_grid_name, inlet_grid_name, outlet_grid_name, pct_window, \
            pp_tolerance, cylinder_grids, output_dir, average_window, plot_freq, \
            num_particles, coll_cell_width, geometry_dir)


case_1.execute_case()
# case_1.readme_output()

# TODO add post processing functionality

