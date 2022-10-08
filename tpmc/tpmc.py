from turtle import st
import numpy as np
from particle import PARTICLE
from postprocess import POST_PROCESS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import KB, read_stl, in_element, start_postproc, gen_posn, gen_velocity, AVOS_NUM
from scipy.spatial.transform import Rotation as R
from scipy import special
import sys
# General things to add
# TODO add pint
# TODO non-even timesteps? automatic timestep generation?

# # case specified parameters
# # run parameters:
# DT = float(sys.argv[1]) 
# T_STEPS = int(sys.argv[2])
# PARTICLES_PER_TIMESTEP = int(sys.argv[3]) # choose a weighting factor such that only n particles are simulated per timestep
# # freestream conditions
# FREESTREAM_VEL = np.array([float(sys.argv[4]), 0, 0]) # m/s, x velocity
# ALPHA = float(sys.argv[5]) # accomidation coeff
# T_TW = float(sys.argv[6]) # wall temp ratio

# specular neutral wall
DT = 1e-6 
T_STEPS = 1000
PARTICLES_PER_TIMESTEP = 20 # choose a weighting factor such that only n particles are simulated per timestep
# freestream conditions
FREESTREAM_VEL = np.array([1000, 0, 0]) # m/s, x velocity
ALPHA = 0 # accomidation coeff
T_TW = 1 # wall temp ratio

# tube geom
TUBE_D = 0.002

# freestream conditions
FREESTREAM_TEMP = 300 # k
KN = 100
M = 28.0134/1000/AVOS_NUM # mass of a N2 molecule [kg]
MOLECULE_D = 364e-12 # [m]

# grids names ( must be continuous when assemebled)
WALL_GRID_NAME   = r"../../geometry/cylinder_d2mm_l20mm_v1.stl"
INLET_GRID_NAME  = r"../../geometry/cylinder_d2mm_l20mm_inlet_v1.stl"
OUTLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_outlet_v1.stl"

# Post processing parameters
PCT_WINDOW = 0.2 # check last n% of simulation
PP_TOLERANCE = 0.1 # be within n% of inlet value to start post processing
CYLINDER_GRIDS = 20 # number of points to extract from cylinder
OUTPUT_DIR = r"./figs/"
AVERAGE_WINDOW = 30 # average for removed particles
PLOT_FREQ = 10

if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """
     # Mesh Info
     wall_grid = read_stl(WALL_GRID_NAME)
     inlet_grid = read_stl(INLET_GRID_NAME)
     outlet_grid = read_stl(OUTLET_GRID_NAME)
     surf_normal = np.array([1, 0, 0])
     no_wall_elems = np.shape(wall_grid.centroids)[0]
     # grid info
     inlet_a = 0
     for c in inlet_grid.areas:
          inlet_a = inlet_a + c

     # number flux
     sigma_t = np.pi/4*MOLECULE_D**2 # collision crossection
     number_density = 1/KN/sigma_t/TUBE_D
     c_m = np.sqrt(2*KB*FREESTREAM_TEMP/M)
     v_bar = np.dot(FREESTREAM_VEL, surf_normal)
     s_n = (np.dot(v_bar, surf_normal)/c_m)[0]# A.26
     f_n = number_density*c_m/2/np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)*s_n*(1 + special.erf(s_n))) # A.27
     # particle inflow
     real_PARTICLES_PER_TIMESTEP = np.ceil(f_n*DT*inlet_a) # A.28
     wp = real_PARTICLES_PER_TIMESTEP/PARTICLES_PER_TIMESTEP # weighting factor

     # output grid info
     n_0 = inlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
     n_l = outlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
     dx = (n_l-n_0)/CYLINDER_GRIDS
     output_grids = np.vstack([np.linspace(n_0 + dx/2, n_l - dx/2, CYLINDER_GRIDS), np.zeros(CYLINDER_GRIDS), np.zeros(CYLINDER_GRIDS)])
     # create output class
     post_proc = POST_PROCESS(output_grids, wall_grid, wp, DT)

     # initalize variables
     particle = []
     removed_particles_time = [[],[]] # 2d list for plotting removed particles
     removed_particles_inlet = []
     removed_particles_outlet = []
     pres_time = [] # pressure matrix over all timesteps

     i = 1 # timestep index
     removed = 0 # initalize number of removed particle objects
     removed_outlet = 0 
     removed_inlet = 0 
     start_post = False # start by not post processing results until convergence criteria reached
     while i < T_STEPS:
          # generate particles for each timestep
          for n in np.arange(0,PARTICLES_PER_TIMESTEP):
               v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
               r = gen_posn(inlet_grid)
               particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=0, bulk_vel=FREESTREAM_VEL)) # fix t_init


          p = 0
          removed = 0
          removed_outlet = 0
          removed_inlet = 0
          pres = [[] for x in np.arange(0,no_wall_elems)] # pressure matrix for current timestep
          ener = np.array([0]*no_wall_elems) # thermal energy matrix
          axial_stress = [[] for x in np.arange(0,no_wall_elems)] # pressure matrix for current timestep
          while p < len(particle):
               dx = particle[p].vel * DT
               particle[p].update_posn_hist(particle[p].posn_hist[-1] + dx)

               # detect wall collisions by looping over cells
               for c in np.arange(np.shape(wall_grid.centroids)[0]):
                    # create element basis centered on centroid
                    cell_n = wall_grid.normals[c]
                    # transform positions to new basis
                    cent = wall_grid.centroids[c]
                    cell_n_i = cell_n.dot(cent - particle[p].posn_hist[-2])
                    cell_n_f = cell_n.dot(cent - particle[p].posn_hist[-1])
                    if np.sign(cell_n_f) != np.sign(cell_n_i):
                         cell_n_mag = np.linalg.norm(wall_grid.normals[c]) # saves time by moving this here
                         cell_n_i = cell_n_i/cell_n_mag
                         cell_n_f = cell_n_f/cell_n_mag

                         pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
                         intersect = particle[p].posn_hist[-2] + pct_vect*DT*particle[p].vel

                         if in_element(wall_grid.points[c], cell_n, intersect):
                              if np.random.rand(1) > ALPHA:
                                   dm = particle[p].reflect_specular(cell_n, DT, cell_n_i, cell_n_f)
                              else:
                                   dm, de = particle[p].reflect_diffuse(cell_n, DT, cell_n_i, cell_n_f, T_TW, c_m)
                                   # energy change
                                   ener[c] = ener[c] + de*M/DT/2 # convert to Joules
                              # pressure contribution from reflection
                              pres_scalar = np.linalg.norm(dm[1:3]/DT/wall_grid.areas[c]) # not a very clevery way to get normal compoent
                              pres[c].append(pres_scalar) 
                              # axial pressure contribution from reflection
                              axial_stress_scalar = np.linalg.norm(dm[0]/DT/wall_grid.areas[c])
                              axial_stress[c].append(axial_stress_scalar)
              
               if particle[p].exit_domain_outlet(n_l):
                    particle.remove(particle[p])
                    removed_outlet+=1
                    removed+=1
               if particle[p].exit_domain_inlet(n_0):
                    particle.remove(particle[p])
                    removed_inlet+=1
                    removed+=1
               else:
                    p += 1
                    
          # find now many particles leave the domain per timestep
          removed_particles_time[0].append(i*DT)
          removed_particles_time[1].append(removed)
          removed_particles_inlet.append(removed_inlet)
          removed_particles_outlet.append(removed_outlet)
          # plot removed particles with time
          post_proc.plot_removed_particles(OUTPUT_DIR, T_TW, ALPHA, removed_particles_time, removed_particles_outlet, removed_particles_inlet, AVERAGE_WINDOW) # dont hardcode this value
          
          # print status to terminal
          print(f"--------------------------------------------------------------------------")
          print(f'Particles removed: {removed}')
          print(f"Total Time: {i*DT}")
          print(f"Time Steps: {100*(i)/T_STEPS} %")

          # detect if steady state is reached and if post processing should start
          if not start_post:
               if start_postproc(PCT_WINDOW, PP_TOLERANCE, removed_particles_time, PARTICLES_PER_TIMESTEP, i):
                    start_post = True # just turn this flag on once
          else:
               # update outputs
               post_proc.update_outputs(particle, pres, ener, axial_stress)
               print(f"Post Processing...")

               # create plots
               if i%PLOT_FREQ == 0:
                    post_proc.plot_n_density(OUTPUT_DIR, T_TW, ALPHA)
                    post_proc.plot_temp(OUTPUT_DIR, T_TW, ALPHA)
                    post_proc.plot_pressure(OUTPUT_DIR, T_TW, ALPHA)
                    post_proc.plot_heat_tfr(OUTPUT_DIR, T_TW, ALPHA)
                    post_proc.plot_n_coll(OUTPUT_DIR, T_TW, ALPHA)
                    post_proc.plot_shear(OUTPUT_DIR, T_TW, ALPHA)

          i+=1 # add to timestep index, continue to next timestep


