from audioop import avg
from curses import pair_content
from distutils.ccompiler import gen_preprocess_options
from lib2to3.pgen2.token import NUMBER
from xml.etree.ElementTree import QName
from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
from postprocess import POST_PROCESS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import gen_velocity, gen_posn, KB, read_stl, in_element, start_postproc
from scipy.spatial.transform import Rotation as R
from scipy import special
import pickle


ureg = UnitRegistry()

# problem constants
# TODO keep track of units
TUBE_D = 0.002
TUBE_L = 0.02 # add other lengths later

ALPHA_T = 0

FREESTREAM_VEL = np.array([1000, 0, 0]) # m/s, x velocity
FREESTREAM_TEMP = 300 # k
KN = 100

M = 28.0134/1000/6.02e23 # mass of a N2 molecule
MOLECULE_D = 364e-12 # [m]
SIGMA_T = np.pi/4*MOLECULE_D**2

NUMBER_DENSITY = 1/KN/SIGMA_T/TUBE_D

WALL_GRID_NAME   = r"../../geometry/cylinder_d2mm_l20mm_v1.stl"
INLET_GRID_NAME  = r"../../geometry/cylinder_d2mm_l20mm_inlet_v1.stl"
OUTLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_outlet_v1.stl"

# Post processing parameters
PCT_WINDOW = 0.2 # check last n% of simulation
PP_TOLERANCE = 0.20 # be within n% of inlet value to start post processing
POST_PROC = False # start by not post processing results until convergence criteria reached
CYLINDER_GRIDS = 10 # number of points to extract from cylinder

if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """

     # Mesh Info
     wall_grid = read_stl(WALL_GRID_NAME)
     inlet_grid = read_stl(INLET_GRID_NAME)
     outlet_grid = read_stl(OUTLET_GRID_NAME)
     surf_normal = np.array([1, 0, 0])
     no_wall_elems = np.shape(wall_grid.centroids)[0]

     # output grid info
     n_0 = inlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
     n_l = outlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
     dx = (n_l-n_0)/CYLINDER_GRIDS
     output_grids = np.vstack([np.linspace(n_0 + dx/2, n_l - dx/2, CYLINDER_GRIDS), np.zeros(CYLINDER_GRIDS), np.zeros(CYLINDER_GRIDS)])
     # create output class
     post_proc = POST_PROCESS(output_grids, wall_grid,)

     # number flux
     c_m = np.sqrt(2*KB*FREESTREAM_TEMP/M)
     v_bar = np.dot(FREESTREAM_VEL, surf_normal)
     s_n = (np.dot(v_bar, surf_normal)/c_m)[0]# A.26 TODO why do I need to select only 1 component?
     f_n = NUMBER_DENSITY*c_m/2/np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)*s_n*(1 + special.erf(s_n))) # A.27

     # grid info
     inlet_a = 0
     for c in inlet_grid.areas:
          inlet_a = inlet_a = c

     # time step info
     dt = 0.5e-6 # TODO non-even timesteps? automatic timestep generation?
     t_steps = 200
     # particle inflow
     real_particles_per_timestep = np.ceil(f_n*dt*inlet_a) # A.28
     particles_per_timestep = 10 # choose a weighting factor such that only n particles are simulated per timestep
     wp = real_particles_per_timestep/particles_per_timestep # weighting factor

     particle = []
     removed_particles = []
     removed_particles_time = [[],[]] # 2d list for plotting removed particles
     pres_time = [] # pressure matrix over all timesteps

     i = 1
     removed = 0
     while i < t_steps:
          # generate particles for each timestep
          for n in np.arange(0,particles_per_timestep):
               v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
               # v = np.array([100, 30*np.cos(np.pi/9), 30*np.sin(np.pi/9)])
               r = gen_posn(inlet_grid)
               # print(r)
               particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=0)) # fix t_init

          print(f"Total Time: {i*dt}")
          print(f"Time Steps: {100*(i)/t_steps} %")
          p = 0
          removed = 0
          pres = [[] for x in np.arange(0,no_wall_elems)] # pressure matrix for current timestep
          while p < len(particle):
               dx = particle[p].vel * dt
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
                         intersect = particle[p].posn_hist[-2] + pct_vect*dt*particle[p].vel # TODO check

                         if in_element(wall_grid.points[c], cell_n, intersect):
                              dm = particle[p].reflect_specular(cell_n, dt, TUBE_D, cell_n_i, cell_n_f)
                              pres_scalar = np.linalg.norm(dm/dt/wall_grid.areas[c])
                              pres[c].append(pres_scalar) # pressure contribution from reflection
               
              
               if particle[p].exit_domain(TUBE_L):
                    removed_particles.append(particle[p])
                    particle.remove(particle[p])
                    removed+=1
               else:
                    p += 1
          pres_time.append(pres) # tack on new pressure matrix for new time step

          print(f'Particles removed: {removed}')

          # find now many particles leave the domain per timestep
          removed_particles_time[0].append(i*dt)
          removed_particles_time[1].append(removed)

          # detect if steady state is reached and if post processing should start
          if not POST_PROC:
               if start_postproc(PCT_WINDOW, PP_TOLERANCE, removed_particles_time, particles_per_timestep, i):
                    POST_PROC = True # just turn this flag on once
          else:
               a = 1
               # do post processing stuff here
               

          i+=1 # add to timestep index, continue to next timestep

     average_window = 30
     removed_particles_avg = [[],[]]
     for w in np.arange(average_window,removed_particles_time[0].__len__()):
          removed_particles_avg[0].append(removed_particles_time[0][w]) 
          removed_particles_avg[1].append(np.mean(removed_particles_time[1][w-average_window:w])) 

     plt.plot(removed_particles_avg[0], removed_particles_avg[1])
     plt.ylabel("Particles Removed Per Timestep")
     plt.xlabel("Time [s]")
     

     plt.show()
     plt.savefig('removed_particles.png')
     

