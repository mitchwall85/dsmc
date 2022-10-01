from curses import pair_content
from distutils.ccompiler import gen_preprocess_options
from lib2to3.pgen2.token import NUMBER
from xml.etree.ElementTree import QName
from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import gen_velocity, gen_posn, KB, read_stl, in_element
from scipy.spatial.transform import Rotation as R
from scipy import special


ureg = UnitRegistry()

# problem constants
# TODO keep track of units
TUBE_D = 0.002
TUBE_L = 0.02 # add other lengths later

ALPHA_T = 0

FREESTREAM_VEL = np.array([100, 0, 0]) # m/s, x velocity
FREESTREAM_TEMP = 300 # k
KN = 100

M = 28.0134/1000/6.02e23 # mass of a N2 molecule
MOLECULE_D = 364e-12 # [m]
SIGMA_T = np.pi/4*MOLECULE_D**2

NUMBER_DENSITY = 1/KN/SIGMA_T/TUBE_D

WALL_GRID_NAME   = r"../../geometry/cylinder_d2mm_l20mm_v2.stl"
INLET_GRID_NAME  = r"../../geometry/cylinder_d2mm_l20mm_inlet_v1.stl"
OUTLET_GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm_outlet_v1.stl"

# Overrides for now
FREESTREAM_TEMP = 1 # K
NUMBER_DENSITY = 1e2 # idk what to set rn, particles/m^3

if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """

     # Mesh Info
     wall_grid = read_stl(WALL_GRID_NAME)
     inlet_grid = read_stl(INLET_GRID_NAME)
     outlet_grid = read_stl(OUTLET_GRID_NAME)

     surf_normal = np.array([1, 0, 0])


     # number flux
     c_m = np.sqrt(2*KB*FREESTREAM_TEMP/M)
     s_n = np.dot(FREESTREAM_VEL, surf_normal)/c_m
     f_n = NUMBER_DENSITY*c_m/2/np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)*s_n*(1 + special.erf(s_n)))

     particle = []
     removed_particles = []



     # time vector
     dt = 1e-5 # TODO non-even timesteps?
     t_steps = 1000
     t = np.linspace(0, t_steps*dt, t_steps)

     # particle inflow
     a_tube = np.pi/4*TUBE_D**2
     inflow_particles_flux = NUMBER_DENSITY*np.linalg.norm(FREESTREAM_VEL)
     particles_per_timestep = np.ceil(inflow_particles_flux*dt*a_tube)
     if particles_per_timestep < 100:
          print("Inflow density is < 100, maybe fix that!")

     print(particles_per_timestep) # make sure this is not to small or too big
     # loop over time
     particles_per_timestep = 5 # TODO eventually replace this with real inflows

     for n in np.arange(0,particles_per_timestep):
          # v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
          v = np.array([100, 60*np.cos(np.pi/9), 60*np.sin(np.pi/9)])
          r = gen_posn(inlet_grid)
          # r = np.array([0,0,0])
          print(r)
          particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=0)) # fix t_init

     for i in np.arange(1,t.size):
          # generate particles for each timestep
          # TODO add loop for generation back in later 

          print(t[i])
          p = 0
          while p < len(particle):
               dx = particle[p].vel * dt
               particle[p].update_posn_hist(particle[p].posn_hist[-1] + dx)

               # detect wall collisions by looping over cells
               for c in np.arange(np.shape(wall_grid.centroids)[0]):
                    # create element basis centered on centroid
                    cell_n = wall_grid.normals[c]/np.linalg.norm(wall_grid.normals[c])
                    # transform positions to new basis
                    cent = wall_grid.centroids[c]
                    cell_n_i = cell_n.dot(cent - particle[p].posn_hist[-2])
                    cell_n_f = cell_n.dot(cent - particle[p].posn_hist[-1])
                    if np.sign(cell_n_f) != np.sign(cell_n_i):

                         pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
                         intersect = particle[p].posn_hist[-2] + pct_vect*dt*particle[p].vel # TODO check

                         if in_element(wall_grid.points[c], cell_n, intersect):
                              particle[p].reflect_specular(cell_n, dt, TUBE_D, cell_n_i, cell_n_f)
               
              
               # TODO check if it collided with other particles, update posn and vel if collided
               if particle[p].exit_domain(TUBE_L):
                    removed_particles.append(particle[p])
                    particle.remove(particle[p])
               else:
                    p += 1

          if removed_particles.__len__() > 10:
               break


     # inspect particles that have left the domain
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     for p in removed_particles:
          ax.plot(p.posn_hist[:,0], p.posn_hist[:,1], p.posn_hist[:,2], '-o')
          plt.xlabel('X')
          plt.ylabel('Y')

     # add grid to plot    
     pts_wall = wall_grid.points.reshape((wall_grid.__len__()*3, 3))
     pts_inlet = inlet_grid.points.reshape((inlet_grid.__len__()*3, 3))
     # ax.scatter(pts_wall[:,0], pts_wall[:,1], pts_wall[:,2], c='m')
     # ax.scatter(pts_inlet[:,0], pts_inlet[:,1], pts_inlet[:,2], c='red')
     # ax.axis('equal')

     plt.show()
     plt.savefig('stuff.png')
     

