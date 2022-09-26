from curses import pair_content
from distutils.ccompiler import gen_preprocess_options
from lib2to3.pgen2.token import NUMBER
from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import gen_velocity, gen_posn, KB, read_stl
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

GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm.stl"

# Overrides for now
FREESTREAM_TEMP = 1 # K
NUMBER_DENSITY = 1e2 # idk what to set rn, particles/m^3

if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """

     # Mesh Info
     # Wall grid
     file_name = GRID_NAME
     your_mesh = read_stl(file_name)

     surf_normal = np.array([1, 0, 0])


     # number flux
     c_m = np.sqrt(2*KB*FREESTREAM_TEMP/M)
     s_n = np.dot(FREESTREAM_VEL, surf_normal)/c_m
     f_n = NUMBER_DENSITY*c_m/2/np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)*s_n*(1 + special.erf(s_n)))

     particle = []
     removed_particles = []



     # time vector
     dt = 1e-5 # TODO non-even timesteps?
     t_steps = 100
     t = np.linspace(0, t_steps*dt, t_steps)

     # particle inflow
     a_tube = np.pi/4*TUBE_D**2
     inflow_particles_flux = NUMBER_DENSITY*np.linalg.norm(FREESTREAM_VEL)
     particles_per_timestep = np.ceil(inflow_particles_flux*dt*a_tube)
     if particles_per_timestep < 100:
          print("Inflow density is < 100, maybe fix that!")

     print(particles_per_timestep) # make sure this is not to small or too big
     # loop over time
     particles_per_timestep = 1 # TODO eventually replace this with real inflows
     for i in np.arange(1,t.size):
          # generate particles for each timestep
          for n in np.arange(0,particles_per_timestep):
               # v = gen_velocity(FREESTREAM_VEL, c_m, s_n) # TODO formulate for general inlet plane orientation
               v = np.array([100, 60*np.cos(np.pi/8), 60*np.sin(np.pi/8)])
               # r = gen_posn(TUBE_D) # TODO this looks kinda odd now, make this just on the inlet face
               r = np.array([0,0,0])
               if np.sqrt(r[1]**2 + r[2]**2) > TUBE_D/2: # TODO what was this for?
                    a = 1
               if v[0] < 0: # skip iteration if the particle is not headed to the domain
                    continue
               particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=i))

          print(t[i])
          p = 0
          while p < len(particle):
               dx = particle[p].vel * dt
               particle[p].update_posn_hist(particle[p].posn_hist[-1] + dx)

               # print(f"particle no: {p}")
               # detect collisions by looping over cells
               for c in np.arange(np.shape(your_mesh.centroids)[0]):
                    # create element basis centered on centroid
                    cell_n = your_mesh.normals[c]/np.linalg.norm(your_mesh.normals[c])
                    # transform positions to new basis
                    cent = your_mesh.centroids[c]
                    cell_n_i = cell_n.dot(cent - particle[p].posn_hist[-2]) # TODO not all these operations needed if just x coord, most of this can be deleted
                    cell_n_f = cell_n.dot(cent - particle[p].posn_hist[-1])
                    if np.sign(cell_n_f) != np.sign(cell_n_i):
                         # this only checks if it crossed any cell, not which specific cell
                         # print(np.linalg.norm(particle[p].posn[1:3])) # sanity check
                         particle[p].reflect_specular(cell_n, dt, TUBE_D, cell_n_i, cell_n_f)
                         break
               
               # TODO check if it collided with other particles, update posn and vel if collided
               if particle[p].exit_domain(TUBE_L):
                    removed_particles.append(particle[p])
                    particle.remove(particle[p])
               else:
                    p += 1


     # inspect particles that have left the domain
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     for p in removed_particles:
          ax.plot(p.posn_hist[:,0], p.posn_hist[:,1], p.posn_hist[:,2], '-o')
          plt.xlabel('X')
          plt.ylabel('Y')

     plt.show()
     plt.savefig('stuff.png')
     

