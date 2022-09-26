from curses import pair_content
from distutils.ccompiler import gen_preprocess_options
from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import gen_velocity, gen_posn, KB, read_stl
from scipy.spatial.transform import Rotation as R


ureg = UnitRegistry()

# problem constants
# TUBE_D = 2*ureg.mm
# TUBE_L = np.array([1, 5, 10, 20])*ureg.mm
# keep track of units
TUBE_D = 0.002
TUBE_L = 0.02 # add other lengths later

FREESTREAM_VEL = np.array([100, 0, 0]) # m/s, x velocity
FREESTREAM_TEMP = 1 # k
NUMBER_DENSITY = 1e2 # idk what to set rn, particles/m^3
M = 28.0134/1000/6.02e23 # mass of a N2 molecule

GRID_NAME = r"../../geometry/cylinder_d2mm_l20mm.stl"

if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """

     particle = []
     removed_particles = []

     file_name = GRID_NAME
     your_mesh = read_stl(file_name)

     # time vector
     dt = 1e-5 # TODO non-even timesteps?
     t_steps = 200
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
               v = gen_velocity(FREESTREAM_VEL, FREESTREAM_TEMP, M, KB) # TODO formulate for general inlet plane orientation
               # r = gen_posn(TUBE_D)
               r = np.array([0,0,0])
               if np.sqrt(r[1]**2 + r[2]**2) > TUBE_D/2:
                    a = 1
               if v[0] < 0: # skip iteration if the particle is not headed to the domain
                    continue
               particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=i))

          print(i)
          p = 0
          while p < len(particle):
               dx = particle[p].vel * dt
               particle[p].posn = particle[p].posn + dx
               particle[p].update_posn_hist(particle[p].posn)

               # print(f"particle no: {p}")
               # detect collisions by looping over cells
               for c in np.arange(np.shape(your_mesh.centroids)[0]):
                    # create element basis centered on centroid
                    cell_n = your_mesh.normals[c]/np.linalg.norm(your_mesh.normals[c])
                    cell_t = (your_mesh.points[c][0:3] - your_mesh.centroids[0])/np.linalg.norm(your_mesh.points[c][0:3] - your_mesh.centroids[0])
                    cell_b = np.cross(cell_n, cell_t)
                    basis = np.array([cell_n, cell_t, cell_b])
                    # transform positions to new basis
                    cent = your_mesh.centroids[c]
                    cell_n_i = basis.dot(cent - particle[p].posn_hist[-2]) # TODO not all these operations needed if just x coord, most of this can be deleted
                    cell_n_f = basis.dot(cent - particle[p].posn_hist[-1])
                    if np.sign(cell_n_f[0]) != np.sign(cell_n_i[0]):
                         # this only checks if it crossed any cell, not which specific cell
                         print(np.linalg.norm(particle[p].posn[1:3])) # sanity check
                         particle[p].reflect_specular(cell_n, dt, TUBE_D)
                         break
               

               # simple radial coordinate reflection
               # if np.linalg.norm(particle[p].posn[1:3]) > TUBE_D:
               #      wall_normal = -np.array([0, particle[p].posn[1], particle[p].posn[2]])
               #      particle[p].reflect_specular(wall_normal, dt, TUBE_D) # add coeff to determine diffuse or specular

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
     

