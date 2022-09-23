from distutils.ccompiler import gen_preprocess_options
from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilities import gen_velocity, gen_posn, KB

ureg = UnitRegistry()

# problem constants
TUBE_D = 2*ureg.mm
TUBE_L = np.array([1, 5, 10, 20])*ureg.mm
# keep track of units
TUBE_D = 0.002
TUBE_L = 0.02 # add other lengths later

FREESTREAM_VEL = np.array([100, 0, 0]) # m/s, x velocity
FREESTREAM_TEMP = 1 # k
NUMBER_DENSITY = 1e2 # idk what to set rn, particles/m^3
M = 28.0134/1000/6.02e23 # mass of a N2 molecule


if __name__ == "__main__":
     """ loop through time for TPMC simulation
     """

     dl = 1e-4 # [m] length of tube
     dd = 5e-3 #  [m] diameter of tube
     a_tube = np.pi/4*dd**2 # [m^2] cross section area of tube

     particle = []
     removed_particles = []

     # time vector
     dt = 1e-5 # TODO non-even timesteps?
     t_steps = 100
     t = np.linspace(0, t_steps*dt, t_steps)

     # particle inflow
     inflow_particles_flux = NUMBER_DENSITY*np.linalg.norm(FREESTREAM_VEL)
     particles_per_timestep = np.ceil(inflow_particles_flux*dt*a_tube)
     if particles_per_timestep < 100:
          print("Inflow density is < 100, maybe fix that!")

     print(particles_per_timestep) # make sure this is not to small or too big
     # loop over time
     particles_per_timestep = 1 # TODO remove this soon
     for i in t:
          # generate particles for each timestep
          for n in np.arange(0,particles_per_timestep):
               v = gen_velocity(FREESTREAM_VEL, FREESTREAM_TEMP, M, KB) # TODO formulate for general inlet plane orientation
               r = gen_posn(dd)
               # print(r)
               if v[0] < 0: # skip iteration if the particle is not headed to the domain
                    continue
               particle.append(PARTICLE(mass = M, r=r, init_posn=r, init_vel=v, t_init=i))

          p = 0
          while p < len(particle):
               dx = particle[p].vel * dt
               particle[p].posn = particle[p].posn + dx
               particle[p].update_posn_hist(particle[p].posn)
               # TODO can do a simple radial coordinate reflection to test stuff
               if np.sqrt(particle[p].posn[1]**2 + particle[p].posn[2]**2) > TUBE_D:
                    wall_normal = -np.array([0, particle[p].posn[1], particle[p].posn[2]])
                    particle[p].reflect_specular(wall_normal)

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
     

