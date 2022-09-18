from pint import UnitRegistry
import numpy as np
from particle import PARTICLE
import matplotlib.pyplot as plt

ureg = UnitRegistry()



TUBE_D = 2*ureg.mm
TUBE_L = np.array([1, 5, 10, 20])*ureg.mm
# keep track of units
TUBE_D = 0.002
TUBE_L = 1 # add other lengths later


if __name__ == "__main__":
     a = 1
     print(TUBE_L)
     """ loop through time 
     """

     dl = 0.1*ureg.mm
     dd = 0.2*ureg.mm
     dl = 0.1*TUBE_D
     dd = 0.1*TUBE_L

     # inlet domain
     r_init = np.array([0,0,0])
     m = 1
     r = 1e-5
     particle = [PARTICLE(mass = m, r=r, init_posn=r_init, init_vel=np.array([1,0.5,1e-1]), t_init=0),
                 PARTICLE(mass = m, r=r, init_posn=r_init, init_vel=np.array([1,0.5,2e-1]), t_init=0),
                 PARTICLE(mass = m, r=r, init_posn=r_init, init_vel=np.array([1,0.5,3e-1]), t_init=0)]
     removed_particles = []

     # timestep params
     t = np.linspace(0, 10, 10)
     dt = t[1] - t[0] # assume even timestep?



     for i in t:
          p = 0
          while p < len(particle):
               dx = particle[p].vel * dt
               particle[p].posn = particle[p].posn + dx
               particle[p].update_posn_hist(particle[p].posn)

               # TODO check if it collided with other particles
               # update posn and vel if collided

               if particle[p].exit_domain(TUBE_L):
                    removed_particles.append(particle[p])
                    particle.remove(particle[p])
               else:
                    p += 1


     # print(p.posn_hist)
     print(np.shape(removed_particles[0].posn_hist)[0])
     print(np.shape(removed_particles[1].posn_hist)[0])
     print(np.shape(removed_particles[2].posn_hist)[0])

     # plots
     ax = plt.axes(projection='3d')
     ax.scatter3D(removed_particles[2].posn_hist[:,0], removed_particles[2].posn_hist[:,1], removed_particles[2].posn_hist[:,2])
     plt.savefig('stuff.png')
     plt.show()

     for tube_len in TUBE_L: