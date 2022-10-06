from cmath import nan
from termios import VEOL
from tkinter.filedialog import SaveFileDialog
import numpy as np
from utilities import KB
import matplotlib.pyplot as plt

class POST_PROCESS:
    """postprocessing class
    """

    def __init__(self, vol_grid: np.array, wall_grid, wp: int):
        """properties for post processing"""
        # class params
        self.step = 0
        self.particles_in_cell = []
        self.wallcells_in_cell = []
        self.wp = wp
        # volume parameters
        self.vol_grid = vol_grid
        self.temp = np.zeros(np.shape(vol_grid)[1]) 
        self.n_density = np.zeros(np.shape(vol_grid)[1]) 
        self.dof = np.shape(vol_grid)[1]
        # surface parameters
        self.wall_grid = wall_grid
        self.collision_rate = np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.pressure = np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.shear = np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.heat_tf = np.zeros(np.shape(wall_grid.centroids)[0]) 
        # cylinder surface params
        self.collision_rate_cyl = np.zeros(np.shape(vol_grid)[1])
        self.pressure_cyl = np.zeros(np.shape(vol_grid)[1])
        self.shear_cyl = np.zeros(np.shape(vol_grid)[1])
        self.heat_tf_cyl = np.zeros(np.shape(vol_grid)[1])

        # invariant params
        self.match_wallcells_in_cell() # calculate wall cells associated with each volume grid once

    def update_outputs(self, particle: list, pressure):

        self.step = self.step + 1 # add one to step
        # update cell-particle associtivity
        self.match_particles_in_cell(particle=particle)

        # volume properties
        i = 0 # cell index
        for c in self.particles_in_cell: # TODO what if there are no particles in a cell?
            n_density = c.__len__()*self.wp # add one to count density in the cell
            self.n_density[i] = self.weight_new_output(self.n_density[i], n_density) # TODO this isnt an actual density, find cell vols with volume of a pyramid
            # calculate temps
            # TODO switch to eqn 1.14 probably...
            vels = np.zeros([3,c.__len__()])
            for p in np.arange(0,c.__len__()):
                vels[:,p] = c[p].vel - c[p].bulk_vel # TODO this should only happen for specular reflected particles

            vels_mean = np.mean(vels,1)
            vels_therm = np.sum(np.square(vels_mean))
            # vels_sq_therm = np.mean(np.square(vels_mean), 0) # TODO no longer subtracting off mean
            # eqn below from: http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/kintem.html#:~:text=The%20kinetic%20temperature%20is%20the,speeds)%20in%20direct%20collisional%20transfer.
            temp = 2/3/KB*c[0].mass*vels_therm # TODO this is jank. cant do mixtures
            # temp = 1/3*c[p].mass*np.sum(vels_sq_therm)/KB # this is jank. cant do mixtures
            self.temp[i] = self.weight_new_output(self.temp[i], temp)

            i+=1 # update cell index

        # surface properties
        
        # calculate average pressure vector
        pressure_sum = [sum(x) for x in pressure]
        pres = np.array(pressure_sum)*self.wp # \\\\ multiply by wp everywhere
        self.pressure = self.weight_new_output(self.pressure, pres)

        i = 0
        for c in self.wallcells_in_cell:
            self.pressure_cyl[i] = np.mean(self.pressure[c])
            i +=1

    def match_wallcells_in_cell(self):
        """which volume cell is each wall cell in?
        """
        self.wallcells_in_cell = [[] for x in np.arange(0,self.dof)]
        for c in np.arange(np.shape(self.wall_grid.centroids)[0]):
            dist = []
            for g in np.arange(0,self.dof):
                dist.append(np.linalg.norm(self.wall_grid.centroids[c] - self.vol_grid[:,g]))

            min_cell = np.argmin(dist)
            self.wallcells_in_cell[min_cell].append(c) # organize particle into cell


    def match_particles_in_cell(self, particle):

        self.particles_in_cell = [[] for x in np.arange(0,self.dof)]  # list for grouping particles into cells

        for p in particle:
            dist = []
            for g in np.arange(0,self.dof):
                dist.append(np.linalg.norm(np.abs(p.posn_hist[-1] - self.vol_grid[:,g])))
        
            min_cell = np.argmin(dist)
            self.particles_in_cell[min_cell].append(p) # organize particle into cell
    
    def weight_new_output(self, old, new):

        return (old*(self.step - 1) + new)/self.step

    def plot_n_density(self):
        plt.plot(self.vol_grid[0,:], self.n_density)
        plt.ylim([0,max(self.n_density)])
        plt.ylabel("Number Density [particles/m^3]")
        plt.xlabel("Axial Distance [m]")
        plt.savefig('n_density.png')
        plt.close()

    def plot_temp(self):
        plt.plot(self.vol_grid[0,:], self.temp)
        plt.ylim([0,max(self.temp)])
        plt.ylabel("Temperature [k]")
        plt.xlabel("Axial Distance [m]")
        plt.savefig('temp.png')
        plt.close()

    def plot_pressure(self):
        plt.plot(self.vol_grid[0,:], self.pressure_cyl)
        plt.ylabel("Pressure [mPa]")
        plt.ylim([0,max(self.pressure_cyl)])
        plt.xlabel("Axial Distance [m]")
        plt.savefig('pressure.png')
        plt.close()

    def plot_removed_particles(self, removed_particles_time, average_window):
        removed_particles_avg = [[],[]]
        for w in np.arange(average_window,removed_particles_time[0].__len__()):
            removed_particles_avg[0].append(removed_particles_time[0][w]) 
            removed_particles_avg[1].append(np.mean(removed_particles_time[1][w-average_window:w])) 

        plt.plot(removed_particles_avg[0], removed_particles_avg[1])
        plt.ylabel("Particles Removed Per Timestep")
        plt.xlabel("Time [s]")
        plt.savefig('removed_particles.png')   
        plt.close()