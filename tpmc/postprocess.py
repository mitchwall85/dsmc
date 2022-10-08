from cmath import nan
from termios import VEOL
from tkinter.filedialog import SaveFileDialog
import numpy as np
from utilities import KB, M
import matplotlib.pyplot as plt

class POST_PROCESS:
    """postprocessing class
    """

    def __init__(self, vol_grid: np.array, wall_grid, wp: int, dt: float):
        """properties for post processing"""
        # class params
        self.step = 0
        self.dt = dt
        self.particles_in_cell = []
        self.wallcells_in_cell = []
        self.volcell_volume = []
        self.areacell_volume = []
        self.wp = wp
        # volume parameters
        self.vol_grid = vol_grid
        self.temp = np.zeros(np.shape(vol_grid)[1]) 
        self.n_density = np.zeros(np.shape(vol_grid)[1]) 
        self.dof = np.shape(vol_grid)[1]
        # surface parameters
        self.wall_grid = wall_grid
        self.collision_rate = np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.pressure =       np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.shear =          np.zeros(np.shape(wall_grid.centroids)[0]) 
        self.heat_tf =        np.zeros(np.shape(wall_grid.centroids)[0]) 
        # cylinder surface params
        self.collision_rate_cyl = np.zeros(np.shape(vol_grid)[1])
        self.pressure_cyl =       np.zeros(np.shape(vol_grid)[1])
        self.shear_cyl =          np.zeros(np.shape(vol_grid)[1])
        self.heat_tf_cyl =        np.zeros(np.shape(vol_grid)[1])

        # invariant params
        self.match_wallcells_in_cell() # calculate wall cells associated with each volume grid once
        self.volume_wall_cell()
        self.area_wall_cell()

    def update_outputs(self, particle: list, pressure, ener: np.array, axial_pressure: list):

        self.step = self.step + 1 # add one to step
        # update cell-particle associtivity
        self.match_particles_in_cell(particle=particle)

        # volume properties
        i = 0 # cell index
        for c in self.particles_in_cell:
            n_density = c.__len__()*self.wp/self.volcell_volume[i] # count macroparticles*wp per cell volume
            self.n_density[i] = self.weight_new_output(self.n_density[i], n_density)

            # jank fix if there are no partilces in the cell
            if c.__len__() > 0:
                # calculate temps
                vels = np.zeros([c.__len__(),3])
                for p in np.arange(0,c.__len__()):
                    vels[p,:] = c[p].vel

                temp_dir = np.zeros(3)
                for t in np.arange(0,3):
                    temp_dir[t] = M/KB*(np.mean(np.square(vels[t,:])) - np.square(np.mean(vels[t,:]))) # the mean of the square and the square of the mean

                temp = np.mean(temp_dir)

                self.temp[i] = self.weight_new_output(self.temp[i], temp)
            else:
                self.temp[i] = self.weight_new_output(self.temp[i], self.temp[i])

            i+=1 # update cell index

        # collision frequency
        i = 0 # cell index
        for c in self.wallcells_in_cell:
            n_coll = 0
            for elem in c:
                n_coll = n_coll + pressure[elem].__len__()
            self.collision_rate_cyl[i] = n_coll/self.areacell_volume[i]/self.dt

            i+=1 # update cell index

        # surface properties
        # calculate average pressure vector
        pressure_sum = [sum(x) for x in pressure]
        pres = np.array(pressure_sum)*self.wp # \\\\ TODO multiply by wp everywhere
        self.pressure = self.weight_new_output(self.pressure, pres)

        # calculate axial pressure vector
        axial_pressure_sum = [sum(x) for x in axial_pressure] # TODO cant test this with specular
        axial_pres = np.array(axial_pressure_sum)*self.wp 
        self.shear = self.weight_new_output(self.shear, axial_pres)

        i = 0
        for c in self.wallcells_in_cell:
            self.pressure_cyl[i] = np.mean(self.pressure[c])
            self.shear_cyl[i] = np.mean(self.shear[c])
            self.heat_tf_cyl[i] = np.sum(ener[c])  # TODO these might come out to all zeros...
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
        
    def volume_wall_cell(self):
        """find volume of each volume cell
        """
        self.volcell_volume = [0]*self.vol_grid[1,:].__len__()
        vol_cell_id = 0
        for c in self.wallcells_in_cell:
            vol = [0] # total volume of cell
            for elem in c:
                a = self.wall_grid.areas[elem]
                v = self.wall_grid.centroids[elem] - self.vol_grid[:, vol_cell_id]  # vector from cell centroid to cell center
                h = np.abs(np.linalg.norm(v)*np.dot(v,self.wall_grid.normals[elem])/np.linalg.norm(v)/np.linalg.norm(self.wall_grid.normals[elem]))
                # https://www.omnicalculator.com/math/pyramid-volume
                vol = vol + 1/3*a*2*h

            self.volcell_volume[vol_cell_id] = vol
            vol_cell_id+=1

    def area_wall_cell(self):
        """find area of each volume cell
        """
        self.areacell_volume = [0]*self.vol_grid[1,:].__len__()
        vol_cell_id = 0
        for c in self.wallcells_in_cell:
            area = [0] # total volume of cell
            for elem in c:
                a = self.wall_grid.areas[elem]
                area = area + a

            self.areacell_volume[vol_cell_id] = area
            vol_cell_id+=1
    
    def weight_new_output(self, old, new):

        return (old*(self.step - 1) + new)/self.step

    def plot_shear(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.shear_cyl)
        plt.ylim([0,max(self.shear_cyl)])
        plt.ylabel("Axial Stress [Pa]")
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_axial_stress.png")
        plt.close()

    def plot_heat_tfr(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.heat_tf_cyl)
        plt.ylim([0,max(self.heat_tf_cyl)])
        plt.ylabel("Heat Transfer [W/m^2]")
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_heat_tfr.png")
        plt.close()

    def plot_n_coll(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.collision_rate_cyl)
        plt.ylim([0,max(self.collision_rate_cyl)])
        plt.ylabel("Collision Frequency [1/m^2/s]")
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_coll_freq.png")
        plt.close()

    def plot_n_density(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.n_density)
        plt.ylim([0, max(self.n_density)])
        plt.ylabel("Number Density [particles/m^3]")
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_n_density.png")
        plt.close()

    def plot_temp(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.temp)
        plt.ylim([0,max(self.temp)])
        plt.ylabel("Temperature [k]")
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_temp.png")
        plt.close()

    def plot_pressure(self, output_dir, t_tw, alpha):
        plt.clf()
        plt.plot(self.vol_grid[0,:], self.pressure_cyl)
        plt.ylabel("Pressure [Pa]")
        plt.ylim([0,max(self.pressure_cyl)])
        plt.xlabel("Axial Distance [m]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_pressure.png")
        plt.close()

    def plot_removed_particles(self, output_dir, t_tw, alpha, removed_particles_time, removed_outlet, removed_inlet, average_window):
        removed_particles_avg = [[],[]]
        removed_inlet_avg = []
        removed_outlet_avg = []
        for w in np.arange(average_window,removed_particles_time[0].__len__()):
            removed_particles_avg[0].append(removed_particles_time[0][w]) 
            removed_particles_avg[1].append(np.mean(removed_particles_time[1][w-average_window:w])) 
            removed_inlet_avg.append(np.mean(removed_inlet[w-average_window:w])) 
            removed_outlet_avg.append(np.mean(removed_outlet[w-average_window:w])) 

        plt.clf()
        plt.plot(removed_particles_avg[0], removed_particles_avg[1], label='Total Removed')
        plt.plot(removed_particles_avg[0], removed_inlet_avg, label='Inlet')
        plt.plot(removed_particles_avg[0], removed_outlet_avg, label='Outlet')
        plt.legend()
        plt.ylabel("Particles Removed Per Timestep")
        plt.xlabel("Time [s]")
        plt.grid()
        plt.savefig(f"{output_dir}ttw{t_tw}_alp{alpha}_removed_particles.png") 
        plt.close
        