
from turtle import pos, st
import numpy as np
from particle import PARTICLE
from postprocess import POST_PROCESS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import KB, read_stl, in_element, start_postproc, gen_posn, gen_velocity, AVOS_NUM, make_directory
from scipy.spatial.transform import Rotation as R
from scipy import special
import sys
import os

# General things to add
# TODO add pint
# TODO non-even timesteps? automatic timestep generation?

class CASE_TPMC:
    """ TPMC case class. Includes all options needed to run TPMC model.
    """


    def __init__(self, case_name: str, dt: float, t_steps: int, particles_per_timestep: int, freestream_vel: np.array, \
        alpha: float, t_tw: float, tube_d: float, freestream_temp: float, kn: float, m: float, molecule_d: \
        float, wall_grid_name: str, inlet_grid_name: str, outlet_grid_name: str, pct_window: float, \
        pp_tolerance: float, cylinder_grids: int, output_dir: float, average_window: int, plot_freq: int ):

        self.case_name = case_name
        # specular neutral wall
        self.dt = dt
        self.t_steps = t_steps
        self.particles_per_timestep = particles_per_timestep # choose a weighting factor such that only n particles are simulated per timestep
        # freestream conditions
        self.freestream_vel = freestream_vel # m/s, x velocity
        self.alpha = alpha # accomidation coeff
        self.t_tw = t_tw # wall temp ratio

        # tube geom
        self.tube_d = tube_d # TODO replace this with just a reference length, I think thats all its used for

        # freestream conditions
        self.freestream_temp = freestream_temp # k
        self.kn = kn
        self.m = m # mass of a N2 molecule [kg]
        self.molecule_d = molecule_d  # [m]

        # grids names ( must be continuous when assemebled)
        self.wall_grid_name = wall_grid_name
        self.inlet_grid_name = inlet_grid_name 
        self.outlet_grid_name = outlet_grid_name

        # Post processing parameters
        self.pct_window = pct_window # check last n% of simulation
        self.pp_tolerance = pp_tolerance # be within n% of inlet value to start post processing
        self.cylinder_grids = cylinder_grids # number of points to extract from cylinder
        self.output_dir = output_dir
        self.average_window = average_window # average for removed particles
        self.plot_freq = plot_freq

        make_directory(self.output_dir)
        

    def execute_case(self):
        ############################
        # loop over TPMC model
        ############################

        # Mesh Info
        wall_grid = read_stl(self.wall_grid_name)
        inlet_grid = read_stl(self.inlet_grid_name)
        outlet_grid = read_stl(self.outlet_grid_name)
        surf_normal = np.array([1, 0, 0])
        no_wall_elems = np.shape(wall_grid.centroids)[0]
        # grid info
        inlet_a = 0
        for c in inlet_grid.areas:
            inlet_a = inlet_a + c

        # number flux
        sigma_t = np.pi/4*self.molecule_d**2 # collision crossection
        number_density = 1/self.kn/sigma_t/self.tube_d
        c_m = np.sqrt(2*KB*self.freestream_temp/self.m)
        v_bar = np.dot(self.freestream_vel, surf_normal)
        s_n = (np.dot(v_bar, surf_normal)/c_m)[0]# A.26
        f_n = number_density*c_m/2/np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)*s_n*(1 + special.erf(s_n))) # A.27
        # particle inflow
        real_particles_per_timestep = np.ceil(f_n*self.dt*inlet_a) # A.28
        wp = real_particles_per_timestep/self.particles_per_timestep # weighting factor

        # output grid info
        n_0 = inlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
        n_l = outlet_grid.points[0][0] # TODO does not generatlze to non-x normal surfaces
        dx = (n_l-n_0)/self.cylinder_grids
        output_grids = np.vstack([np.linspace(n_0 + dx/2, n_l - dx/2, self.cylinder_grids), np.zeros(self.cylinder_grids), np.zeros(self.cylinder_grids)])
        # create output class
        post_proc = POST_PROCESS(output_grids, wall_grid, wp, self.dt)

        # initalize variables
        particles = np.array(['rx1','ry1','rz1','rx2','ry2','rz2','vx','vy','vz','bvx','bvy','bvz','m','mol_diam',], dtype=object)
        removed_particles_time = [[],[]] # 2d list for plotting removed particles
        removed_particles_inlet = []
        removed_particles_outlet = []

        # particle variable indices
        posn_1 = slice(0, 3)
        posn_2 = slice(3, 6)
        vel = slice(6, 9)
        bulk_vel = slice(9, 12)
        m = 12
        mol_diam = 13

        i = 1 # timestep index
        removed = 0 # initalize number of removed particle objects
        removed_outlet = 0 
        removed_inlet = 0 
        start_post = False # start by not post processing results until convergence criteria reached
        while i < self.t_steps:
            # generate particles for each timestep
            for n in np.arange(0,self.particles_per_timestep):
                v = gen_velocity(self.freestream_vel, c_m, s_n) # TODO formulate for general inlet plane orientation
                r = gen_posn(inlet_grid)
                # create list of inputs for particle
                new_particle = self.create_particle(r, v)
                particles = np.vstack([particles,new_particle])


            p = 1 # start at 1 since first row is lables
            removed = 0
            removed_outlet = 0
            removed_inlet = 0
            pres = [[] for x in np.arange(0,no_wall_elems)] # pressure matrix for current timestep
            ener = [0]*no_wall_elems # thermal energy matrix
            axial_stress = [[] for x in np.arange(0,no_wall_elems)] # pressure matrix for current timestep
            while p < np.shape(particles)[0]:
                # print(f'Checking Particle {p}')
                dx = particles[p][vel] * self.dt
                particles[p][posn_1] = particles[p][posn_1] + dx

                # detect wall collisions by looping over cells
                for c in np.arange(np.shape(wall_grid.centroids)[0]):
                        # create element basis centered on centroid
                        cell_n = wall_grid.normals[c]
                        # transform positions to new basis
                        cent = wall_grid.centroids[c]
                        cell_n_i = cell_n.dot(cent - particles[p][posn_2])
                        cell_n_f = cell_n.dot(cent - particles[p][posn_1])
                        if np.sign(cell_n_f) != np.sign(cell_n_i):
                            cell_n_mag = np.linalg.norm(wall_grid.normals[c]) # saves time by moving this here
                            cell_n_i = cell_n_i/cell_n_mag
                            cell_n_f = cell_n_f/cell_n_mag

                            pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
                            intersect = particles[p][posn_2] + pct_vect*self.dt*particles[p][vel]

                            if in_element(wall_grid.points[c], cell_n, intersect):
                                if np.random.rand(1) > self.alpha:
                                    dm, particle_reflected = self.reflect_specular( particles[p], posn_1, posn_2, vel, m, cell_n, cell_n_i, cell_n_f)
                                particles[p] = particle_reflected
                                # TODO fix specular reflection function
                                # else:
                                #     dm, de = particle[p].reflect_diffuse(cell_n, self.dt, cell_n_i, cell_n_f, self.t_tw, c_m)
                                #     # energy change
                                #     ener[c] = ener[c] + de*self.m/self.dt/2*wp # convert to Joules
                                # pressure contribution from reflection
                                pres_scalar = np.linalg.norm(dm[1:3]/self.dt/wall_grid.areas[c]) # not a very clevery way to get normal compoent
                                pres[c].append(pres_scalar) 
                                # axial pressure contribution from reflection
                                axial_stress_scalar = np.linalg.norm(dm[0]/self.dt/wall_grid.areas[c])
                                axial_stress[c].append(axial_stress_scalar)
                
                if self.exit_domain_outlet(particles[p], posn_1, n_l):
                        particles = np.delete(particles, p, 0)
                        removed_outlet+=1
                        removed+=1
                        p-=1
                if self.exit_domain_inlet(particles[p], posn_1, n_0):
                        particles = np.delete(particles, p, 0)
                        removed_inlet+=1
                        removed+=1
                        p-=1

                p += 1
                        
            # find now many particles leave the domain per timestep
            removed_particles_time[0].append(i*self.dt)
            removed_particles_time[1].append(removed)
            removed_particles_inlet.append(removed_inlet)
            removed_particles_outlet.append(removed_outlet)
            # plot removed particles with time
            post_proc.plot_removed_particles(self.output_dir, self.t_tw, self.alpha, removed_particles_time, removed_particles_outlet, removed_particles_inlet, self.average_window) # dont hardcode this value
            
            # print status to terminal
            print(f"--------------------------------------------------------------------------")
            print(f'Particles removed: {removed}')
            print(f"Total Time: {i*self.dt}")
            print(f"Time Steps: {100*(i)/self.t_steps} %")

            # detect if steady state is reached and if post processing should start
            if not start_post:
                if start_postproc(self.pct_window, self.pp_tolerance, removed_particles_time, self.particles_per_timestep, i):
                        start_post = True # just turn this flag on once
            # TODO start this back up later, will require postprocess being changed a lot
            # else:
            #     # update outputs
            #     post_proc.update_outputs(particles, pres, np.array(ener), axial_stress) # this is totally broken
            #     print(f"Post Processing...")

            #     # create plots
            #     if i%self.plot_freq == 0:
            #             post_proc.plot_n_density(self.output_dir, self.t_tw, self.alpha)
            #             post_proc.plot_temp(self.output_dir, self.t_tw, self.alpha)
            #             post_proc.plot_pressure(self.output_dir, self.t_tw, self.alpha)
            #             post_proc.plot_heat_tfr(self.output_dir, self.t_tw, self.alpha)
            #             post_proc.plot_n_coll(self.output_dir, self.t_tw, self.alpha)
            #             post_proc.plot_shear(self.output_dir, self.t_tw, self.alpha)

            i+=1 # add to timestep index, continue to next timestep


    def readme_output(self):
        """write out readme to output directory with info about the model
        """
        a = 1

    def create_particle(self, r, v):
        """append particle to current list

        Args:
            case (_type_): _description_
        """
        # TODO this is gross
        particle_data = np.append(r,[0,0,0])
        particle_data = np.append(particle_data, v)
        particle_data = np.append(particle_data, self.freestream_vel)
        particle_data = np.append(particle_data, self.m)
        particle_data = np.append(particle_data, self.molecule_d)

        return particle_data


    def reflect_specular(self, particle, posn_1, posn_2, vel, m, wall: np.array, cell_n_i, cell_n_f):
        """calculate the reflected velocity for a specular wall impact
        Args:
            c (np.array): incomming velocity
            wall (np.array): wall normal vector, inwards facing, # TODO I think this needs to be inward facing...
            dt (float): timestep length
            tube_d (float): diameter of tube
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        # intersect = self.posn_hist[-2] + pct_vect*dt*self.vel
        intersect = particle[posn_2] + pct_vect*self.dt*particle[vel]


        # ensure wall vector is a unit vector
        wall = wall/np.linalg.norm(wall)
        v0 = particle[vel]
        c_n = np.dot(particle[vel],wall)*wall # normal component to wall
        c_p = particle[vel] - c_n # perpendicular component to wall
        particle[vel] = c_p - c_n # flip normal component
        dm = particle[m]*(particle[vel] - v0) # change in momentum from wall collission

        # create copy of vector
        # collision location
        particle[posn_2] = intersect
        # post - collision location
        particle[posn_1] = intersect + self.dt*(1 - pct_vect)*particle[vel] # update position with fraction of remaining timestep

        return dm, particle


    def exit_domain_inlet(self, particle, posn_1, inlet_plane: float):
        """determine if particle has left domain
        Args:
            exit_plane (float): Z location of exit plane
        Returns:
            BOOL: if partice is within domain 
        """
        # TODO add in point and vector definition of plane to use dot product
        if particle[posn_1][0] < inlet_plane:
            return True

    def exit_domain_outlet(self, particle, posn_1, exit_plane: float):
        """determine if particle has left domain
        Args:
            exit_plane (float): Z location of exit plane
        Returns:
            BOOL: if partice is within domain 
        """
        # TODO add in point and vector definition of plane to use dot product
        if particle[posn_1][0] >= exit_plane:
            return True
    