
from random import random
from turtle import pos, st
import numpy as np
from particle import PARTICLE
from postprocess import POST_PROCESS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from utilities import KB, read_stl, in_element, start_postproc, gen_posn, gen_velocity, AVOS_NUM, make_directory, plot_walls
from scipy.spatial.transform import Rotation as R
from scipy import special
import sys
import os
import time
import pickle

# General things to add
# TODO add pint
# TODO non-even timesteps? automatic timestep generation?


class CASE_TPMC:
    """ TPMC case class. Includes all options needed to run TPMC model.
    """

    def __init__(self, case_name: str, dt: float, t_steps: int, particles_per_timestep: int, freestream_vel: np.array,
                 alpha: float, t_tw: float, tube_d: float, freestream_temp: float, kn: float, m: float, molecule_d:
                 float, wall_grid_name: str, inlet_grid_name: str, outlet_grid_name: str, pct_window: float,
                 pp_tolerance: float, cylinder_grids: int, output_dir: float, average_window: int, plot_freq: int,
                 num_particles: int, coll_cell_width: float, geometry_dir: str):

        self.case_name = case_name
        # specular neutral wall
        self.dt = dt
        self.t_steps = t_steps
        # choose a weighting factor such that only n particles are simulated per timestep
        self.particles_per_timestep = particles_per_timestep
        # freestream conditions
        self.freestream_vel = freestream_vel  # m/s, x velocity
        self.alpha = alpha  # accomidation coeff
        self.t_tw = t_tw  # wall temp ratio

        # geom
        # TODO replace this with just a reference length, I think thats all its used for
        self.tube_d = tube_d
        self.coll_cell_width = coll_cell_width

        # preallocate particle array
        self.num_particles = num_particles

        # freestream conditions
        self.freestream_temp = freestream_temp  # k
        self.kn = kn
        self.mass = m  # mass of a N2 molecule [kg]
        self.molecule_d = molecule_d  # [m]

        # grids names ( must be continuous when assemebled)
        self.geometry_dir = geometry_dir
        self.wall_grid_name = wall_grid_name
        self.inlet_grid_name = inlet_grid_name
        self.outlet_grid_name = outlet_grid_name

        # Post processing parameters
        self.pct_window = pct_window  # check last n% of simulation
        # be within n% of inlet value to start post processing
        self.pp_tolerance = pp_tolerance
        # number of points to extract from cylinder
        self.cylinder_grids = cylinder_grids
        self.output_dir = output_dir
        self.average_window = average_window  # average for removed particles
        self.plot_freq = plot_freq

        make_directory(self.output_dir)

        # process grid and create connectivity from collision cells to wall emenets
        self.process_grid()

        # initalize some variables
        self.removed = 0  # initalize number of removed particle objects
        self.removed_outlet = 0
        self.removed_inlet = 0

    def execute_case(self):
        ############################
        # loop over DSMC model
        ############################
        
        # number flux
        sigma_t = np.pi/4*self.molecule_d**2  # collision crossection
        number_density = 1/self.kn/sigma_t/self.tube_d
        self.c_m = np.sqrt(2*KB*self.freestream_temp/self.mass)
        v_bar = np.dot(self.freestream_vel, self.surf_normal)
        s_n = (np.dot(v_bar, self.surf_normal)/self.c_m)[0]  # A.26
        f_n = number_density*self.c_m/2 / \
            np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)
                            * s_n*(1 + special.erf(s_n)))  # A.27
        # particle inflow
        real_particles_per_timestep = np.ceil(f_n*self.dt*self.inlet_a)  # A.28
        wp = real_particles_per_timestep/self.particles_per_timestep  # weighting factor

        # create output class
        post_proc = POST_PROCESS(
            self.output_grids, self.wall_grid, wp, self.dt, self.output_dir)

        self.generate_particle_array(self.num_particles)

        # 2d list for plotting removed particles
        removed_particles_time = [[], []]
        removed_particles_inlet = []
        removed_particles_outlet = []

        i = 1  # timestep index
        # start by not post processing results until convergence criteria reached
        start_post = False
        while i < self.t_steps:
            timestep_time = time.perf_counter()

            # zero entries where a particles can be inserted
            empty_entries = np.nonzero(self.particles[1:-1, -1] - 1)[0]
            # generate particles for each timestep
            # this should die if there are not enough open spots for partices
            particle_gen_time = time.perf_counter()
            for n in np.arange(0, self.particles_per_timestep):
                # TODO formulate for general inlet plane orientation
                v = gen_velocity(self.freestream_vel, self.c_m, s_n)
                r = gen_posn(self.inlet_grid)
                # create list of inputs for particle
                new_particle = self.create_particle(r, v)
                index = empty_entries[n] + 1
                # add particle to particle array
                self.particles[index, :] = new_particle
            print(
                f"Particle Generation Time: {time.perf_counter() - particle_gen_time}")

            self.removed = 0
            self.removed_outlet = 0
            self.removed_inlet = 0
            # pressure matrix for current timestep
            self.pres = [[] for x in np.arange(0, self.no_wall_elems)]
            self.ener = [0]*self.no_wall_elems  # thermal energy matrix
            # pressure matrix for current timestep
            self.axial_stress = [[] for x in np.arange(0, self.no_wall_elems)]

            # print(f'Checking Particle {p}')
            # propogating particles
            self.particles[1:, self.posn_2] = self.particles[1:, self.posn_1]
            self.particles[1:, self.posn_1] = self.particles[1:,
                                                             self.vel]*self.dt + self.particles[1:, self.posn_2]

            # find particle entries that exist
            self.extant_particles = np.nonzero(self.particles[1:, self.has_particle])[0]
            self.particle_index = np.add(self.extant_particles, 1)

            # update collision cell indices for each particle
            self.update_collision_cell()

            # loop over particles to find wall collisions
            # this should be a subset of particles that are near the wall
            collision_detect_time = time.perf_counter()
            self.wall_collisions()
            print(
                f"Collision Detect Time: {time.perf_counter() - collision_detect_time}")

            # find now many particles leave the domain per timestep
            removed_particles_time[0].append(i*self.dt)
            removed_particles_time[1].append(self.removed)
            removed_particles_inlet.append(self.removed_inlet)
            removed_particles_outlet.append(self.removed_outlet)
            # plot removed particles with time
            post_proc.plot_removed_particles(self.t_tw, self.alpha, removed_particles_time, removed_particles_outlet,
                                             removed_particles_inlet, self.average_window)  # dont hardcode this value

            # print status to terminal
            self.print_status(i, self.extant_particles)

            # detect if steady state is reached and if post processing should start
            if not start_post:
                if start_postproc(self.pct_window, self.pp_tolerance, removed_particles_time, self.particles_per_timestep, i):
                    start_post = True  # just turn this flag on once
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

            print(f"Timestep Time: {time.perf_counter() - timestep_time}")
            i += 1  # add to timestep index, continue to next timestep

    def print_status(self, i, extant_particles):
        """display timestep status to terminal
        """
        print(
        f"--------------------------------------------------------------------------")
        print(f'Particles removed: {self.removed}')
        print(f"Total Time: {i*self.dt}")
        print(f"Time Steps: {100*(i)/self.t_steps} %")
        # TODO fix this since it doesnt work with the preallocated version
        print(f"Particles in Domain: {extant_particles.__len__()}")


    def generate_particle_array(self, dim: int):
        """_summary_
        """
        # initalize variables
        col_names = np.array([['rx1', 'ry1', 'rz1', 'rx2', 'ry2', 'rz2', 'vx', 'vy',
                             'vz', 'bvx', 'bvy', 'bvz', 'm', 'mol_diam', 'has_particle','cell_id']], dtype=object)
        self.particles = np.concatenate((col_names, np.zeros(
            [dim, np.size(col_names)])), axis=0)
        # r1 is the current position, r2 is the old position

        # particle variable indices
        self.posn_1 = slice(0, 3) # current position
        self.posn_2 = slice(3, 6)
        self.vel = slice(6, 9)
        self.bulk_vel = slice(9, 12)
        self.m = 12
        self.mol_diam = 13
        self.has_particle = 14
        self.cell_id = 15

    def test_visualization(self, vis_particles):
        """_summary_

        Args:
            num_particles (_type_): _description_
        """

        self.generate_particle_array(vis_particles)

        self.pres = [[] for x in np.arange(0, self.no_wall_elems)]
        self.ener = [0]*self.no_wall_elems  # thermal energy matrix
        # pressure matrix for current timestep
        self.axial_stress = [[] for x in np.arange(0, self.no_wall_elems)]
        self.c_m = 2 # temp value

        for n in np.arange(1, vis_particles+1):
            # TODO formulate for general inlet plane orientation
            v = gen_velocity(100, self.c_m, 0) # TODO find fake values of c_m and s_n
            # v = np.array([200,20,0])
            r = gen_posn(self.inlet_grid)
            # r = np.array([0,0,0]) 
            # create list of inputs for particle
            self.particles[n, :] = self.create_particle(r, v)

        i = 1  # timestep index
        posn_hist = [np.empty((1,3))]*vis_particles
        while i < self.t_steps:

            # propogating particles TODO make this a function later?
            self.particles[1:, self.posn_2] = self.particles[1:, self.posn_1]
            self.particles[1:, self.posn_1] = self.particles[1:,
                                                             self.vel]*self.dt + self.particles[1:, self.posn_2]

            # print(f"Posn: {self.particles[1][self.posn_1]}")

            # find particle entries that exist
            self.extant_particles = np.nonzero(self.particles[1:, self.has_particle])[0]
            self.particle_index = np.add(self.extant_particles, 1)

            # update collision cell indices for each particle
            self.update_collision_cell()

            # loop over particles to find wall collisions
            self.wall_collisions()

            # save particle position. this missed the wall collision location if a wall collision happens
            pid = 0
            for p in self.particle_index:
                posn_hist[p-1] = np.vstack([posn_hist[p-1], self.particles[p][self.posn_2], self.particles[p][self.posn_1]])
                pid+=1

                if np.abs(self.particles[p][self.posn_2][1]) > 0.005 and np.abs(self.particles[p][self.posn_2][2]) > 0.005:
                    a = 1

            # end simulation when all particles have left
            if self.removed == vis_particles:
                break 

            # print status to terminal
            self.print_status(i, self.extant_particles)

            i+=1


        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for p in np.arange(0, vis_particles):
            ax.plot3D(posn_hist[p][:,0].astype(float), posn_hist[p][:,1].astype(float), posn_hist[p][:,2].astype(float), marker=".")
        plot_walls(ax, self.wall_grid)
        plt.xlabel('X')
        plt.ylabel('Y')
        fig.savefig('temp.png')
        plt.show()


    def update_collision_cell(self):
        """determine what collision cell each particle is in"""
        y_dim = self.coll_cells[1].__len__()
        z_dim = self.coll_cells[2].__len__()

        for p in self.particle_index:
            ind = np.zeros([1,3])[0]
            for i in [0,1,2]:
                ind[i] = np.abs((self.particles[p][self.posn_1][i]  - self.coll_cells[i])).argmin()
            self.particles[p][self.cell_id] = ind[0]*y_dim*z_dim + ind[1]*z_dim + ind[2] # pattern to determine what cell the particle is located in


    def process_grid(self):
        """generate grid information and collision cell to wall cell connectivity
        """

        # Mesh Info
        self.wall_grid = read_stl(os.path.join(self.geometry_dir,self.wall_grid_name))
        self.inlet_grid = read_stl(os.path.join(self.geometry_dir,self.inlet_grid_name))
        self.outlet_grid = read_stl(os.path.join(self.geometry_dir,self.outlet_grid_name))
        self.surf_normal = np.array([1, 0, 0]) # TODO un-jankify this, should not be hard coded
        self.no_wall_elems = np.shape(self.wall_grid.centroids)[0]
        # grid info
        self.inlet_a = 0 # inlet area
        for c in self.inlet_grid.areas:
            self.inlet_a = self.inlet_a + c

        # output grid info
        # TODO does not generatlze to non-x normal surfaces
        self.n_0 = self.inlet_grid.points[0][0]
        # TODO does not generatlze to non-x normal surfaces
        self.n_l = self.outlet_grid.points[0][0]
        dx = (self.n_l-self.n_0)/self.cylinder_grids
        self.output_grids = np.vstack([np.linspace(self.n_0 + dx/2, self.n_l - dx/2, self.cylinder_grids),
                                 np.zeros(self.cylinder_grids), np.zeros(self.cylinder_grids)])
        

        # create collision cells
        self.coll_cells = [[], [], []]  # list of grids in each direction
        for i in [0, 1, 2]:
            # length of coll cell domain in each dir
            dir_len = self.wall_grid.max_[i] - self.wall_grid.min_[i]
            # number of divisions in each direction
            n_divs = np.floor(dir_len/self.coll_cell_width).astype(int)
            self.coll_cells[i] = np.linspace(self.wall_grid.min_[
                                        i] + self.coll_cell_width/2, self.wall_grid.max_[i] - self.coll_cell_width/2, n_divs)
        num_coll_cells = self.coll_cells[0].__len__(
        )*self.coll_cells[1].__len__()*self.coll_cells[2].__len__()



        self.coll_cell_centroid = np.zeros([num_coll_cells, 3])
        c = 0
        for x in np.arange(0, self.coll_cells[0].__len__()):
            for y in np.arange(0, self.coll_cells[1].__len__()):
                for z in np.arange(0, self.coll_cells[2].__len__()):
                    self.coll_cell_centroid[c, :] = np.array(
                        [self.coll_cells[0][x], self.coll_cells[1][y], self.coll_cells[2][z]])
                    c += 1

        # check if this grid has already been processed
        pkl_check = os.path.join(self.geometry_dir,self.wall_grid_name[:-4]+f"_{self.coll_cell_width}"+'.wc_cc_info')
        if os.path.exists(pkl_check ):
            f = open(pkl_check, 'rb')
            self.cc_intersctions = pickle.load(f)
            return

        # find what cc intersect element planes
        self.cc_intersctions = [[]]*num_coll_cells
        corners = np.zeros([8,3])
        cid = 0
        for cc in self.coll_cell_centroid:
            elem_list = [] # list to store element ids
            dx = self.coll_cell_width
            # corners of box
            corners[0, :] = np.array(
                [cc[0] + dx/2, cc[1] + dx/2, cc[2] + dx/2])
            corners[1, :] = np.array(
                [cc[0] + dx/2, cc[1] + dx/2, cc[2] - dx/2])
            corners[2, :] = np.array(
                [cc[0] + dx/2, cc[1] - dx/2, cc[2] + dx/2])
            corners[3, :] = np.array(
                [cc[0] + dx/2, cc[1] - dx/2, cc[2] - dx/2])
            corners[4, :] = np.array(
                [cc[0] - dx/2, cc[1] + dx/2, cc[2] + dx/2])
            corners[5, :] = np.array(
                [cc[0] - dx/2, cc[1] + dx/2, cc[2] - dx/2])
            corners[6, :] = np.array(
                [cc[0] - dx/2, cc[1] - dx/2, cc[2] + dx/2])
            corners[7, :] = np.array(
                [cc[0] - dx/2, cc[1] - dx/2, cc[2] - dx/2])
            for wc in np.arange(np.shape(self.wall_grid.centroids)[0]):
                normal_comp = np.zeros([1,8])[0]
                g = 0 # corner index
                for i in corners: # check this loops over the xyz points
                    cell_n = self.wall_grid.normals[wc]
                    # transform positions to new basis
                    cent = self.wall_grid.centroids[wc]
                    normal_comp[g] = cell_n.dot(cent - i) # normal component to wall cell
                    g+=1
                # check if cell intersects plane of cell
                if not (normal_comp == 1).all() or (normal_comp == -1).all() or (normal_comp == 0).any():
                    pts = self.wall_grid.points[wc].reshape(3, 3)
                    edge_len = np.max([np.linalg.norm(
                        pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[0] - pts[2])]) # max edge length of cell
                    for i in corners:
                        if edge_len > np.linalg.norm(i - self.wall_grid.centroids[wc]):
                            elem_list.append(wc)
                            break
            self.cc_intersctions[cid] = elem_list
            cid+=1

        # pickle this file since it takes forever
        with open(pkl_check, 'wb') as f:
            pickle.dump(self.cc_intersctions, f)

    def wall_collisions(self):

        for p in self.particle_index:
            dt_fract = self.dt
            is_collided = True
            while is_collided: # keep colliding with wall while particle has time left in the current timestep
                is_collided = False
                # detect wall collisions by looping over wall elements associated with collision cell
                for c in self.cc_intersctions[self.particles[p][self.cell_id].astype(int)]:
                    # create element basis centered on centroid
                    cell_n = self.wall_grid.normals[c]
                    # transform positions to new basis
                    cent = self.wall_grid.centroids[c]
                    cell_n_i = cell_n.dot(cent - self.particles[p][self.posn_2])
                    cell_n_f = cell_n.dot(cent - self.particles[p][self.posn_1])
                    if np.sign(cell_n_f) != np.sign(cell_n_i) and cell_n_i != 0.0: # try something like this later np.isclose(cell_n_i, 0, 1e-6
                        # saves time by moving this here
                        cell_n_mag = np.linalg.norm(self.wall_grid.normals[c])
                        cell_n_i = cell_n_i/cell_n_mag
                        cell_n_f = cell_n_f/cell_n_mag

                        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
                        intersect = self.particles[p][self.posn_2] + \
                            pct_vect*self.dt*self.particles[p][self.vel]

                        if in_element(self.wall_grid.points[c], cell_n, intersect):
                            # specular reflection
                            if np.random.rand(1) > self.alpha:
                                is_collided = True
                                dm, dt_fract = self.reflect_specular(
                                    p, cell_n, cell_n_i, cell_n_f, dt_fract)
                            # diffuse reflection
                            else:
                                dm, de, dt_fract = self.reflect_diffuse(
                                    p, cell_n, cell_n_i, cell_n_f, dt_fract)
                                # energy change
                                # self.ener[c] = ener[c] + de*self.m/self.dt/2*wp # convert to Joules
                            # pressure contribution from reflection
                            # not a very clevery way to get normal compoent
                            # pres_scalar = np.linalg.norm(
                                # dm[1:3]/self.dt/self.wall_grid.areas[c])
                            # self.pres[c].append(pres_scalar)
                            # axial pressure contribution from reflection
                            # axial_stress_scalar = np.linalg.norm(
                                # dm[0]/self.dt/self.wall_grid.areas[c])
                            # self.axial_stress[c].append(axial_stress_scalar)
                            break # dont loop over other wall elements if a collison happens


            # particle_remove_time = time.perf_counter()
            if self.exit_domain_outlet(p):
                # flip flag to "no particle"
                self.particles[p][self.has_particle] = 0
                self.removed_outlet += 1
                self.removed += 1
                # print(f"Particle Removal: {time.perf_counter() - particle_remove_time}")
            if self.exit_domain_inlet(p):
                # flip flag to "no particle"
                self.particles[p][self.has_particle] = 0
                self.removed_inlet += 1
                self.removed += 1
                # print(f"Particle Removal: {time.perf_counter() - particle_remove_time}")

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
        particle_data = np.append(r, [0, 0, 0])
        particle_data = np.append(particle_data, v)
        particle_data = np.append(particle_data, self.freestream_vel)
        particle_data = np.append(particle_data, self.mass)
        particle_data = np.append(particle_data, self.molecule_d)
        # set flag to 1 to show particle does exist
        particle_data = np.append(particle_data, 1)
        particle_data = np.append(particle_data, np.nan)

        return particle_data

    def reflect_specular(self, p, wall_n: np.array, cell_n_i, cell_n_f, dt):
        """calculate the reflected velocity for a specular wall impact
        Args:
            p (np.array): particle index
            wall (np.array): wall normal vector, inwards facing, # TODO I think this needs to be inward facing...
            dt (float): timestep length
            tube_d (float): diameter of tube
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        # intersection of wall with wall element
        intersect = self.particles[p][self.posn_2] + \
            pct_vect*dt*self.particles[p][self.vel]

        # ensure wall vector is a unit vector
        wall_n = wall_n/np.linalg.norm(wall_n)
        v0 = self.particles[p][self.vel].astype(float)
        # normal component to wall
        c_n = np.dot(self.particles[p][self.vel], wall_n)*wall_n
        # perpendicular component to wall
        c_p = self.particles[p][self.vel] - c_n
        self.particles[p][self.vel] = c_p - c_n  # flip normal component
        # change in momentum from wall collission
        dm = self.particles[p][self.m]*(self.particles[p][self.vel] - v0)

        # create copy of vector
        # collision location
        self.particles[p][self.posn_2] = intersect
        dt = dt*(1 - pct_vect)
        # post - collision location
        # update position with fraction of remaining timestep
        self.particles[p][self.posn_1] = intersect + \
            dt*self.particles[p][self.vel]

        return dm, dt

    def reflect_diffuse(self, p, wall_n: np.array, cell_n_i, cell_n_f, dt: float):
        """_summary_
        Args:
            wall_n (np.array): _description_
            dt (float): _description_
            tube_d (_type_): _description_
            cell_n_i (_type_): _description_
            cell_n_f (_type_): _description_
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        intersect = self.particles[p][self.posn_2] + \
            pct_vect*dt*self.particles[p][self.vel]
        
        # ensure wall vector is a unit vector
        wall_n = wall_n/np.linalg.norm(wall_n)
        # random vector for generating normals
        gen_vect = True
        while gen_vect:
            random_vect = np.random.rand(3)
            random_vect = random_vect/np.linalg.norm(random_vect)
            wall_t = np.cross(wall_n, random_vect) # find tangential vector to surface
            if (wall_t == np.array([0,0,0])).all(): # make sure the vector is not zero
                continue
            else:
                gen_vect = False

        wall_b = np.cross(wall_n, wall_t)/np.linalg.norm(np.cross(wall_n, wall_t)) # find a third basis vector for the cell csys

        # find cell csys velocity
        v0 = self.particles[p][self.vel].astype(float)
        zero_bulk = np.array([0 ,0, 0])
        cell_vel = gen_velocity(zero_bulk , np.sqrt(self.t_tw)*self.c_m, 0) # c_m scaled and s_n = 0. Velcity in cell csys
        self.particles[p][self.vel] = cell_vel[0]*wall_n + cell_vel[1]*wall_t + cell_vel[2]*wall_b # normal velocity with X in the cell_normal direction
        dm = self.particles[p][self.m]*(self.particles[p][self.vel] - v0) # change in momentum from wall collission
        de = 0.5*self.particles[p][self.m]*np.linalg.norm(self.particles[p][self.vel])**2 - np.linalg.norm(v0)**2 # only needed for diffuse, this is zero for specular

        # wall_intersection
        self.particles[p][self.posn_2] = intersect
        dt = dt*(1 - pct_vect)
        # new position
        self.particles[p][self.posn_1] = intersect + \
            dt*self.particles[p][self.vel]

        return dm, de, dt

    def exit_domain_inlet(self, p: int):
        """ Checks if particle has left through inlet

        Returns:
            Bool: has the particle left?
        """
        # TODO add in point and vector definition of plane to use dot product
        if self.particles[p][self.posn_1][0] < self.n_0:
            return True

    def exit_domain_outlet(self, p: int):
        """Checks if particle has left through outlet

        Returns:
            Bool: has the particle left?
        """
        # TODO add in point and vector definition of plane to use dot product
        if self.particles[p][self.posn_1][0] >= self.n_l:
            return True
