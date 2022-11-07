
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
import time

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
                 num_particles: int, coll_cell_width: float):

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
        self.m = m  # mass of a N2 molecule [kg]
        self.molecule_d = molecule_d  # [m]

        # grids names ( must be continuous when assemebled)
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

    def execute_case(self):
        ############################
        # loop over TPMC model
        ############################

        # Mesh Info
        self.wall_grid = read_stl(self.wall_grid_name)
        self.inlet_grid = read_stl(self.inlet_grid_name)
        self.outlet_grid = read_stl(self.outlet_grid_name)
        surf_normal = np.array([1, 0, 0])
        no_wall_elems = np.shape(self.wall_grid.centroids)[0]
        # grid info
        inlet_a = 0
        for c in self.inlet_grid.areas:
            inlet_a = inlet_a + c

        # create collision cells
        coll_cells = [[], [], []]  # list of grids in each direction
        for i in [0, 1, 2]:
            # length of coll cell domain in each dir
            dir_len = self.wall_grid.max_[i] - self.wall_grid.min_[i]
            # number of divisions in each direction
            n_divs = np.floor(dir_len/self.coll_cell_width).astype(int)
            coll_cells[i] = np.linspace(self.wall_grid.min_[
                                        i] + self.coll_cell_width/2, self.wall_grid.max_[i] - self.coll_cell_width/2, n_divs)
        num_coll_cells = coll_cells[0].__len__(
        )*coll_cells[1].__len__()*coll_cells[2].__len__()

        coll_cell_ids = np.zeros([num_coll_cells, 3])
        c = 0
        for x in np.arange(0, coll_cells[0].__len__()):
            for y in np.arange(0, coll_cells[1].__len__()):
                for z in np.arange(0, coll_cells[2].__len__()):
                    coll_cell_ids[c, :] = np.array(
                        [coll_cells[0][x], coll_cells[1][y], coll_cells[2][z]])
                    c += 1

        # now find what collision cells are close to the wall and what elements they are close to
        # sphere inscribing cell
        coll_cell_prox = np.sqrt(3)*self.coll_cell_width
        # find max edge length of a wall cell
        max_edge_len = 0
        for c in self.wall_grid.points:
            # reshape into a 3x3 WILL BREAK IF THERE ARE SQUARES IN THE STL SOMEHOW
            pts = c.reshape(3, 3)
            edge_len = np.max([np.linalg.norm(
                pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[0] - pts[2])])
            if edge_len > max_edge_len:
                max_edge_len = edge_len

        # estimate of when a cell and wall element might be intersecting
        interaction_tolerance = max_edge_len + coll_cell_prox

        # TODO this is a terrible way to do it. check if all the signs of the cc points save the same sign in the element normal direction
        # find collision cells that are near the wall
        intersect_wall_cells = []
        cid = 0  # start at cell id 0
        for cc in coll_cell_ids:
            break_flag = False
            wid = 0  # wall cell ID
            for wc in self.wall_grid.centroids:
                if break_flag:
                    break
                # janky expression for finding cells near wall
                if np.linalg.norm(cc - wc) < interaction_tolerance:
                    intersect_wall_cells.append(cid)
                    # cid+=1
                    break_flag = True
                    break
                else:
                    break_flag = False

                wid += 1
            cid += 1

        # find wall elements near each collision cell
        # this will need to be higher dimentional later
        elems_near_cell = [[]]*num_coll_cells
        cid = 0
        for cc in coll_cell_ids[intersect_wall_cells, :]:
            wid = 0
            for wc in self.wall_grid.centroids:
                if np.linalg.norm(cc - wc) < interaction_tolerance:
                    if bool(elems_near_cell[cid]):
                        elems_near_cell[cid].append(wid)
                    else:
                        elems_near_cell[cid] = [wid]
                wid += 1
            cid += 1

        # number flux
        sigma_t = np.pi/4*self.molecule_d**2  # collision crossection
        number_density = 1/self.kn/sigma_t/self.tube_d
        c_m = np.sqrt(2*KB*self.freestream_temp/self.m)
        v_bar = np.dot(self.freestream_vel, surf_normal)
        s_n = (np.dot(v_bar, surf_normal)/c_m)[0]  # A.26
        f_n = number_density*c_m/2 / \
            np.sqrt(np.pi)*(np.exp(-s_n**2) + np.sqrt(np.pi)
                            * s_n*(1 + special.erf(s_n)))  # A.27
        # particle inflow
        real_particles_per_timestep = np.ceil(f_n*self.dt*inlet_a)  # A.28
        wp = real_particles_per_timestep/self.particles_per_timestep  # weighting factor

        # output grid info
        # TODO does not generatlze to non-x normal surfaces
        self.n_0 = self.inlet_grid.points[0][0]
        # TODO does not generatlze to non-x normal surfaces
        self.n_l = self.outlet_grid.points[0][0]
        dx = (self.n_l-self.n_0)/self.cylinder_grids
        output_grids = np.vstack([np.linspace(self.n_0 + dx/2, self.n_l - dx/2, self.cylinder_grids),
                                 np.zeros(self.cylinder_grids), np.zeros(self.cylinder_grids)])
        # create output class
        post_proc = POST_PROCESS(
            output_grids, self.wall_grid, wp, self.dt, self.output_dir)

        # initalize variables
        col_names = np.array([['rx1', 'ry1', 'rz1', 'rx2', 'ry2', 'rz2', 'vx', 'vy',
                             'vz', 'bvx', 'bvy', 'bvz', 'm', 'mol_diam', 'has_particle']], dtype=object)
        self.particles = np.concatenate((col_names, np.zeros(
            [self.num_particles, np.size(col_names)])), axis=0)
        # r1 is the current position, r2 is the old position
        # 2d list for plotting removed particles
        removed_particles_time = [[], []]
        removed_particles_inlet = []
        removed_particles_outlet = []

        # particle variable indices
        self.posn_1 = slice(0, 3)
        self.posn_2 = slice(3, 6)
        self.vel = slice(6, 9)
        self.bulk_vel = slice(9, 12)
        self.m = 12
        self.mol_diam = 13
        self.has_particle = 14

        i = 1  # timestep index
        self.removed = 0  # initalize number of removed particle objects
        self.removed_outlet = 0
        self.removed_inlet = 0
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
                v = gen_velocity(self.freestream_vel, c_m, s_n)
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
            self.pres = [[] for x in np.arange(0, no_wall_elems)]
            self.ener = [0]*no_wall_elems  # thermal energy matrix
            # pressure matrix for current timestep
            self.axial_stress = [[] for x in np.arange(0, no_wall_elems)]

            # print(f'Checking Particle {p}')
            # propogating particles
            self.particles[1:, self.posn_2] = self.particles[1:, self.posn_1]
            self.particles[1:, self.posn_1] = self.particles[1:,
                                                             self.vel]*self.dt + self.particles[1:, self.posn_2]

            # find particle entries that exist
            extant_particles = np.nonzero(self.particles[1:-1, -1])[0]
            self.particle_index = np.add(extant_particles, 1)

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
            print(
                f"--------------------------------------------------------------------------")
            print(f'Particles removed: {self.removed}')
            print(f"Total Time: {i*self.dt}")
            print(f"Time Steps: {100*(i)/self.t_steps} %")
            # TODO fix this since it doesnt work with the preallocated version
            print(f"Particles in Domain: {extant_particles.__len__()}")

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

    def wall_collisions(self):

        for p in self.particle_index:

            # detect wall collisions by looping over cells
            for c in np.arange(np.shape(self.wall_grid.centroids)[0]):
                # create element basis centered on centroid
                cell_n = self.wall_grid.normals[c]
                # transform positions to new basis
                cent = self.wall_grid.centroids[c]
                cell_n_i = cell_n.dot(cent - self.particles[p][self.posn_2])
                cell_n_f = cell_n.dot(cent - self.particles[p][self.posn_1])
                if np.sign(cell_n_f) != np.sign(cell_n_i):
                    # saves time by moving this here
                    cell_n_mag = np.linalg.norm(self.wall_grid.normals[c])
                    cell_n_i = cell_n_i/cell_n_mag
                    cell_n_f = cell_n_f/cell_n_mag

                    pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
                    intersect = self.particles[p][self.posn_2] + \
                        pct_vect*self.dt*self.particles[p][self.vel]

                    # TODO precalculate volume cell associativity with wall cells so the cell wall detection only needs to happen a few times.
                    if in_element(self.wall_grid.points[c], cell_n, intersect):
                        if np.random.rand(1) > self.alpha:
                            dm = self.reflect_specular(
                                p, cell_n, cell_n_i, cell_n_f)
                        # TODO fix diffuse reflection function
                        # else:
                        #     dm, de = particle[p].reflect_diffuse(cell_n, self.dt, cell_n_i, cell_n_f, self.t_tw, c_m)
                        #     # energy change
                        #     self.ener[c] = ener[c] + de*self.m/self.dt/2*wp # convert to Joules
                        # pressure contribution from reflection
                        # not a very clevery way to get normal compoent
                        pres_scalar = np.linalg.norm(
                            dm[1:3]/self.dt/self.wall_grid.areas[c])
                        self.pres[c].append(pres_scalar)
                        # axial pressure contribution from reflection
                        axial_stress_scalar = np.linalg.norm(
                            dm[0]/self.dt/self.wall_grid.areas[c])
                        self.axial_stress[c].append(axial_stress_scalar)

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
        particle_data = np.append(particle_data, self.m)
        particle_data = np.append(particle_data, self.molecule_d)
        # set flag to 1 to show particle does exist
        particle_data = np.append(particle_data, 1)

        return particle_data

    def reflect_specular(self, p, wall_n: np.array, cell_n_i, cell_n_f):
        """calculate the reflected velocity for a specular wall impact
        Args:
            p (np.array): particle index
            wall (np.array): wall normal vector, inwards facing, # TODO I think this needs to be inward facing...
            dt (float): timestep length
            tube_d (float): diameter of tube
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        # intersect = self.posn_hist[-2] + pct_vect*dt*self.vel
        intersect = self.particles[p][self.posn_2] + \
            pct_vect*self.dt*self.particles[p][self.vel]

        # ensure wall vector is a unit vector
        wall_n = wall_n/np.linalg.norm(wall_n)
        v0 = self.particles[p][self.vel]
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
        # post - collision location
        # update position with fraction of remaining timestep
        self.particles[p][self.posn_1] = intersect + \
            self.dt*(1 - pct_vect)*self.particles[p][self.vel]

        return dm

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
