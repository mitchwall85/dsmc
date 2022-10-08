import numpy as np
from utilities import gen_posn, gen_velocity

class PARTICLE:
    """particle model for TPMC & DSMC code
    """
    def __init__(self, mass: float, r: float, init_posn: np.array, init_vel: np.array, t_init: float, bulk_vel: np.array):
        """inital properties of the particle"""
        self.mass = mass
        self.r = r # radius
        self.vel = init_vel
        self.t = t_init
        self.posn_hist =  np.array([init_posn])
        self.bulk_vel = bulk_vel
        # TODO add flag for diffuse impact
        

    def reflect_specular(self, wall: np.array, dt: float, cell_n_i, cell_n_f):
        """calculate the reflected velocity for a specular wall impact
        Args:
            c (np.array): incomming velocity
            wall (np.array): wall normal vector, inwards facing, # TODO I think this needs to be inward facing...
            dt (float): timestep length
            tube_d (float): diameter of tube
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        intersect = self.posn_hist[-2] + pct_vect*dt*self.vel


        # ensure wall vector is a unit vector
        wall = wall/np.linalg.norm(wall)
        v0 = self.vel
        c_n = np.dot(self.vel,wall)*wall # normal component to wall
        c_p = self.vel - c_n # perpendicular component to wall
        self.vel = c_p - c_n # flip normal component
        dm = self.mass*(self.vel - v0) # change in momentum from wall collission

        # collision location
        self.posn_hist[-1] = intersect
        # post - collision location
        self.update_posn_hist(intersect + dt*(1 - pct_vect)*self.vel) # update position with fraction of remaining timestep

        return dm

    def reflect_diffuse(self, wall_n: np.array, dt: float, cell_n_i, cell_n_f, t_tw, c_m):
        """_summary_

        Args:
            wall_n (np.array): _description_
            dt (float): _description_
            tube_d (_type_): _description_
            cell_n_i (_type_): _description_
            cell_n_f (_type_): _description_
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        intersect = self.posn_hist[-2] + pct_vect*dt*self.vel
        
        # ensure wall vector is a unit vector
        wall_n = wall_n/np.linalg.norm(wall_n)
        tht = 1 # random direction to rotate into new vector
        dcm = np.array([[np.cos(tht), -np.sin(tht), 0], [np.sin(tht), np.cos(tht), 0], [0, 0, 1]])
        wall_t = np.cross(wall_n, dcm.dot(wall_n)) # find tangential vector to surface
        wall_b = np.cross(wall_n, wall_t)/np.linalg.norm(np.cross(wall_n, wall_t)) # find a third basis vector for the cell csys

        # find cell csys velocity
        v0 = self.vel
        cell_vel = gen_velocity(self.bulk_vel, np.sqrt(t_tw)*c_m, 0) # c_m scaled and s_n = 0
        self.vel = cell_vel[0]*wall_n + cell_vel[1]*wall_t + cell_vel[2]*wall_b # normal velocity with X in the cell_normal direction
        dm = self.mass*(self.vel - v0) # change in momentum from wall collission
        de = np.linalg.norm(self.vel)**2 - np.linalg.norm(v0)**2 # only needed for diffuse, this is zero for specular

        # update position
        self.update_posn_hist(intersect + dt*(1 - pct_vect)*self.vel) # update position with fraction of remaining timestep
        # remove bulk velocity
        self.bulk_vel = np.array([0, 0, 0])

        return dm, de


    def update_posn_hist(self, r: np.array):
        """append current position to position history
        Args:
            r (np.array): position vector
        """
        self.posn_hist = np.vstack([self.posn_hist[-1],r])

    def exit_domain_inlet(self, inlet_plane: float):
        """determine if particle has left domain
        Args:
            exit_plane (float): Z location of exit plane
        Returns:
            BOOL: if partice is within domain 
        """
        # TODO add in point and vector definition of plane to use dot product
        if self.posn_hist[-1][0] < inlet_plane:
            return True

    def exit_domain_outlet(self, exit_plane: float):
        """determine if particle has left domain
        Args:
            exit_plane (float): Z location of exit plane
        Returns:
            BOOL: if partice is within domain 
        """
        # TODO add in point and vector definition of plane to use dot product
        if self.posn_hist[-1][0] >= exit_plane:
            return True

