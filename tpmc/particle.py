import numpy as np

class PARTICLE:
    """particle model for TPMC & DSMC code
    """
    def __init__(self, mass: float, r: float, init_posn: np.array, init_vel: np.array, t_init: float):
        """inital properties of the particle"""
        self.mass = mass
        self.r = r # radius
        self.vel = init_vel
        self.t = t_init
        # self.posn_hist =  np.vstack([np.array([0,0,0]),init_posn]) # zeros isnt perfect but it gets overwritten
        self.posn_hist =  init_posn # zeros isnt perfect but it gets overwritten

    def reflect_specular(self, wall: np.array, dt: float, tube_d, cell_n_i, cell_n_f):
        """calculate the reflected velocity for a specular wall impact
        Args:
            c (np.array): incomming velocity
            wall (np.array): wall normal vector, inwards facing, # TODO I think this needs to be inward facing...
            dt (float): timestep length
            tube_d (float): diameter of tube
        """

        pct_vect = np.abs(cell_n_i)/np.abs(cell_n_i - cell_n_f)
        intersect = self.posn_hist[-2] + pct_vect*dt*self.vel # TODO check 


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

    def update_posn_hist(self, r: np.array):
        """append current position to position history
        Args:
            r (np.array): position vector
        """
        self.posn_hist = np.vstack([self.posn_hist,r])

    def exit_domain(self, exit_plane: float):
        """determine if particle has left domain
        Args:
            exit_plane (float): Z location of exit plane
        Returns:
            BOOL: if partice is within domain 
        """
        # TODO add in point and vector definition of plane to use dot product
        if self.posn_hist[-1][0] >= exit_plane:
            return True

    # TODO s
    # function to add itself to tracked particles