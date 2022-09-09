import numpy as np

class PARTICLE:
    """particle model for TPMC & DSMC code
    """
    def __init__(self, mass, r, init_posn, init_vel, t_init):
        """inital properties of the particle"""
        self.mass = mass
        self.r = r # radius
        self.posn = init_posn
        self.vel = init_vel
        self.t = t_init


    def reflect_spectular(self, wall: np.array):
        """calculate the reflected velocity for a specular wall impact

        Args:
            c (np.array): incomming velocity
            wall (np.array): wall normal vector
        """
        # ensure wall vector is a unit vector
        wall = wall/np.linalg.norm(wall)
        v0 = self.vel
        c_n = np.dot(self.vel,wall)*wall # normal component to wall
        c_p = self.vel - c_n # perpendicular component to wall
        self.vel = c_p - c_n # flip normal component
        dm = self.mass*(self.vel - v0) # change in momentum from wall collission
        return dm

    # TODO s
    # function to add itself to tracked particles
    # function to remove itself from tracked particles
    # function to record posn history