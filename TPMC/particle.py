class particle:
    """particle model for TPMC & DSMC code
    """
    def __init__(self, mass, r, init_posn, init_vel, t_init):
        """inital properties of the particle"""
        self.mass = mass
        self.r = r
        self.init_posn = init_posn
        self.init_vel = init_vel
        self.t_init = t_init