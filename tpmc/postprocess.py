import numpy as np

class POST_PROCESS:
    """postprocessing class
    """

    def __init__(self, vol_grid: np.array, wall_grid):
        """properties for post processing"""
        # volume parameters
        self.vol_grid = vol_grid
        self.n_density = np.zeros(np.shape(vol_grid)) 
        self.temp = np.zeros(np.shape(vol_grid)) 
        self.n_density = np.zeros(np.shape(vol_grid)) 
        self.dof = np.shape(vol_grid)[1]
        # surface parameters
        self.wall_grid = wall_grid
        self.collision_rate = np.zeros(np.shape(wall_grid.centroids)) 
        self.pressure = np.zeros(np.shape(wall_grid.centroids)) 
        self.shear = np.zeros(np.shape(wall_grid.centroids)) 
        self.heat_tf = np.zeros(np.shape(wall_grid.centroids)) 
        