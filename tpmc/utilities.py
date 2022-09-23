import numpy as np
from stl import mesh
from particle import PARTICLE

# Constants
KB = 1.380649e-23 # [m^2*kg/s^2/K]

def check_in_cell(r: np.array, cell: np.array):
    a = 1
    # TODO move to particle class?


def read_stl(file: str):

    return mesh.Mesh.from_file(file)

# TODO s
# something to visualize geometry of grid and test object

def gen_velocity(blk: np.ndarray, T, m, k ):
    """generate a boltzmann distribution of velocities

    Args:
        blk (np.ndarray): bulk velocity
        T (float): temperature
        m (float): mass of molecule
        k (np.array, optional): boltzmann const.

    Returns:
        c: velocity vector 
    """

    # blk = [0, 300, 1000] # bulk velocities to test
    # k = 1.380649e-23 # boltzmann constant, kb
    # Ttr = 300 # translational temp
    # m = 28.0134/1000/6.02e23 # mass of a N2 molecule
    
    r1 = np.random.rand(3)
    r2 = np.random.rand(3)
    return  blk + np.sqrt(2*k*T/m)*np.sin(2*np.pi*r1)*np.sqrt(-np.log(r2)) # A.20 from boyd
    # return  blk + np.array([np.sqrt(2*k*T/m)*np.sin(2*np.pi*r1)*np.sqrt(-np.log(r2)), 0, 0]) # A.20 from boyd


def gen_posn(diam: float):
    """generate position on circular face, assumes z=0 at inlet face

    Args:
        diam (float): diameter of inlet face

    Returns:
        posn: xyz coords
    """

    r = diam*np.sqrt(np.random.rand(1))
    tht = np.random.rand(1)*2*np.pi
    posn = np.concatenate([np.array([0]), r*np.cos(tht), r*np.sin(tht)])

    return posn