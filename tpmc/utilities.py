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

def gen_velocity(blk: np.ndarray, c_m, s_n ):
    """generate a boltzmann distribution of velocities

    Args:
        blk (np.ndarray): bulk velocity
        T (float): temperature
        m (float): mass of molecule
        k (np.array, optional): boltzmann const.

    Returns:
        c: velocity vector [surface normal, tangential 1, tangential 2]
    """

    # blk = [0, 300, 1000] # bulk velocities to test
    # k = 1.380649e-23 # boltzmann constant, kb
    # Ttr = 300 # translational temp
    # m = 28.0134/1000/6.02e23 # mass of a N2 molecule
    
    # tangential components
    r1 = np.random.rand(2)
    r2 = np.random.rand(2)
    v_2_3 = c_m*np.sin(2*np.pi*r1)*np.sqrt(-np.log(r2))

    # normal component
    pdf_max = 0.5*(np.sqrt(s_n**2 + 2) - s_n)
    h = np.sqrt(s_n**2 + 2)
    k_cap = 2/(s_n + h)*np.exp(0.5 + 0.5*s_n*(s_n - h))
    y = cx/c_m # cm = 1/beta? p 317 boyd
    pdf_norm = k_cap*(y + s_n)*np.exp(-y**2)

    y = -3 + 6*np.random.rand(1)


    return  blk + np.sqrt(2*k*T/m)*np.sin(2*np.pi*r1)*np.sqrt(-np.log(r2)) # A.20 from boyd


def gen_posn(diam: float):
    """generate position on circular face, assumes z=0 at inlet face

    Args:
        diam (float): diameter of inlet face

    Returns:
        posn: xyz coords
    """

    r = diam/2*np.sqrt(np.random.rand(1))
    tht = np.random.rand(1)*2*np.pi
    posn = np.concatenate([np.array([0]), r*np.cos(tht), r*np.sin(tht)])

    return posn