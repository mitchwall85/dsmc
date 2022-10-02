import numpy as np
from stl import mesh
from particle import PARTICLE
import matplotlib.pyplot as plt

# Constants
KB = 1.380649e-23 # [m^2*kg/s^2/K]

def check_in_cell(r: np.array, cell: np.array):
    a = 1
    # TODO move to particle class?


def read_stl(file: str):

    return mesh.Mesh.from_file(file)

# TODO s
# something to visualize geometry of grid and test object

def gen_velocity(blk: np.ndarray, c_m, s_n ): # TODO How does this even work anymore?
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


def gen_posn(grid): # whats the type hint here?
    """ generate random point on inlet surface

    Args:
        grid (stl.mesh.Mesh): Requires STL inlet surface grid. Must be flat with x normal

    Returns:
        list: [y, z] point on inlet
    """
    # only works on flat surface with X normal
    dy = grid.max_[1] - grid.min_[1]
    dz = grid.max_[2] - grid.min_[2]
    try_again = True
    while try_again:

        y = dy*np.random.rand(1) - grid.max_[1] # does this require a center at 0,0?
        z = dz*np.random.rand(1) - grid.max_[2]
        r = np.array([0, y[0], z[0]])


        for c in np.arange(np.shape(grid.centroids)[0]): # TODO generalize to squares?
            v1 = grid.points[c][0:3] - grid.points[c][3:6]
            v1_1 = np.cross(np.array([1,0,0]), v1)
            s1 = v1_1.dot(r - grid.points[c][0:3])

            v2 = grid.points[c][3:6] - grid.points[c][-3:]
            v2_1 = np.cross(np.array([1,0,0]), v2)
            s2 = v2_1.dot(r - grid.points[c][3:6])

            v3 = grid.points[c][-3:] - grid.points[c][0:3]
            v3_1 = np.cross(np.array([1,0,0]), v3)
            s3 = v3_1.dot(r - grid.points[c][-3:])

            if s1 < 0 and s2 < 0 and s3 < 0:
                try_again = False
                break
            else:
                try_again = True

    return np.array([0, y[0], z[0]])


def in_element(cell_points, normal, intersect):

    num_p = int(cell_points.__len__()/3)
    cell_points = cell_points.reshape(3,num_p)

    in_elem = True
    for p in np.arange(0,num_p):
        v1 = cell_points[0] - cell_points[1]
        v2 = np.cross(normal, v1) # IS THIS AN OUTWARD FACING NORMAL?
        s1 = v2.dot(intersect - cell_points[0])

        if s1 > 0:
            in_elem = False 
            break
        else:
            np.roll(cell_points,3)

    return in_elem
    
    