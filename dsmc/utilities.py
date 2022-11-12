import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import os

# Constants
KB = 1.380649e-23 # [m^2*kg/s^2/K]
M = 28.0134/1000/6.02e23 # mass of a N2 molecule [kg] TODO this is terrible to hardcode. fix later
AVOS_NUM = 6.02e23

def read_stl(file: str):

    return mesh.Mesh.from_file(file)

# TODO s
# something to visualize geometry of grid and test object



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
            cell_points = np.roll(cell_points,3)

    return in_elem


def start_postproc(pct_window, pp_tolerance, particles, particles_per_timestep, i):

    # detect if steady state is reached and if post processing should start
    window = int(np.ceil(pct_window*i))
    avg_window = np.mean(particles[1][-window:])
    if avg_window > particles_per_timestep*(1 - pp_tolerance) and avg_window < particles_per_timestep*(1 + pp_tolerance):
        print('*******************START POST PROCESSING*******************')
        return True
    

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
    pdf_max = 0.5*(np.sqrt(s_n**2 + 2) - s_n) # A.32
    h = np.sqrt(s_n**2 + 2) # A.34
    k_cap = 2/(s_n + h)*np.exp(0.5 + 0.5*s_n*(s_n - h)) # A.34
    # while loop until staisfactory value is found
    value = False
    while value == False:
        y = -3 + 6*np.random.rand(1) # random number: -3 < y < 3
        r2 = np.random.rand(1) # step 2 of procedure on p 319
        pdf_norm = k_cap*(y + s_n)*np.exp(-y**2) # A.33
        if r2 < pdf_norm:
            v_1 = y*c_m # step 3 
            value = True


    return  np.array([v_1[0], v_2_3[0], v_2_3[1]]) + blk # return velcity vector with bulk added on


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

        y = dy*np.random.rand(1) + grid.min_[1] # does this require a center at 0,0? -- yes, fix that TODO
        z = dz*np.random.rand(1) + grid.min_[2]
        r = np.array([0, y[0], z[0]])

        
        for c in np.arange(np.shape(grid.centroids)[0]): # replace with in_element()

            if in_element(grid.points[c], grid.normals[c], r):
                try_again = False
                break
            else:
                try_again = True

    return np.array([0, y[0], z[0]])

def gen_posn_3d(dx, dy, dz): # janky function for generating position in a 3d box

    return np.array([np.random.rand()*dx, np.random.rand()*dy, np.random.rand()*dz])

def make_directory(dir_name: str):
    """makes directory if it does not already exist"""

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def plot_walls(ax, grid):
    a = 1
    num_pts = grid.points.shape[0]
    pts = np.zeros([num_pts*3, 3])
    c = 0
    for p in np.arange(0,num_pts):
        for i in [0,1,2]:
            pts[c,:] = grid.points[p,i*3:i*3+3]
            c+=1
    
    ax.scatter3D(pts[:,0], pts[:,1], pts[:,2], c='r')


