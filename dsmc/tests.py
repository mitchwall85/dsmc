import numpy as np

from particle import PARTICLE

from utilities import read_stl, gen_velocity, gen_posn, KB

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from case_tpmc import CASE_TPMC



# test random inlet generation
def random_inlet_positions(case: CASE_TPMC, n: int):

    inlet_grid = read_stl(case.inlet_grid_name)

    r = []
    for i in np.arange(1,n):
        b = gen_posn(inlet_grid)    
        r.append(b) 

    y = [item[1] for item in r]
    z = [item[2] for item in r]

    pts = inlet_grid.points.reshape((inlet_grid.__len__()*3, 3))

    # debugging plots
    plt.scatter(y, z, s = 2, label='Generated Points')
    plt.scatter(pts[:,1],pts[:,2],c='red', label= 'Inlet Grid Vertices')
    plt.legend()
    plt.xlabel('Y Coordinate [m]')
    plt.ylabel('Z Coordinate [m]')
    plt.grid()
    plt.title(f'Position Test: {case.case_name}')
    plt.savefig(f"{case.output_dir}/inlet_posn_verifiction.png")
    plt.close()

def random_inlet_velocity(case: CASE_TPMC, n: int):
    """test random velocity generation
    """

    inlet_grid = read_stl(case.inlet_grid_name)

    vx = []
    vy = []
    vz = []
    surf_normal = np.array([1, 0, 0]) # assume surgace normal is in X

    c_m = np.sqrt(2*KB*case.freestream_temp/case.m)
    v_bar = np.dot(case.freestream_vel, surf_normal)
    s_n = (np.dot(v_bar, surf_normal)/c_m)[0]# A.26

    for i in np.arange(1,50000):
        b = gen_velocity(case.freestream_vel, c_m, s_n)  
        vx.append(b[0])
        vy.append(b[1])
        vz.append(b[2])

    # analitical maxwell-boltzmann
    vi = np.linspace(-2000,2000,1000)
    mb = np.sqrt(case.m/2/np.pi/KB/case.freestream_temp)*np.exp(-case.m*vi**2/2/KB/case.freestream_temp)
    # https://openstax.org/books/university-physics-volume-2/pages/2-4-distribution-of-molecular-speeds#:~:text=Maxwell%2DBoltzmann%20Distribution%20of%20Speeds&text=f%20(%20v%20)%20%3D%204%20%CF%80,2%20k%20B%20T%20)%20)%20.
    # mb_speed = 4/np.sqrt(np.pi)*(M/2/KB/FREESTREAM_TEMP)**(3/2)*vi_speed**2*np.exp(-M*vi_speed**2/2/KB/FREESTREAM_TEMP)
    mb_speed = 1/4*vi*mb
    mb_speed = 0.002*mb_speed/np.max(mb_speed)

    plt.figure(1)
    plt.subplot(311)
    plt.hist(vx,100, density=True)
    plt.plot(vi,mb_speed)
    plt.xlim([0, max(vi)])
    plt.ylim([0, 1.1*max(mb_speed)])
    plt.ylabel('vX PDF')
    plt.subplot(312)
    plt.hist(vy,100, density=True)
    plt.plot(vi,mb)
    plt.ylabel('vY PDF')
    plt.subplot(313)
    plt.hist(vz,100, density=True)
    plt.plot(vi,mb)
    plt.ylabel('vZ PDF')
    plt.xlabel('Velocity [m/s]')
    plt.title(f'Velocity Test: {case.case_name}')
    plt.savefig(f"{case.output_dir}/inlet_velocity_verifiction.png")
    plt.close()


def plot_grid(grid_name: str):

    # test random inlet generation
    inlet_grid = read_stl(grid_name)

    num_points = np.shape(inlet_grid.x)[0]

    x = np.reshape(inlet_grid.x, (1,num_points*3))
    y = np.reshape(inlet_grid.y, (1,num_points*3))
    z = np.reshape(inlet_grid.z, (1,num_points*3))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z)
    plt.show()

