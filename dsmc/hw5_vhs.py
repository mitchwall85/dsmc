from signal import Sigmasks
import numpy as np
from utilities import gen_posn_3d, gen_velocity, KB
import scipy.special as sp


# Homework 5
# Mitch Wall

# Givens
rho = 8e-5  # kg/m^3
T = 500  # k
dref = 3.915e-10  # corrected typo from HW set
Tref = 273 # k
omega = 0.81
dx = 1.25e-3  # m
dz = 1e-3 # m
dt = 1.427e-7  # s
m = 66.3e-27  # kg # mass of argon
d = 4.17e-10  # m diameter of argon
#

# model properties
n_particles = 50000  # simulated particles
n_iter = 10
n_density = rho/m  # particles/m^3
v_dsmc = dx**2*dz
wp = n_density*v_dsmc/n_particles
nu = omega - 0.5  # eqn 6.24
mr = m/2  # 1.67
max_mult = 10 # sigma_t_g_max multiplier 

# allocate particle arrays
particle_posn = np.zeros([n_particles, 3])
particle_vel = np.zeros([n_particles, 3])

# velocity props
c_m = np.sqrt(2*KB*T/m)
s_n = 0 
bulk = np.array([0, 0, 0])

# randomly sample particles
for j in np.arange(0, n_particles):
    particle_posn[j, :] = gen_posn_3d(dx, dx, dz)
    particle_vel[j, :] = gen_velocity(bulk, c_m, s_n)
# number of possible collision pairs
n_pairs = np.floor(n_particles/2).astype(int)


crf = np.zeros([n_iter, 1])
sigma_t_g_max = 0
for n in np.arange(0, n_iter):

    # find collision pairs
    pairs = np.zeros([n_pairs, 2])  # array for storing collision pairs
    particle_index = np.linspace(0, n_particles-1, n_particles)
    # select two random particle indices
    for i in np.arange(0, n_pairs):
        # first particle in pair
        p1_ind = np.random.randint(particle_index.__len__())
        pairs[i, 0] = particle_index[p1_ind]
        particle_index = np.delete(particle_index, p1_ind, 0)

        # second particle in pair
        p2_ind = np.random.randint(particle_index.__len__())
        pairs[i, 1] = particle_index[p2_ind]
        particle_index = np.delete(particle_index, p2_ind, 0)

    pairs = pairs.astype(int)  # convert to ints to use as indices

    # relative velocities
    g_vect = particle_vel[pairs[:, 0], :] - particle_vel[pairs[:, 1], :]
    g = np.linalg.norm(g_vect, axis=1)

    # eqn 6.21
    gref = ((2*KB*Tref/mr)**nu/sp.gamma(2 - nu))**(1/2/nu) # fixed expression
    # eqn 6.17
    sigma_t = np.pi*dref**2*(gref/g)**(2*nu)
    sigma_t_g = np.multiply(g, sigma_t)
    max_val = np.max(sigma_t_g)*max_mult  # max rate of swept vol, m^3/s

    if max_val > sigma_t_g_max: # update if it has increased
        sigma_t_g_max = max_val

    # max number of collisions
    n_coll_max = 0.5*n_particles*(n_particles - 1)*sigma_t_g_max * wp*dt/v_dsmc  # should use sigma_t_g_max
    n_pairs_test = np.floor(n_coll_max + 0.5)
    fc = n_coll_max/n_pairs_test

    p_coll = fc*sigma_t_g[0:n_pairs_test.astype(int)]/sigma_t_g_max # only loop over first n_pairs_test collision pairs

    n_coll = 0
    for p in p_coll:
        if np.random.rand() <= p:
            n_coll += 1

    crf[n] = n_coll/n_particles

nu_sim_avg = np.mean(crf)/dt
print(f"nu_sim = {nu_sim_avg}")
