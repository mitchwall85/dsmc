import numpy as np
import matplotlib.pyplot as plt

n = 10000
blk = [0, 300, 1000] # bulk velocities to test
k = 1.380649e-23 # boltzmann constant, kb
Ttr = 300 # translational temp
m = 28.0134/1000/6.02e23 # mass of a N2 molecule

ci = [[[]]]

ci = np.empty([np.size(blk),4,n]) # 2nd index: [vX, vY, vZ, norm(v)]
b_enter = np.empty([np.size(blk),1])
for b in np.arange(0,np.size(blk)):
    r1 = np.random.rand(3,n)
    r2 = np.random.rand(3,n)
    ci[b,0:-1,:] = blk[b] + np.sqrt(2*k*Ttr/m)*np.sin(2*np.pi*r1)*np.sqrt(-np.log(r2)) # A.20 from boyd
    ci[b,-1,:] = np.sqrt(ci[b,0,:]**2 + ci[b,1,:]**2 + ci[b,2,:]**2) # place rss in 3th index
    count = np.bincount(ci[b,0,:] > 0) # count how many particles have +x velocity and will enter domain
    b_enter[b] = count[1]

# calculate MB distribution
# expression found from: https://physics.stackexchange.com/questions/320500/maxwell-boltzmann-distribution-average-speed-in-one-direction
vi = np.linspace(-2000,2000,1000)
mb = np.sqrt(m/2/np.pi/k/Ttr)*np.exp(-m*vi**2/2/k/Ttr)

    
# A:    
# plot each component
comp = ['X', 'Y', 'Z']
for d in np.arange(0,np.size(comp)):
    plt.figure()
    plt.hist(ci[0,d,:],100, density=True)
    plt.plot(vi,mb)
    plt.xlabel(f"{comp[d]} Velocity [m/s]")
    plt.ylabel('PDF')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{comp[d]}_velocity.png")

# plot RSS
    plt.figure()
    plt.hist(ci[0,3,:],100, density=True)
    plt.xlabel(f"RSS Velocity [m/s]")
    plt.ylabel('PDF')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"RSS_velocity.png")

# B:
print('B: Number of particles entering domain for each freestream velocity')
print(b_enter)