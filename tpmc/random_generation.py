import numpy as np
import matplotlib.pyplot as plt
import time

n = 10000

r_rand = []
tht_rand = []

t_loop = time.time()
for i in np.arange(0,n):
    r_rand.append(2*np.sqrt(np.random.rand(1,1)))
    tht_rand.append(np.random.rand(1)*2*np.pi)
t_loop = time.time() - t_loop
print(f"'t_loop:', {t_loop}")

t_vect = time.time()
r_vect = 2*np.sqrt(np.random.rand(1,n))
tht_vect = np.random.rand(1,n)*2*np.pi
t_vect = time.time() - t_vect
print(f"'t_vect:', {t_vect}")

# comparing the two times for the loop and the vector we get:
# 't_loop:', 0.0661473274230957
# 't_vect:', 0.0001685619354248047


fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='polar')
ax1.scatter(tht_rand, r_rand, s=1)  
plt.savefig('random_polar_distribution.png')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='polar')
ax2.scatter(tht_vect, r_vect, s=1)  
plt.show()