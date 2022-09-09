import numpy as np
from particle import PARTICLE


t = np.linspace(0,1,10)
dt = t[1] - t[0]
c = np.array([1,2,0])
r = np.array([0,0,0])
particle = PARTICLE(1, 1, r, c, t[0])

posn_hist = []
for i in t:
    dx = particle.vel*dt
    posn_hist.append(particle.posn)
    particle.posn = particle.posn + dx

print(posn_hist)
