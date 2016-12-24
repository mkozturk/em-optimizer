# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:33:53 2016

@author: kaan
"""

# Find the global optimum of the Ackley function.
# Keep track of the objective function value.


import numpy as np
import matplotlib.pylab as plt
from emoptimizer import EMoptimizer

def ackley(x):
    # range : [-32,32] in each direction
    # global minimum at (0,0)
    return -20*np.exp(-0.2*np.sqrt((x[0]**2+x[1]**2)/2)) - \
    np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + 20 + np.exp(1)

em = EMoptimizer(dim=2, nparticles=16, objective=ackley, lower=[-32,-32], upper=[32,32])

ofvlist = [em.getbestofv()]

# Make 100 iterations
while(em.iterno < 100):
    em.iterate()
    ofvlist.append(em.getbestofv())

plt.figure()
plt.plot(ofvlist,linewidth=2)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("best objective-function value")

# Restart and iterate until the ofv drops below 0.01.
em = EMoptimizer(dim=2, nparticles=16, objective=ackley, lower=[-32,-32], upper=[32,32])

ofvlist = [em.getbestofv()]

while(ofvlist[-1] > 0.01 and em.iterno < 10000):
    em.iterate()
    ofvlist.append(em.getbestofv())

plt.figure()
plt.plot(ofvlist,'r',linewidth=2)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("best objective-function value")