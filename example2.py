# -*- coding: utf-8 -*-

# EMoptimizer usage example 2:
# Plot the position of all particles at each time step.
# WARNING: Creates a large number of PNG files in the current directory.

import numpy as np
import matplotlib.pylab as plt
from emoptimizer import EMoptimizer

# Define some commonly used optimization functions

def damavandi(x):
    # Range: [0,14] in each dimension
    # Global minimum at (2,2)
    x1,x2 = x[0], x[1]
    y1,y2 = np.pi*(x1-2), np.pi*(x2-2)
    return ( 1 - abs( np.sin(y1)*np.sin(y2) / (y1*y2) )**5 ) * (2 + (x1-7)**2 + 2*(x2-7)**2)

def damavandiV(x,y): # vectorized form for plotting
    return ( 1 - abs( np.sin(np.pi*(x-2))*np.sin(np.pi*(y-2)) / (np.pi**2 * (x-2)*(y-2)) )**5 ) * (2 + (x-7)**2 + 2*(y-7)**2)

def alpine02(x):
    # range: [0,10] in each direction
    # Global minimum near (8,8)
    return -np.sqrt(x[0]*x[1])*np.sin(x[0])*np.sin(x[1])

def alpine02V(x,y):  # vectorized form for plotting
    return -np.sqrt(x*y)*np.sin(x)*np.sin(y)

def eggcrate(x):
    # range: [-5,5] in each direction
    # global minimum at (0,0)
    return x[0]**2 + x[1]**2 + 25*(np.sin(x[0])**2 + np.sin(x[1])**2)

def eggcrateV(x,y):
    return x**2 + y**2 + 25*(np.sin(x)**2 + np.sin(y)**2)

def ackley(x):
    # range : [-32,32] in each direction
    # global minimum at (0,0)
    return -20*np.exp(-0.2*np.sqrt((x[0]**2+x[1]**2)/2)) - \
    np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + 20 + np.exp(1)

def ackleyV(x,y):
    return -20*np.exp(-0.2*np.sqrt((x**2+y**2)/2)) - \
    np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*x))) + 20 + np.exp(1)

# Use the Damavandi function
    
xmin, xmax, ymin, ymax = 0,14,0,14
    
x = np.linspace(xmin, xmax, 100)
y = np.linspace(ymin, ymax, 100)
X, Y = np.meshgrid(x,y)
Z = damavandiV(X,Y)

nparticles = 16
maxiters = 100
em = EMoptimizer(2, nparticles, damavandi, [xmin,ymin],[xmax,ymax])

while(em.iterno <= maxiters):

    plt.contour(X,Y,Z) # plot the contours
    # show particles as crosses
    plt.scatter(em.getpos()[:,0], em.getpos()[:,1],marker="x")
    # show the best point as a red circle         
    plt.plot(em.best.pos[0],em.best.pos[1],"ro")
    plt.title("iteration %d, best OFV %.4f" % (em.iterno,em.getbestofv()))
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    # save as png, numbered with iterations.
    plt.savefig("%03d.png" % (em.iterno,),format="png")
    plt.clf()
    
    em.iterate()
