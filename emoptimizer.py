# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:15:51 2016

@author: kaan
"""
import numpy as np
from copy import deepcopy

class EMoptimizer:
    """A class for EM optimization.
        
    Initialization:

    em = EMoptimizer(dim, nparticles, objective, lower, upper)

    where
    dim : problem dimension (number of arguments to the objective function)
    nparticles: number of particles to use
    objective:  the objective function to minimize. Must take a single iterable argument.
    lower : the lower "corner" of the region of interest.
    upper : the upper "corner" of the region of interest.    
    
    Attributes:
        iterno: The current iteration number
    
    Methods:
        iterate(): Make one iteration of the algorithm over all particles.
        getpos(): Return the positions of all particles.
        getofv(): Return the objective function values of all particles.
        getbestpos(): Return the position of the current best particle.
        getbestofv(): Return the objective function value of the current best particle.
        
    Example usage
    -------------
    def alpine02(x):
        return -np.sqrt(x[0]*x[1])*np.sin(x[0])*np.sin(x[1])

    em = EMoptimizer(dim=2, nparticles=8, objective=alpine02,
                     lower=[0,0], upper=[10,10])
    
    while(em.iterno < 100):
        em.iterate()
    
    print em.getbestpos(), em.getbestofv()
    
    """
    class Particle:
        def __init__(self,x):
            self.pos = x
            self.q = 0
            self.force = 0
            
    def __init__(self, dim, nparticles, objective, lower, upper):
        self.dim = dim # dimension of the problem
        self.npart = nparticles # number of particles
        self.f = objective  # the function to minimize, R^n -> R
        self.upper = np.array(upper) # the upper limit of the region
        self.lower = np.array(lower) # the lower limit of the region
        self.pack = self.initpack()
        self.best = self.getbest() # the particle in the best position
        self.iterno = 0
        self.initpack()
    
    def getq(self,x):
        """Return the charge for the given position x."""
        denom = sum([self.f(x)-self.f(self.best.pos) for p in self.pack])
        return np.exp(-self.dim * (self.f(x)-self.f(self.best.pos))/denom)

    def initpack(self):
        """Return a randomly initialized list of Particles."""
        pack=[]
        for i in range(self.npart):
            x = np.zeros(self.dim)
            for k in range(self.dim):
                x[k] = np.random.uniform(self.lower[k], self.upper[k])
            p = self.Particle(x)
            p.force = np.zeros(self.dim)
            pack.append(p)
        # set the charges...
        best = self.getbest(pack)
        denom = sum([self.f(p.pos) - self.f(best.pos) for p in pack])
        for p in pack:
            p.q = np.exp(-self.dim * (self.f(p.pos)-self.f(best.pos)/denom))
        
        return pack
        
    def getbest(self, pack=None):
        """Determine the best particle in the pack."""
        # Call either as getbest() or getbest(pack)
        if pack==None:
            pack = self.pack
            
        best = pack[0]
        for p in pack[1:]:
            if self.f(p.pos) < self.f(best.pos):
                best = p
        return best
    
    def calcF(self,p):
        """Set the force vector of one Particle."""
        p.force = np.zeros(self.dim)
        for pp in self.pack:
            if p is pp:
                continue
            if self.f(pp.pos) < self.f(p.pos):
                p.force += (pp.pos - p.pos)*pp.q*p.q/np.dot(pp.pos-p.pos,pp.pos-p.pos)
            else:
                p.force -= (pp.pos - p.pos)*pp.q*p.q/np.dot(pp.pos-p.pos,pp.pos-p.pos)
    
    def move(self,p):
        """Update particle position in the force direction."""
        if p is self.best:
            return # do not move if this is the best position
        r = np.random.uniform()
        force = p.force
        force = force / np.sqrt(np.dot(force,force))  # get the direction
        for k in range(self.dim):
            if force[k] > 0:
                p.pos[k] += r * force[k] * (self.upper[k] - p.pos[k])
            else:
                p.pos[k] += r * force[k] * (p.pos[k] - self.lower[k])


    def localsearch(self, maxiter=100, delta=0.001):
        """Updates Particle position with a local minimum search."""

        particles = [self.best]
        #particles = self.pack
        for p in particles:
            counter = 1
            for k in range(self.dim):
                direction = np.random.choice((-1,1))
                while counter < maxiter:
                    y = deepcopy(p.pos)
                    r = np.random.uniform()
                    y[k] += direction*r*delta*(self.upper[k]-self.lower[k])
                    if self.f(y) < self.f(p.pos):
                        p.pos = deepcopy(y)
                        break
                    counter += 1
                    

    def iterate(self):
        """Make one iteration of the optimizer."""
        self.localsearch()
        self.best = self.getbest()
        for p in self.pack:
            self.calcF(p)
        for p in self.pack:
            self.move(p)
        # Update the particle charges
        self.best = self.getbest()
        denom = sum([self.f(p.pos) - self.f(self.best.pos) for p in self.pack])
        for p in self.pack:
            p.q = np.exp(-self.dim * (self.f(p.pos)-self.f(self.best.pos)/denom))
        self.iterno += 1
        
    def getpos(self):
        """Return the positions of particles as an m-by-n array."""
        return np.array([p.pos for p in self.pack])
    def getbestpos(self):
        """Return the position of the best particle."""
        return self.best.pos
    def getofv(self):
        """Return the objective function values as an array of size m."""
        return np.array([self.f(p.pos) for p in self.pack])
    def getbestofv(self):
        """Return the objective function value of the current best particle."""
        return self.f(self.best.pos)