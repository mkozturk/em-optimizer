# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

class EMoptimizer:
    """A class for EM optimization.
        
    Initialization:

    em = EMoptimizer(dim, nparticles, objective, lower, upper, stepreduction=False, circular=False)

    where
    dim : problem dimension (number of arguments to the objective function)
    nparticles: number of particles to use
    objective:  the objective function to minimize. Must take a single iterable argument.
    lower : the lower "corner" of the region of interest.
    upper : the upper "corner" of the region of interest.    
    stepreduction : If True, step sizes are scaled with (iteration)^(-0.25).
    circular : If True, the region is treated as a torus (out from one edge, in from the other edge).

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
    def __init__(self, dim, nparticles, objective, lower, upper,
                 stepreduction = False, circular = False):
        self.dim = dim # dimension of the problem
        self.nparticles = nparticles # number of particles
        self.f = objective  # the function to minimize, R^n -> R
        self.upper = np.array(upper) # the upper limit of the region
        self.lower = np.array(lower) # the lower limit of the region
        self.pack = _Pack(self)  # Initialize a Pack, passing itself as parameter to it.
        self.iterno = 0
        self.stepreduction = stepreduction  # If true, reduces the step size as t^(-0.25)
        self.circular = circular  # if true, the boundaries are circular (region is toroidal)
        self.localsearchmaxiter = 100
        self.localsearchstep = 0.01
        
    def iterate(self):
        """Make one iteration of the algorithm over all particles."""
        self.pack.moveall()
        self.iterno += 1

    def getpos(self):
        """Return the positions of particles as an m-by-n array."""
        return np.array([p.pos for p in self.pack.particles])

    def getofv(self):
        """Return the objective function values as an array of size m."""
        return np.array([self.f(p.pos) for p in self.pack.particles])

    def getbestpos(self):
        """Return the position of the best particle."""
        return self.pack.best.pos
        
    def getbestofv(self):
        """Return the objective function value of the current best particle."""
        return self.f(self.pack.best.pos)

class _Particle:
    """Properties and operations on a single particle. Initialized and used
    by the _Pack class."""
    def __init__(self, pos, pack):
        self.pack = pack
        self.pos = pos
        self.q = None
        self.force = None

    def localsearch(self):
        """Updates Particle position with a local coordinate search."""
        counter = 1
        for k in range(self.pack.opt.dim):
            direction = np.random.choice((-1,1))
            while counter < self.pack.opt.localsearchmaxiter:
                y = deepcopy(self.pos)
                r = np.random.uniform()
                y[k] += direction*r*self.pack.opt.localsearchstep*(self.pack.opt.upper[k]-self.pack.opt.lower[k])
                if self.pack.opt.f(y) < self.pack.opt.f(self.pos):
                    self.pos = deepcopy(y)
                    break
                counter += 1
    def move(self):
        """Update particle position in the force direction."""
        if self is self.pack.best:
            return # do not move if this is the best position
        r = np.random.uniform()
        if self.pack.opt.stepreduction:
            r = (self.pack.opt.iterno+1)**(-0.25) * r
        force = self.force
        forcemag = np.sqrt(np.dot(force,force))
        if forcemag > 1e-15:
            force = force / np.sqrt(np.dot(force,force))  # get the direction

        if self.pack.opt.circular:
            for k in range(self.pack.opt.dim):
                self.pos[k] += r*force[k]*(self.pack.opt.upper[k]-self.pack.opt.lower[k])
                # Adjust particle positions with periodic boundary conditions:
                if self.pos[k] > self.pack.opt.upper[k]:
                    self.pos[k] = self.pos[k] - (self.pack.opt.upper[k]-self.pack.opt.lower[k])
                if self.pos[k] < self.pack.opt.lower[k]:
                    self.pos[k] = self.pos[k] + (self.pack.opt.upper[k]-self.pack.opt.lower[k])
        else:
            for k in range(self.pack.opt.dim):
                if force[k] > 0:
                    self.pos[k] += r * force[k] * (self.pack.opt.upper[k] - self.pos[k])
                else:
                    self.pos[k] += r * force[k] * (self.pos[k] - self.pack.opt.lower[k])

class _Pack:
    """A class for a pack of Particles. Initialized and used by EMoptimizer."""
    def __init__(self, optimizer):
        self.opt = optimizer # The optimizer that launched this Pack instance.
        
        self.particles = []
        for n in range(self.opt.nparticles):
            pos = np.zeros(self.opt.dim)
            for d in range(self.opt.dim):
                pos[d] = np.random.uniform(self.opt.lower[d], self.opt.upper[d])
            self.particles.append(_Particle(pos,self))

        self.best = self.getbest()
        self.farthest = self.getfarthest()
        self.setcharges()
        self.setforces()

    def moveall(self):
        """Moves every particle, updates the charges and forces."""
        # This is the only function directly called by EMoptimizer.
        
        # All the idiosyncratic details about moving an individual particle,
        # such as special handling of the best, 
        # must be implemented in _Particle.move().
        for p in self.particles:
            p.move()
            
        # Particles can't know if they are the best. It can be checked only
        # at the Pack level.
        self.best = self.getbest()
        self.best.localsearch()
        self.farthest = self.getfarthest()        
        # Similarly, charge and force values require knowledge of the whole pack.
        self.setcharges()
        self.setforces()        

    def getbest(self):
        """Return the particle with the lowest OFV among the pack."""
        best = self.particles[0]
        for p in self.particles:
            if self.opt.f(p.pos) < self.opt.f(best.pos):
                best = p
        return best

    def getfarthest(self):
        """Return the particle that is farthest from the current best."""
        farthest = self.particles[0]  
        best = self.getbest()
        for p in self.particles:
            if np.linalg.norm(p.pos-best.pos) > np.linalg.norm(farthest.pos-best.pos):
                farthest = p
        return farthest
        
    def setforces(self):
        """Sets the force vectors of every particle in pack, in place."""
        for p in self.particles:
            p.force = np.zeros(self.opt.dim)
            factor = 1
            for pp in self.particles:
                if p is pp:
                    continue
                if p is self.farthest:
                    factor = np.random.choice([-1,1])
                if self.opt.f(pp.pos) < self.opt.f(p.pos):
                    p.force += factor*(pp.pos - p.pos)*pp.q*p.q/np.dot(pp.pos-p.pos,pp.pos-p.pos)
                else:
                    p.force -= factor*(pp.pos - p.pos)*pp.q*p.q/np.dot(pp.pos-p.pos,pp.pos-p.pos)

    def setcharges(self):
        """Sets the charges of every particle in pack, in place."""
        denom = sum([self.opt.f(p.pos)-self.opt.f(self.best.pos) for p in self.particles])
        for p in self.particles:
            p.q = np.exp(-self.opt.dim * (self.opt.f(p.pos)-self.opt.f(self.best.pos))/denom)
