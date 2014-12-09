'''
A Bayesian model for backing out 3-dimensional information of ellipsoids from their 2d projections.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta,cosine
import emcee

class ProjectEllipsoid:
    '''
    Project an ellipsoid with the semi-major axis aligned with the x-axis, down a line-of-sight at
    an angle of phi (x-y plane) and theta
    '''
    def __init__(self,phi,theta,axis_ratio=None):
        self.phi = phi
        self.theta = theta
        if axis_ratio==None:
            axis_ratio = 0.7
        self.axis_ratio = axis_ratio

        #ellipsoid coordinates
        self.a = 1.0
        self.b = self.a*self.axis_ratio
        self.c = self.a*self.axis_ratio
        
        #rotated constants
        self.A = np.cos(self.theta)**2/self.c**2*(np.sin(self.phi)**2 + np.cos(self.phi)**2/self.b**2) + np.sin(self.theta)**2/self.b**2
        self.B = np.cos(self.theta)*np.sin(2*self.phi)*(1-1.0/self.b**2)*1.0/self.c**2
        self.C = (np.sin(self.phi)**2/self.b**2 + np.cos(self.phi)**2)*1.0/self.c**2

        self.projected_ratio = np.sqrt((self.A+self.C - np.sqrt((self.A-self.C)**2 + self.B**2))/(self.A+self.C+np.sqrt((self.A-self.C)**2 + self.B**2)))


#To randomly pick a point on a unit sphere
def unitsphere():
    return 2*np.pi*np.random.uniform(0,1),np.arccos(1-2*np.random.uniform(0,1))

#return sin value between 0 and pi for theta probability
def anglevalue(phi,theta):
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(theta)
    xangle = np.arccos(x)
    rv = cosine()
    return rv.pdf(xangle*2 - np.pi)

#return random beta value
def betavalue(ratio):
    rv = beta(14,6)
    return rv.pdf(ratio)

#define a probability model
def lnprob(p,ratio):
    if p[0] < 0 or p[0] > 2*np.pi: P0 = -np.inf
    else: P0 = 0.0
    if p[1] < 0 or p[1] > np.pi: P1 = -np.inf
    else: P1 = 0.0
    betamod = beta(np.int(np.floor(ratio*100)),100-np.int(np.floor(ratio*100)))
    P = ProjectEllipsoid(p[0],p[1],p[2])
    return np.log(betamod.pdf(P.projected_ratio)*anglevalue(p[0],p[1])*betavalue(p[2])) + P0 + P1


if __name__ == '__main__':
    measured_ratio = 0.99999999
    ndim,nwalkers = 3,50
    theta_init = np.random.uniform(0,np.pi,nwalkers)
    phi_init = np.random.uniform(0,2*np.pi,nwalkers)
    dratio_init = np.random.uniform(0.6,1.0,nwalkers)
    p0 = np.vstack((phi_init,theta_init,dratio_init)).T
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[measured_ratio])
    print 'Burn in for MCMC...'
    pos, prob, state = sampler.run_mcmc(p0,100)
    sampler.reset()
    print 'Stepping MCMC for intrinsic scatter...'
    sampler.run_mcmc(pos, 1000, rstate0=state)

