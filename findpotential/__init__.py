''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
NUMERICAL AND NFW MASS/DENSITY/POTENTIAL/V_ESC CALCULATIONS
    
    Description: The main function to call is find_potential(). There are several other
    supporting functions which can be called as well.

    find_potential(gal_data, clus_data, red_z, Rvir):

    gal_data -- A (Nx6) array with the column order as: gal_x,gal_y,gal_z,gal_vx,gal_vy,gal_vz.
                The positions are assummed to be in [Mpc] and the velocities in [km/s]

    clus_data -- A 6 element array with order: clus_x,clus_y,clus_z,clus_vx,clus_vy,clus_vz.
                 The positions are assummed to be in [Mpc] and the velocities in [km/s]

    red_z -- The cluster redshift

    Rvir -- The virial radius or critical radius (r200). Assummed to be in [Mpc]

    Outputs: The program calculates the cumulative mass profile, density profile, 
    potential profile, and escape velocity for each cluster. There are no file 
    outputs, however the program creates two plots that can be saved. The first is 
    the cumulative/NFW mass profiles and the second is the phase space plot with the 
    various escape velocity solutions overplotted.

Author: Dan Gifford
Year: 2014
Institution: University of Michigan
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import division
from math import *
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import astropy
import numpy as np
import scipy.optimize as optimize
from astropy.io import ascii
from scipy import ndimage
from astropy.io import fits as pyfits
from scipy.integrate import cumtrapz

#####CONSTANTS#####
h = 0.73 #km/s/Mpc/100.0
rmax = 10 #Mpc
G =4.518e-48 #Mpc^3 / (Msol s2)
bin_size = 0.01/h #Mpc

def NFW_cumulative_mass(r,rho_0,r_0):
    '''returns NFW cumulative mass integrated to infinity (it converges)'''
    D = r / r_0
    return 4 * np.pi * rho_0 * (r_0**3) * ( np.log(1+D) - ( D / (1+D) ) )

def fit_NFW(radii,cummass,rfit,p0):
    '''uses scipy.optimize.curve_fit to find solution. Takes the rgrid,
    cumulative mass profile, rcrit radius, and parameter guesses [rho_0,r_0]
    '''
    #select indices within rfit (should be the virial radius)
    vir, = np.where(radii <= rfit)	
    NFW_array,pcov = optimize.curve_fit(NFW_cumulative_mass,radii[vir],cummass[vir],\
        p0=p0)
    return NFW_array,pcov

def density(r,mass):
    '''Takes radial grid and cumulative mass profile and returns a density profile in Msol/Mpc^3'''
    dens = np.zeros(r.size)
    for ii in range(dens.size):
        if ii == 0: dens[ii] = mass[ii]/(4/3.0*np.pi*(r[ii]**3-r[ii-1]**3))
        else:
            dens[ii] = (mass[ii]-mass[ii-1])/(4/3.0*np.pi*(r[ii]**3-r[ii-1]**3))
    return dens

def numerical_potential(r,dens):
    '''Integrates the density profile to solve for the potential profile. This is the 2 integral method.
    Returned units are in Mpc^2/s^2
    '''
    deriv1 = dens*r**2
    deriv2 = dens*r
    inner = cumtrapz(deriv1,r)
    outer = -cumtrapz(deriv2[::-1],r[::-1])
    return -4*np.pi*G*(1.0/r[1:-1]*inner[:-1] + outer[::-1][1:])

def phi_N(r,rho_0,r_0):
    '''Returns NFW potential integrated to infinity (it converges). Returned units are in Mpc^2/s^2'''
    D = r / r_0
    return (-4*np.pi*G*rho_0*r_0**2*((np.log(1+D))/D))


def find_potential(gal_data,clus_data,red_z=None,Rvir=None):
    '''Main Program. See main doc for details'''
    
    #####TESTING#####
    try:
        assert gal_data.shape[1] == 6
    except:
        raise Exception, 'Your galaxy array has shape {0}x{1} and needs to be {0}x6. Please reshape!'.format(gal_data.shape[0],gal_data.shape[1])
    try:
        assert clus_data.size == 6
    except:
        raise Exception, 'Oops! You have not fed the function the correct number of cluster values.'
    if red_z == None: raise Exception, 'Please pass the function a cluster redshift!'
    if Rvir == None: raise Exception, 'Please pass the function a cluster virial/critical radius!'
    #################


    GAL_R = np.sqrt((gal_data[:,0]-clus_data[0])**2 + (gal_data[:,1]-clus_data[1])**2 + (gal_data[:,2]-clus_data[2])**2) #3D galaxy radial distances
    GAL_V = np.sqrt((gal_data[:,3]-clus_data[3])**2 + (gal_data[:,4]-clus_data[4])**2 + (gal_data[:,5]-clus_data[5])**2) #3D galaxy peculiar velocities
    
    Mpcbin = np.where(GAL_R <= rmax) #identify indices of particles within 'rmax' of cluster.

    crit = 2.774946e11*h*h*(0.75+0.25/(1+red_z)**3)  #critical density of universe in solar masses/Mpc^3
    av_dens = (0.25/(1+red_z)**3)*crit #average density of universe

    #filter particles to within 'Mpcbin' range
    GAL_R_3Mpc = GAL_R[Mpcbin]
    GAL_V_3Mpc = GAL_V[Mpcbin]

    radial = gal_data[:,3][Mpcbin]*(gal_data[:,0][Mpcbin]/GAL_R_3Mpc) + gal_data[:,4][Mpcbin]*(gal_data[:,1][Mpcbin]/GAL_R_3Mpc) + gal_data[:,5][Mpcbin]*(gal_data[:,2][Mpcbin]/GAL_R_3Mpc) #radial velocity in km/s

    number_of_particles_in_bin,bin_edges = np.histogram(GAL_R_3Mpc,np.arange(0,rmax,bin_size)) #bin particles by radial distance

    GAL_R_edge = (bin_edges[1:]+bin_edges[:-1])/2.0 #the grid of r-values associated with each radial bin

    #### calculate cumulative mass profile (M<r):
    particle_mass = 8.6e8/h #Millennium particle mass

    cumulative_mass_profile = particle_mass * np.cumsum(number_of_particles_in_bin)# solar masses

    '''plot cumulative mass profile M(<r)'''
    plt.plot(GAL_R_edge, cumulative_mass_profile)

    ############ NFW fit #############
    #parameter guesses:
    rho_0_guess = 5e14 # Msol / Mpc3
    r_0_guess = 0.4 #Mpc

    NFW_array,pcov = fit_NFW(GAL_R_edge,cumulative_mass_profile,Rvir,[rho_0_guess,r_0_guess])

    #first element of optimization curve fit output array. units: Msol/[Mpc]^3
    rho_0 = NFW_array[0]  #* u.Msol * u.Mpc**-3

    #second element of optimization curve fit output array. units: [Mpc]
    r_0 = NFW_array[1] #* u.Mpc

    print 'normalization: {0:.3e} Msol/Mpc^3    scale radius: {1:.3f}'.format(rho_0,r_0)

    '''plot cumulative mass profile M(<r) FIT'''
    plt.plot(GAL_R_edge, NFW_cumulative_mass(GAL_R_edge,rho_0,r_0))
    #plt.savefig('path_to_figs/'+str(i)+'profile.png')
    plt.close()
    #plt.show()

    #Now numerically solve for density and potential profiles
    dens_NFW = density(GAL_R_edge,NFW_cumulative_mass(GAL_R_edge,rho_0,r_0)) - av_dens #Msol/Mpc^3
    dens = density(GAL_R_edge,cumulative_mass_profile) - av_dens #Msol/Mpc^3
    potential_numerical = numerical_potential(GAL_R_edge,dens) #Mpc^2/s^2
    potential_NFW_numerical = numerical_potential(GAL_R_edge,dens_NFW) #Mpc^2/s^2


    ############# Escape Velocity profiles  #############

    #escape velocity profile
    v_esc_NFW = np.sqrt(-2 * phi_N(GAL_R_edge,rho_0,r_0))*3.086e19 #km/s
    v_esc_NFW_numerical = np.sqrt(-2 * potential_NFW_numerical)*3.086e19 #km/s
    v_esc_numerical = np.sqrt(-2 * potential_numerical)*3.086e19 #km/s
    
    #hubble param escape calculation
    q = 0.25/2.0 - 0.75
    H2 = (h*100*3.24e-20)**2
    re = (G*cumulative_mass_profile[1:-1]/(-q*H2))**(1/3.0)
    #re = (G*M200[i]*1e10/(-q*H2))**(1/3.0)
    v_esc_hflow = np.sqrt(v_esc_numerical**2 + (q*(h*100)**2*(GAL_R_edge[1:-1]**2)))# - 3*re**2))) #km/s

    #Chris's GM/R +  dm calc
    base = -G*cumulative_mass_profile[GAL_R_edge<=Rvir][-1]/Rvir
    dm = -G*np.append(cumulative_mass_profile[0],cumulative_mass_profile[1:] - cumulative_mass_profile[:-1])/GAL_R_edge
    potential_chris = (base + dm)*(3.086e19)**2 + (q*(h*100)**2*(GAL_R_edge**2)) #km^2/s^2
    v_esc_chris = np.sqrt(-2*potential_chris) #km/s

    #Chris's integral + dm calc
    base = potential_numerical
    dm = dm[1:-1]
    potential_chris_tot = (base + dm)*(3.086e19)**2 + (q*(h*100)**2*(GAL_R_edge[1:-1]**2)) #km^2/s^2
    v_esc_chris_tot = np.sqrt(-2*potential_chris_tot) #km/s

    '''plots'''
    s,ax = plt.subplots(1,figsize=(17,10))
    # plot up particles
    ax.plot(GAL_R_3Mpc,radial,'ko',markersize=1,alpha=0.3)
    
    ax.plot(GAL_R_edge,v_esc_NFW,'b')# NFW escape velocity
    ax.plot(GAL_R_edge,-v_esc_NFW,'b')

    ax.plot(GAL_R_edge[1:-1],v_esc_NFW_numerical,'b--')# numerical NFW escape velocity
    ax.plot(GAL_R_edge[1:-1],-v_esc_NFW_numerical,'b--')
    
    ax.plot(GAL_R_edge[1:-1],v_esc_hflow,color='g')# numerical escape velocity
    ax.plot(GAL_R_edge[1:-1],-v_esc_hflow,color='g')
    
    ax.plot(GAL_R_edge,v_esc_chris,color='orange')# Chris escape velocity
    ax.plot(GAL_R_edge,-v_esc_chris,color='orange')
    
    #format plot
    ax.axvline(Rvir,color='k',ls='--',alpha=0.5)
    ax.set_xlabel('r [Mpc]')
    ax.set_ylabel('$ \sqrt{-2\phi_{N}}$ [km/s]',fontsize=13)
    ax.set_xlim(0,3)#Rvir)
    #plt.savefig('path_to_figs/'+str(i)+'phase.png')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    #####Read in catalog data#####
    redshift,rcrit,M200,veldisp,halox,haloy,haloz,halovx,halovy,halovz = np.loadtxt('/n/Christoq1/giffordw/Millenium/biglosclusters.csv',dtype='float',delimiter=',',usecols=(4,5,6,8,9,10,11,12,13,14),unpack=True)
    fileID = np.loadtxt('/n/Christoq1/MILLENNIUM/100Halos/particles/cmiller.csv',dtype='string',delimiter=',',skiprows=1,usecols=(0,),unpack=True)
    for i in range(1):#fileID.size): #Loop over range(N) clusters
        IDh = fileID[i]
        ID = np.loadtxt('/n/Christoq1/giffordw/Millenium/biglosclusters.csv',dtype='string',delimiter=',',usecols=(0,),unpack=True)
        Rvir = (rcrit[ID==IDh]/(1+redshift[ID==IDh]))[0] / h # Mpc
        fits = pyfits.open('/n/Christoq1/MILLENNIUM/100Halos/particles/t'+str(i)+'_cmiller.dat.fits') #open particle fits file
        data = fits[1].data
        gal_x = data.field('PPX')/h
        gal_y = data.field('PPY')/h
        gal_z = data.field('PPZ')/h
        gal_vx = data.field('VVX')
        gal_vy = data.field('VVY')
        gal_vz = data.field('VVZ')
        find_potential(np.column_stack((gal_x,gal_y,gal_z,gal_vx,gal_vy,gal_vz)),np.array([0.0,0.0,0.0,halovx[ID==IDh],halovy[ID==IDh],halovz[ID==IDh]]),redshift[ID==IDh][0],Rvir)
        print 'DONE WITH CLUSTER {0}'.format(i)
