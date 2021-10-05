# All the non-standard models ive looked into

import os
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from numpy.random import multivariate_normal, normal, seed
from chainconsumer import ChainConsumer
import emcee
seed(0)

#### Paramater labels ####
# om = \Omega_M             --- matter density
# ol = \Omega_{\Lambda}     --- cosmological constant density
# w  = \omega               --- equation of state parameter where w = -1 is consistent with the cosmological constant
# q  = q                    --- q for cardassian models
# n  = n                    --- n for cardassian models
# A = A                     --- A for Chaplygin models
# a = \alpha                --- \alpha for Chaplygin models
# rc = \Omega_rc            --- \Omega_rc for DGP models

#### Define cosntants ####
H0 = 70
c_H0 = 299792.458 / H0  #Speed of light divided by Hubble's constant in: (km/s)/(km/s/Mpc) = Mpc 
model = 4 # selecting what model to use (numbers below)

#### Models ####

# 1) Flat Cosmological Constant with 1x paramater, \Omega_M - DONE
def FLCDM_Hz_inverse(z,om, ol):
    #ol = 1-om
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz
    
def FLCDM(zs, parameters):
    om, = parameters
    ol = 1 - om
    x = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$"]
    return dist_mod


# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} - DONE
def LCDM_Hz_inverse(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz

def LCDM(zs, parameters):
    om, ol = parameters
    ok = 1.0 - om - ol
    x = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\Omega_{\Lambda}$"]
    return dist_mod


# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega - DONE
def FwCDM_Hz_inverse(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz

def FwCDM(zs, parameters):
    om, w = parameters
    ol = 1 - om
    x = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\omega$"]
    return dist_mod


# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega - DONE
def wCDM_Hz_inverse(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz
    
def wCDM(zs, parameters):
    om, ol, w = parameters
    ok = 1.0 - om - ol
    x = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\Omega_{\Lambda}$",r"$\omega$"]
    return dist_mod


# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwa_Hz_inverse(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return 1.0 / Hz

def Fwa(zs, parameters):
    om, w0, wa = parameters
    x = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$w_0$","$w_a$"]
    return dist_mod

def Fwa_Hz_inverse1(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return 1.0 / Hz

def Fwa1(zs, parameters):
    om, w0, wa, m = parameters
    x = np.array([quad(Fwa_Hz_inverse1, 0, z, args=(om, w0, wa))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist) + m
    label = ["$\Omega_m$","$w_0$","$w_a$"]
    return dist_mod


# 6) Cardassian with 3x parameters, \Omega_M, q and n
def FCa_Hz_inverse(z, om, q ,n ):
    Hz = np.sqrt(
        (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
    return 1.0 / Hz

def FCa(zs, parameters):
    om, q, n = parameters
    x = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$q$","$n$"]
    return dist_mod


# 7) Flat Chaplygin 1x parameters, A - Not doing


# 8) Chaplygin 2x parameters, A and \Omega_K
def Chap_Hz_inverse(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return 1.0 / Hz

def Chap(zs, parameters):
    A, ok = parameters
    x = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$A$","$\Omega_k$"]
    return dist_mod


# 9) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap_Hz_inverse(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return 1.0 / Hz

def FGChap(zs, parameters):
    A, a = parameters
    x = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$A$", r"$\alpha$"]
    return dist_mod


# 10) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap_Hz_inverse(z, ok, A ,a):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def GChap(zs, parameters):
    ok, A, a = parameters
    x = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_K$","$A$",r"$\alpha$"]
    return dist_mod


# 11) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP_Hz_inverse(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return 1.0 / Hz

def DGP(zs, parameters):
    rc, ok = parameters
    x = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{rc}$", r"$\Omega_K$"]
    return dist_mod

# 12) Flat DGP 1x parameters, \Omega_rc - Not doing

# 13) LTB Gauss
# 14) LTB Sharp
#(((1/3)*Q*ol*(1+z)**(-3*(1+w)-Q)) + cdm*(1+z)**3)

# Theres were kinda close
#ol*((np.exp(-Q*ol/(1+z)))*((1+z)**(3*(1+w)))))
#cdm*((np.exp(Q*ol/(1+z)))*((1+z)**(3)))
# 14) IDE
def IDE_Hz_inverse(z, cdm, ol, w, Q):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(ok*(1+z)**(2) + ((((1+3*np.log((1/(1+z))))**(-1))*Q*cdm*((3*(1-w)+Q)**(-1))*(1+z)**(3*(1-w)+Q))) + (cdm*(1+z)**(3*(1-w)+Q)))
    #(((1/3)*Q*ol*(1+z)**(-3*(1+w)-Q)) + (cdm-(1/3)*Q)*(1+z)**3)) + ol*(1+z)**(-3*(1+w)-Q) - not so close
    return 1.0 / Hz

def IDE(zs, parameters):
    cdm, ol, w, Q = parameters
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse, 0, z, args=(cdm, ol, w, Q))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$Q$"]
    return dist_mod

### Other Functions ####

# Likelihood function
def log_likelihood_LCDM(theta, zz, mu, mu_err):
    m, b = theta
    model = LCDM(zz,m,b)
    delta = model - mu
    chit2 = np.sum(delta**2 / mu_err**2)
    B = np.sum(delta/mu_err**2)
    C = np.sum(1/mu_err**2)
    chi2 = chit2 - (B**2 / C) + np.log(C/2* np.pi)
    # original log_likelihood ---->    -0.5 * np.sum((mu - model) ** 2 /mu_err**2) 
    return -0.5*chi2


# Getlabel function
def get_label(x):
    if x == 'FLCDM':
        return [r"$Omega_m$"]
    if x == 'LCDM':
        return [r"$\Omega_m$",r"$\Omega_{\Lambda}$"]
    if x == 'FwCDM':
        return [r"$\Omega_m$",r"$\omega$"]
    if x == 'wCDM':
        return [r"$\Omega_m$",r"$\Omega_{\Lambda}$",r"$\omega$"]
    if x == 'Fwa':
        return [r"$\Omega_m$",r"$w_0$",r"$w_a$"]
    if x == 'FCa':
        return [r"$\Omega_m$",r"$q$","$n$"]
    if x == 7:
        return ["A"]
    if x == 'Chap':
        return [r"$A$",r"$\Omega_k$"]
    if x == 'FGChap':
        return [r"$A$", r"$\alpha$"]
    if x == 'GChap':
        return [r"$\Omega_K$",r"$A$",r"$\alpha$"]
    if x == 'DGP':
        return [r"$\Omega_{rc}$", r"$\Omega_K$"]
    if x == 12:
        return [r"\Omega_{rc}"]
    if x == 'IDE':
        return [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$Q$"]
    if x == 'Fwa1':
        return [r"$\Omega_m$",r"$w_0$",r"$w_a$",r"$M$"]
