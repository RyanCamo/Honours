import numpy as np


t0 = 2/3 # H_0 ^-1 is factored out - t0 = 2/(3*H_0)

def get_OMK(r, om_out, om_in, r0):
    # specific Omega_m profile and associated value of Omega_k
    om = om_out + (om_in-om_out)*np.exp(-(r/r0)**2) # CHANGE THIS FOR DIFF FUNCTIONS
    ok = 1-om
    return om, ok

def get_H_0_ang(om, ok):
    if ok == 0:
        return 2/(3*(t0))
    if ok > 0:
        return (np.sqrt(ok) - om*np.arcsinh(np.sqrt(ok/om)))/(t0*ok**(2/3))


def tau(t, r, om_out, om_in, r0):
    om, ok = get_OMK(r, om_out, om_in, r0)
    H_0_ang = get_H_0_ang(om,ok)
    t_a = (1/H_0_ang) *

