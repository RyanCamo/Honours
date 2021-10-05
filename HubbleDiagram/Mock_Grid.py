import os
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

def Hz_inverse(z, om, ol):
    """ Calculate 1/H(z). Will integrate this function. """
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz


def dist_mod(zs, om, ol):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ol
    x = np.array([quad(Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
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
    return dist_mod

def cov_log_likelihood(parameters, zz, mu, cov):
    om, ol, m= parameters
    model = dist_mod(zz,om,ol) + m
    delta = np.array([model - mu])
    inv_cov = np.linalg.inv(cov)
    #print(inv_cov.shape)
    deltaT = np.transpose(delta)
    #print(deltaT.shape)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov @ inv_cov)
    check = delta @ inv_cov @ deltaT - B**2/C + np.log(C/(2*np.pi))
    #print(check)

    chi2 = np.sum(delta @ inv_cov**2 @ deltaT- B**2/C + np.log(C/(2*np.pi)))  # - B**2/C + np.log(C/(2*np.pi))
    return -0.5*chi2

def log_likelihood(model, zz, mu, mu_err): 
    delta = model - mu
    chit2 = np.sum(delta**2 / mu_err**2)
    B = np.sum(delta/mu_err**2)
    C = np.sum(1/mu_err**2)
    chi2 = chit2 - (B**2 / C) + np.log(C/(2* np.pi))
    #chi2 =  np.sum((mu - model) ** 2 /mu_err**2) 
    # original log_likelihood ---->    -0.5 * np.sum((mu - model) ** 2 /mu_err**2) 
    return chi2
    
if __name__ == "__main__":

    # ---------- Import data ---------- #
    # Import data
    data = np.genfromtxt("HubbleDiagram/FITOPT000_MUOPT006.M0DIF",names=True,comments='#',dtype=None, skip_header=14)
    zz = data['z']
    mu_dif = data['MUDIF']
    mu = data['MUDIF'] + data['MUREF']
    mu_error = data['MUDIFERR'] 
    cov_arr = np.genfromtxt("HubbleDiagram/FITOPT000_MUOPT006.COV",comments='#',dtype=None, skip_header=1)
    cov = cov_arr.reshape(20,20)
    cov2 = np.diagonal(cov)
    mu_error1 = np.diag(mu_error)**2
    print(mu_error1.shape)

    # Define cosntants
    H0 = 70.
    c_H0 = 299792.458 / H0  #Speed of light divided by Hubble's constant in: (km/s)/(km/s/Mpc) = Mpc 

    # ---------- Set up fitting ranges ---------------------------
    n = 20                           # Increase this for a finer grid
    oms = np.linspace(0, 0.5, n)     # Array of matter densities
    ols = np.linspace(0, 1.5, n)     # Array of cosmological constant values
    chi2 = np.ones((n, n)) * np.inf  # Array to hold our chi2 values, set initially to super large values

    n_marg = 200                            # Number of steps in marginalisation
    mscr_guess = 5.0 * np.log10(c_H0) + 25  # Initial guess for best mscr
    mscr = np.linspace(mscr_guess - 0.5, mscr_guess + 0.5, n_marg)  # Array of mscr values to marginalise over
    mscr_used = np.zeros((n, n))            # Array to hold the best fit mscr value for each om, ol combination

    # ---------- Do the fit ---------------------------
    saved_output_filename = "saved_grid_%d.txt" % n
    saved_mscr_filename = "mscr_record_%d.txt" % n

    if os.path.exists(saved_output_filename):  # Load the last run with n grid if we can find it
        print("Loading saved data. Delete %s if you want to regenerate the points\n" % saved_output_filename)
        chi2      = np.loadtxt(saved_output_filename)
        mscr_used = np.loadtxt(saved_mscr_filename  )
    else:
        for i, om in enumerate(oms):             # loop through matter densities
            for j, ol in enumerate(ols):         # loop through cosmological constant densities
                mu_model = dist_mod(zz, om, ol)  # calculate the distance modulus vs redshift for that model 
                for k, m in enumerate(mscr):     # loop over the arbitrary vertical scale, so we can marginalise
                    mu_model_norm = mu_model  # normalise the model by the mscr value chosen
                    #chi2_test = cov_log_likelihood([om, ol, m], zz , mu, cov+mu_error1)
                    chi2_test = log_likelihood(mu_model_norm, zz , mu, mu_error)
                    #chi2_test = np.sum((mu_model_norm - mu) ** 2 / mu_error**2) # test how good a fit that new scaled model is

            
                    if chi2_test < chi2[i, j]:   # if the scaled model is better than the original one...
                        chi2[i, j] = chi2_test   # ...then save it as the best model so far
                        mscr_used[i, j] = 0   # ...and save the value of mscr you used
            print("Done %d out of %d" % (i+1, oms.size))
        #np.savetxt(saved_output_filename, chi2, fmt="%10.4f")
        #np.savetxt(saved_mscr_filename,mscr_used)

    likelihood = np.exp(-0.5 * (chi2-np.amin(chi2)))  # convert the chi^2 to a likelihood (np.amin(chi2) calculates the minimum of the chi^2 array)
    chi2_reduced = chi2 / (len(data)-2)               # calculate the reduced chi^2, i.e. chi^2 per degree of freedom, where dof = number of data points minus number of parameters being fitted 

    # Calculate the best fit values (where chi2 is minimum)
    indbest = np.argmin(chi2)                 # Gives index of best fit but where the indices are just a single number
    ibest   = np.unravel_index(indbest,[n,n]) # Converts the best fit index to the 2d version (i,j)
    print( 'Best fit values are (om,ol)=(%s,%s)'%( oms[ibest[0]], ols[ibest[1]] ) )
    print( 'Reduced chi^2 for the best fit is %0.2f'%chi2_reduced[ibest[0],ibest[1]] )
    print( 'Mscr used is %3.3f.  After testing over range %3.3f to %3.3f.'%( mscr_used[ibest[0],ibest[1]], mscr.min(), mscr.max() ) )
    if (mscr_used[ibest[0],ibest[1]]<=mscr.min() or mscr_used[ibest[0],ibest[1]]>=mscr.max()):  # Note this test will not work if you change mscr and reload old data with a different mscr
        print(  '!!WARNING!!  Mscr is outside the possible range.  Adjust ranges so that mscr lies within the possible range or you will have a biased answer. ')

    # Plot contours of 1, 2, and 3 sigma
    plt.contour(oms,ols,np.transpose(chi2-np.amin(chi2)),cmap="winter",**{'levels':[2.30,6.18,11.83]})
    plt.xlabel("$\Omega_m$", fontsize=12)
    plt.ylabel("$\Omega_\Lambda$", fontsize=12)
    plt.savefig("contours.png", bbox_inches="tight", transparent=True)
    plt.show()


    Best_Fit = dist_mod(zz,oms[ibest[0]],ols[ibest[1]]) + mscr_used[ibest[0],ibest[1]]
    plt.errorbar(zz,mu,yerr=mu_error,fmt='.',elinewidth=0.7,markersize=4 ) #,label=r'$\Lambda$CDM Mock Data')
    plt.plot(zz,Best_Fit, 'k--',  markersize=2)
    plt.xlabel('Redshift, z')
    plt.ylabel(r'Distance Modulus (Mag)')
    plt.show()

    plt.close()