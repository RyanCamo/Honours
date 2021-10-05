import os
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

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
    #mu_error = data['MUDIFERR'] 
    cov_arr = np.genfromtxt("HubbleDiagram/FITOPT000_MUOPT006.COV",comments='#',dtype=None, skip_header=1)
    cov = cov_arr.reshape(20,20)
    mu_error = np.diagonal(cov)**2
    mu_error1 = np.diag(mu_error)**2
    print(mu_error1.shape)

    # Define cosntants
    H0 = 70.
    c_H0 = 299792.458 / H0  #Speed of light divided by Hubble's constant in: (km/s)/(km/s/Mpc) = Mpc 

    # ---------- Set up fitting ranges ---------------------------
    n = 30                           # Increase this for a finer grid
    oms = np.linspace(0, 0.5, n)     # Array of matter densities
    ols = np.linspace(-2, -1, n)     # Array of cosmological constant values
    olp = np.linspace(0.1,0.9,n)
    chi2 = np.ones((n, n, n)) * np.inf  # Array to hold our chi2 values, set initially to super large values

    n_marg = 200                            # Number of steps in marginalisation
    mscr_guess = 5.0 * np.log10(c_H0) + 25  # Initial guess for best mscr
    mscr = np.linspace(mscr_guess - 0.5, mscr_guess + 0.5, n_marg)  # Array of mscr values to marginalise over
    mscr_used = np.zeros((n, n, n))            # Array to hold the best fit mscr value for each om, ol combination

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
                for k, op in enumerate(olp):  
                    mu_model = Fwa(zz, [om,ol,op])  # calculate the distance modulus vs redshift for that model 
                    for l, m in enumerate(mscr):     # loop over the arbitrary vertical scale, so we can marginalise
                        mu_model_norm = mu_model + m # normalise the model by the mscr value chosen
                        #chi2_test = cov_log_likelihood([om, ol, m], zz , mu, cov+mu_error1)
                        #chi2_test = log_likelihood(mu_model_norm, zz , mu, mu_error)
                        chi2_test = np.sum((mu_model_norm - mu) ** 2 / mu_error**2) # test how good a fit that new scaled model is

                
                    if chi2_test < chi2[i, j, k]:   # if the scaled model is better than the original one...
                        chi2[i, j, k] = chi2_test   # ...then save it as the best model so far
                        mscr_used[i, j, k] = m   # ...and save the value of mscr you used
            print("Done %d out of %d" % (i+1, oms.size))
        #np.savetxt(saved_output_filename, chi2, fmt="%10.4f")
        #np.savetxt(saved_mscr_filename,mscr_used)

    likelihood = np.exp(-0.5 * (chi2-np.amin(chi2)))  # convert the chi^2 to a likelihood (np.amin(chi2) calculates the minimum of the chi^2 array)
    chi2_reduced = chi2 / (len(data)-3)               # calculate the reduced chi^2, i.e. chi^2 per degree of freedom, where dof = number of data points minus number of parameters being fitted 

    # Calculate the best fit values (where chi2 is minimum)
    indbest = np.argmin(chi2)                 # Gives index of best fit but where the indices are just a single number
    ibest   = np.unravel_index(indbest,[n,n,n]) # Converts the best fit index to the 2d version (i,j)
    print( 'Best fit values are (om,ol,op)=(%s,%s,%s)'%( oms[ibest[0]], ols[ibest[1]], olp[ibest[2]] ) )
    print( 'Reduced chi^2 for the best fit is %0.2f'%chi2_reduced[ibest[0],ibest[1], ibest[2]] )
    print( 'Mscr used is %3.3f.  After testing over range %3.3f to %3.3f.'%( mscr_used[ibest[0],ibest[1], ibest[1]], mscr.min(), mscr.max() ) )
    if (mscr_used[ibest[0],ibest[1]]<=mscr.min() or mscr_used[ibest[0],ibest[1]]>=mscr.max()):  # Note this test will not work if you change mscr and reload old data with a different mscr
        print(  '!!WARNING!!  Mscr is outside the possible range.  Adjust ranges so that mscr lies within the possible range or you will have a biased answer. ')

    # Plot contours of 1, 2, and 3 sigma
    plt.contour(oms,ols,np.transpose(chi2-np.amin(chi2)),cmap="winter",**{'levels':[2.30,6.18,11.83]})
    plt.xlabel("$\Omega_m$", fontsize=12)
    plt.ylabel("$\Omega_\Lambda$", fontsize=12)
    plt.savefig("contours.png", bbox_inches="tight", transparent=True)
    plt.show()


    Best_Fit = Fwa(zz,oms[ibest[0]],ols[ibest[1]], olp[ibest[2]]) + mscr_used[ibest[0],ibest[1], ibest[2]]
    plt.errorbar(zz,mu,yerr=mu_error,fmt='.',elinewidth=0.7,markersize=4 ) #,label=r'$\Lambda$CDM Mock Data')
    plt.plot(zz,Best_Fit, 'k--',  markersize=2)
    plt.xlabel('Redshift, z')
    plt.ylabel(r'Distance Modulus (Mag)')
    plt.show()

    plt.close()