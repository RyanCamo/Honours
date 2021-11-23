import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from Gen_Emcee import *

# Fits and then plots a normalised Hubble diagram of some data

# To add models:
    # Add model to:     ModelMerger.py, 
    # Update:           get_legend_label
    # Update:           get_label
    # Update:           get_begin_values

#### Paramater labels ####
# om = \Omega_M             --- matter density
# ol = \Omega_{\Lambda}     --- cosmological constant density
# w  = \omega               --- equation of state parameter where w = -1 is consistent with the cosmological constant
# w0 = 
# wa = 
# q  = q                    --- q for cardassian models
# n  = n                    --- n for cardassian models
# A = A                     --- A for Chaplygin models
# a = \alpha                --- \alpha for Chaplygin models
# rc = \Omega_rc            --- \Omega_rc for DGP models


# ---------- Import data ---------- #
# Import data
data = np.genfromtxt("Hubble Diagram/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
zz = data['zCMB']
mu = data['MU']
error = data['MUERR']
cov_arr = np.genfromtxt("Hubble Diagram/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)
cov2 = np.diagonal(cov) 
mu_diag = np.diag(error)**2
mu_error = mu_diag+cov
errorbar = error



# List of models to loop through from ModelMerger file
models = [FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]

# Milne Model for scaling figure
milne = LCDM(zz,[0,0]) 
params_all = []
# Get Best fit params using MCMC for each model
def get_bestfit(models, zz1, mu1, mu_error1):     # get_labels dont have marginalised value in but MCMC uses it
    for i, model in enumerate(models):
        nsamples = int(1e5)
        label, begin, legend = get_info(model.__name__)
        params_begin = begin
        proposal = []
        params_begin1 = []
        for i, begin_param in enumerate(params_begin):
            proposal.append(abs(begin_param)*0.06)
            params_begin1.append(begin_param)
        samples, samples1, pdf = emcee_run(zz1, mu1, mu_error1, params_begin1, nsamples, proposal, model)
        params_all.append(get_param(samples, label, model.__name__))


get_bestfit(models, zz, mu, mu_error)

# Plot the data points
offset1 = (np.sum(mu - (milne)))/len(zz)
plt.errorbar(zz,mu-(offset1)-milne,yerr=errorbar,fmt='.',elinewidth=0.7,markersize=4, color='k' )

if __name__ == "__main__":
    # Plot each model
    for i, (params, model) in enumerate(zip(params_all, models)):
        hubble = model(zz, params) 
        label, begin, legend = get_info(model.__name__, *params)
        offset = (np.sum(hubble - (mu-offset1)))/len(zz)
        plt.plot(zz, hubble-offset-milne, markersize=2, label = legend)
        print(np.sum(hubble-offset-milne)/len(zz))

 


    # Figure properties
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # - outside plot
    plt.legend(loc='lower left', ncol=2, fontsize=12)
    plt.xlabel('Redshift, z', fontsize=20)
    plt.ylabel(r'$\Delta$ Distance Modulus (Mag)', fontsize=20)
    plt.savefig("hubble_diagram.png",bbox_inches='tight')
    plt.show()
