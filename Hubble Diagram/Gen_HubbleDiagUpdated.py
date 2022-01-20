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
# Import data - Maria
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
models = [FLCDM, LCDM, GLT]
#[FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]

# Milne Model for scaling figure
milne = LCDM(zz,[0,0]) 
params_all = []
# Get Best fit params using MCMC for each model
def get_bestfit(models, zz1, mu1, mu_error1, like):     # get_labels dont have marginalised value in but MCMC uses it
    for i, model in enumerate(models):
        nsamples = int(1e5)
        label, begin, legend = get_info(model.__name__)
        params_begin = begin
        proposal = []
        params_begin1 = []
        for i, begin_param in enumerate(params_begin):
            proposal.append(abs(begin_param)*0.06)
            params_begin1.append(begin_param)
        samples, samples1, pdf = emcee_run(zz1, mu1, mu_error1, params_begin1, nsamples, proposal, model, like)
        params_all.append(get_param(samples, label, model.__name__, 1)) # 1 = plot contour, 0 = dont plot contour
        param = get_param(samples, label, model.__name__, 1)
        np.savetxt('Hubble Diagram/Chains_OUTPUT/%s_CHAIN_%s.txt' % (model.__name__, like) , samples, fmt="%10.4f")
        np.savetxt('Hubble Diagram/Chains_OUTPUT/%s_POSTERIOR_%s.txt' % (model.__name__, like) , pdf, fmt="%10.4f")
        np.savetxt('Hubble Diagram/Chains_OUTPUT/%s_PARAMS_%s.txt' % (model.__name__, like) , param, fmt="%10.4f")


LL = 'loglike'
CLL = 'covloglike'
get_bestfit(models, zz, mu, mu_error, CLL)




if __name__ == "__main__":
    # Plot each model

    # Figure properties
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # - outside plot 
    from matplotlib import rcParams, rc
    rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, axs = plt.subplots(2, 1, sharex=True,figsize=(12,7),gridspec_kw = {'height_ratios':[1, 1]})
    fig.subplots_adjust(hspace=0)
    axs[0].set_xlabel('Redshift, z', fontsize=22)
    axs[0].set_ylabel(r'$\Delta$ Distance Modulus, $\mu$', fontsize=22)

    # Plot the data points
    offset1 = (np.sum(mu - (milne)))/len(zz)
    axs[0].errorbar(zz,mu-(offset1)-milne,yerr=errorbar,fmt='.',elinewidth=2,markersize=10, color='k')
    axs[0].axhline(0,color = 'k', linewidth=1)
    axs[0].tick_params(axis='both', labelsize=18)
    #plt.errorbar(zz,mu-(offset1)-milne,yerr=errorbar,fmt='.',elinewidth=0.7,markersize=4, color='k', alpha=0.3 )
    colours = ['k','b','r']
    linestylE = ['--',':','-.']
    for i, (params, model) in enumerate(zip(params_all, models)):
        hubble = model(zz, params) 
        label, begin, legend = get_info(model.__name__, *params)
        if model.__name__ == 'FLCDM':
            reference = model(zz, params)
        offset = (np.sum(hubble - (mu-offset1)))/len(zz)
        axs[0].plot(zz, hubble-offset-milne, markersize=2, label = legend, color=colours[i], linestyle=linestylE[i])
        #plt.plot(zz, hubble-offset-milne, markersize=2, label = legend)  # delete above line and un comment this if dont work
        #print(np.sum(hubble-offset-milne)/len(zz))

    #zzfull = np.linspace(0,1.1,50)
    # delete whole thing if dont work (below)
    offset2 = (np.sum(mu-reference))/len(zz)
    for i, (params, model) in enumerate(zip(params_all, models)):
        hubble = model(zz, params) 
        label, begin, legend = get_info(model.__name__, *params)
        offset3 = (np.sum(hubble - (mu-offset2)))/len(zz)
        axs[1].plot(zz, hubble - offset3 - reference , markersize=2, label = legend, color=colours[i], linestyle=linestylE[i])
        #print(np.sum(hubble-offset-milne)/len(zz))


    axs[1].errorbar(zz,mu-offset2-reference,yerr=errorbar,fmt='.',elinewidth=2,markersize=10, color='k' )
    axs[1].set_xlabel('Redshift, z', fontsize=22)
    axs[1].set_ylabel(r'$\mu$ Residual', fontsize=22)
    axs[1].set_yticks([-0.05,0,0.05])
    axs[1].tick_params(axis='both', labelsize=18)
 

    #Back to norm
    plt.legend(loc='lower left', ncol=1,frameon=False,fontsize=14)
    #plt.xlabel('Redshift, z', fontsize=20)
    #plt.ylabel(r'$\Delta$ Distance Modulus (Mag)', fontsize=20)
    plt.savefig("hubble_diagram.png",bbox_inches='tight')
    plt.show()
