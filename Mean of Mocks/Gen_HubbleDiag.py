import numpy as np
from scipy.integrate import quad
from ModelMerger import *
from Gen_MCMC import *
from Gen_Emcee import emcee_run

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

def get_legend_label(x, params):
    if x == 'FLCDM':
        return r'F$\Lambda$: $\Omega_m = %0.2f $' % (params[0])
    if x == 'LCDM':
        return r'$\Lambda$: $\Omega_m = %0.2f $, $\Omega_{\Lambda} = %0.2f $' % (params[0], params[1] )
    if x == 'FwCDM':
        return r'F$\omega$: $\Omega_m = %0.2f $, $\omega = %0.2f $' % (params[0], params[1] )
    if x == 'wCDM':
        return r'$\omega$: $\Omega_m = %0.2f $, $\Omega_{\Lambda} = %0.2f $, $\omega = %0.2f $' % (params[0], params[1], params[2] )
    if x == 'Fwa':
        return r'F$\omega$(a): $\Omega_m = %0.2f $, $\omega_0 = %0.2f $, $\omega_a = %0.2f $' % (params[0], params[1], params[2] )
    if x == 'FCa':
        return r'FCa: $\Omega_m = %0.2f $, $q = %0.2f $,$n = %0.2f $' % (params[0], params[1], params[2] )
    if x == 'Chap':
        return r'SCh: $A = %0.2f $ $\Omega_K = %0.2f $' % (params[0], params[1] )
    if x == 'FGChap':
        return r'FGCh: $A = %0.2f $ $\alpha = %0.2f $' % (params[0], params[1])
    if x == 'GChap':
        return r'GCh: $A = %0.2f $ $\alpha = %0.2f $ $\Omega_K = %0.2f $' % (params[0], params[1], params[2])
    if x == 'DGP':
        return r'DGP: $\Omega_{rc} = %0.2f $, $\Omega_K = %0.2f $' % (params[0], params[1])
    if x == 'IDE':
        return r'IDE: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $Q = %0.2f $' % (params[0], params[1], params[2], params[3])

# List of begining values for each model to feed to the MCMC sampler - Dont select 0 as starting position
def get_begin_values(x):
    if x == 'FLCDM':
        return [0.3]
    if x == 'LCDM':
        return [0.3, 0.7]
    if x == 'FwCDM':
        return [0.3, -1]
    if x == 'wCDM':
        return [0.3, 0.7, -1]
    if x == 'Fwa':
        return [0.3, -1.1, 0.8]
    if x == 'FCa':
        return [0.3, 1, 0.01]
    if x == 'Chap':
        return [0.8, 0.2]
    if x == 'FGChap':
        return [0.7, 0.2]
    if x == 'GChap':
        return [0.01, 0.7, 0.03]
    if x == 'DGP':
        return [0.13, 0.02]
    if x == 'IDE':
        return [0.3, 0.7, -1, -0.02]


# ---------- Import data ---------- #
# Import data
data = np.genfromtxt("DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
zz = data['zCMB']
mu = data['MU']
error = data['MUERR']
cov_arr = np.genfromtxt("COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)
cov2 = np.diagonal(cov) 
mu_diag = np.diag(error)**2
mu_error = mu_diag+cov
errorbar = error


# List of parameters, these will need to be updated from MCMC/Grid later - params = [om,ol,w,q,n,A,a,rc]

params_FLCDM = [zz, 0.3]
params_LCDM = [zz, 0.3, 0.7]
params_FwCDM = [zz, 0.3, -1]
params_wCDM = [zz, 0.3, 0.7, -1]
params_Fwa = [zz, 0.3, -1.1, 0.8]
params_FCa = [zz, 0.3, 1, 0]
params_Chap = [zz, 0.8, 0.2]
params_FGChap = [zz, 0.7, 0.2]
params_GChap = [zz, 0, 0.7, 0.03]
params_DGP = [zz, 0.13, 0.02]

#params_all = [params_FLCDM,params_LCDM,params_FwCDM,params_wCDM,params_Fwa,params_FCa,params_Chap,params_FGChap,params_GChap,params_DGP]
# List of models to loop through from ModelMerger file
models = [FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]

# Milne Model for scaling figure
milne = LCDM(zz,[0,0]) 

# Get Best fit params using MCMC for each model
params_all = []
def get_bestfit(models):     # get_labels dont have marginalised value in but MCMC uses it
    for i, model in enumerate(models):
        nsamples = int(1e5)
        params_begin = np.array(get_begin_values(model.__name__))
        proposal = []
        params_begin1 = []
        for i, begin_param in enumerate(params_begin):
            proposal.append(abs(begin_param)*0.06)
            params_begin1.append(begin_param)
        samples = emcee_run(zz, mu, mu_error, params_begin1, nsamples, proposal, model)
        label = get_label(model.__name__)
        params_all.append(get_params(samples, label, model.__name__))

get_bestfit(models)

# Plot each model
for i, (params, model) in enumerate(zip(params_all, models)):
    hubble = model(zz, params) 
    labels = get_legend_label(model.__name__ , params)
    plt.plot(zz, hubble-milne-(hubble[0]-milne[0]), markersize=2, label = labels)

# Plot the data points
plt.errorbar(zz,mu-milne-(mu[0]-milne[0]),yerr=errorbar,fmt='.',elinewidth=0.7,markersize=4, color='k' )


# Figure properties
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.xlim(0,1.6)
#plt.ylim(-0.6,0.6)
plt.xlabel('Redshift, z')
plt.ylabel(r'$\Delta$ Distance Modulus (Mag)')
plt.savefig("hubble_diagram.png",bbox_inches='tight')
plt.show()


