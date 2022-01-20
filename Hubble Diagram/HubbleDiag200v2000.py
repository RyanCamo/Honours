import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from Gen_Emcee import *
import statistics

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
#data = np.genfromtxt("Hubble Diagram/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
#z = data['zCMB']
#mu = data['MU']
#error = data['MUERR']
#cov_arr = np.genfromtxt("Hubble Diagram/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
#cov = cov_arr.reshape(20,20)
#cov2 = np.diagonal(cov) 
#mu_diag = np.diag(error)**2
#mu_error = mu_diag+cov
#errorbar = error


# Data - Mocks #
#data200 = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/MockData/2000v200/FLCDM200_0_.txt")
#data275 = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/MockData/2000v200/FLCDM275_0_.txt")
#data2750 = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/MockData/2000v200/FLCDM2750_0_.txt")
#data275test = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/MockData/2000v200/FLCDM275_0_test.txt")
#data2000 = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/MockData/2000v200/FLCDM2000_0.txt")

# Unbinned
DES3YR_UNBIN = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/DES3YR_UNBINNED.txt")
DES3YR_COV_UNBIN = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/COV_DES3YR_UNBINNED.txt")
ZZ_UNBIN = DES3YR_UNBIN[:,1]
MU_ERR_UNBIN_ARR = DES3YR_UNBIN[:,5]
MU_ERR_UNBIN_DIAG = np.diag(MU_ERR_UNBIN_ARR)**2
MU_ERR_UNBIN = MU_ERR_UNBIN_DIAG + DES3YR_COV_UNBIN
MU_UNBIN = DES3YR_UNBIN[:,4]

#Binned 
DES3YR_BIN = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/DES3YR_BINNED.txt")
DES3YR_COV_BIN = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/COV_DES3YR_BINNED.txt")
ZZ_BIN = DES3YR_BIN[:,1]
MU_ERR_BIN_ARR = DES3YR_BIN[:,5]
MU_ERR_BIN_DIAG = np.diag(MU_ERR_BIN_ARR)**2
MU_ERR_BIN = MU_ERR_BIN_DIAG + DES3YR_COV_BIN
MU_BIN = DES3YR_BIN[:,4]



#zz2750 = np.tile(datatxt[:,1], 10)
#zz2000 = np.logspace(-2,0.2,2000)
#zz200 = np.logspace(-2,0.2,200)
#zz = [zz200, zz2000]
#mu = [data200, data2000]
#mu_error2000 = np.linspace(0.1,0.1,2000)
#mu_error200 = np.linspace(0.1,0.1,200)
#mu_error275 = np.linspace(0.1,0.1,275)
#mu_error2750 = np.linspace(0.1,0.1,2750)
#mu_error = [mu_error200, mu_error2000]
#errorbar200 = mu_error200
#errorbar2000 = mu_error2000
#errorbar = [errorbar200, errorbar2000]

data = np.genfromtxt("Hubble Diagram/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
zzmock = data['zCMB']
mumock = data['MU']
errormock = data['MUERR']
cov_arr = np.genfromtxt("Hubble Diagram/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)
cov2 = np.diagonal(cov) 
mu_diag = np.diag(errormock)**2
mu_errormock = cov + mu_diag
errorbar = errormock


# List of models to loop through from ModelMerger file
models = [IDE1]
#[FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]

# Milne Model for scaling figure
#milne = LCDM(zz,[0,0]) 
# Get Best fit params using MCMC for each model
def get_bestfit(models, zz1, mu1, mu_error1, like):     # 'Like' Parameter tells the function to use loglikelihood or cov
    params_all = []
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
        params_all.append(get_param(samples, label, model.__name__, 0)) # 1 = plot contour, 0 = dont plot contour
    return params_all, samples, samples1, pdf, label

LL = 'loglike'
CLL = 'covloglike'
#params_all200, samples200, samples1200, pdf200, label = get_bestfit(models, zz[0], mu[0], mu_error[0], LL)
#params_all275, samples275, samples1275, pdf275, label = get_bestfit(models, zz275, data275, mu_error275, LL)
#params_all2750, samples2750, samples12750, pdf2750, label = get_bestfit(models, zz2750, data2750, mu_error2750, LL)
###params_UNBIN, samples_UNBIN, samples1_UNBIN, pdf_UNBIN, label = get_bestfit(models, ZZ_UNBIN, MU_UNBIN, MU_ERR_UNBIN, CLL)
params_BIN, samples_BIN, samples1_BIN, pdf_BIN, label = get_bestfit(models, ZZ_BIN, MU_BIN, MU_ERR_BIN, CLL)
#params_all2000, samples2000, samples12000, pdf2000, label = get_bestfit(models, zz[1], mu[1], mu_error[1], LL)
###params_allmock, samplesmock, samples1mock, pdfmock, label = get_bestfit(models, zzmock, mumock, mu_errormock, CLL)

y = np.linspace(0,0,30)
x = np.linspace(0.48,0.88,30)

c = ChainConsumer()
#c.add_chain(samples200, parameters=label, linewidth=2.0, name="200 SN", kde=1.5, color="black").configure(summary=True, shade_alpha=0.0) #linestyles="--"
#c.add_chain(samples275, parameters=label, linewidth=2.0, name="275 SN Mock", kde=1.5, color="blue").configure(summary=True, shade_alpha=0.0) #linestyles="--"
###c.add_chain(samples_UNBIN, parameters=label, linewidth=2.0, name="DES3YR Unbinned SN", kde=1.5, color="red").configure(summary=True, shade_alpha=0.0) #linestyles="--"
c.add_chain(samples_BIN, parameters=label, linewidth=2.0, name="DES3YR Binned SN", kde=1.5, color="black").configure(summary=True, shade_alpha=0.0) #linestyles="--"
#c.add_chain(samples2750, parameters=label, linewidth=2.0, name="2750 SN", kde=1.5, color="green").configure(summary=True, shade_alpha=0.0) #linestyles="--"
#c.add_chain(samples2000, parameters=label, linewidth=1.0, name="2000 SN", kde=1.5, color="red").configure(summary=True,shade_alpha=0.0)
###c.add_chain(samplesmock, parameters=label, linewidth=2.0, name="DES5YR Mock", kde=1.5, color="blue").configure(summary=True, shade_alpha=0.0) #, linestyles="--"
#c.plotter.plot(figsize="COLUMN", chains=["200 SN", "2000 SN", "DES5Yr Mock"],filename='200 vs 2000 vs Mock SN, Model: FGChap' ) 
#fig, axes = plt.subplots()
#plt.minorticks_on()
#c.plotter.plot_contour(axes,r"$A$",r"$\alpha$")
#axes.axhline(0,color = 'k', ls = ':')
#axes.set_xlabel(r"$A$")
#axes.set_ylabel(r"$\alpha$")
#axes.set_xlim(0.5,0.85)
#axes.set_ylim(-1,1.5)
#axes.legend()
#axes.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
c.plotter.plot(figsize="COLUMN", chains=["DES3YR Binned SN"],filename='IDE Binned vs Unbinned vs DES5YR_take2' )  
plt.show()
