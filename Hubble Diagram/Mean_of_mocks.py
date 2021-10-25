import numpy as np
from ModelMergerUpdated import *
from Gen_Emcee import *
from chainconsumer import ChainConsumer


# Get Best fit params using MCMC for each model
def get_bestfit(model, zz1, mu1, mu_error1):     # get_labels dont have marginalised value in but MCMC uses it
    params_all = []
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
    return params_all, samples, samples1, pdf


count = 2
zz = np.logspace(-2,0.2,20)
mu_error = np.linspace(0.01,0.01,20)
models = [FLCDM, LCDM]
#[FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]
meanmock = []
milne = LCDM(zz,[0,0])
for i, model in enumerate(models):
    c = ChainConsumer()
    params = []
    label, begin, legend = get_info(model.__name__)
    for j in range(count):
        mu = np.genfromtxt('Hubble Diagram/MockData/%s/%s_%d.txt' % (model.__name__,model.__name__, j))
        mockfit, samples, samples1, pdf = get_bestfit(model, zz, mu, mu_error)
        c.add_chain(samples1[0], posterior=pdf, parameters=label, color='b', name="Simulation validation")
        params.append(mockfit)
        mu_mock = model(zz, *mockfit)
        offset = (np.sum(mu - mu_mock))/len(zz)
        plt.plot(zz, (mu_mock-(offset))-milne, 'b', alpha = 0.2) # Plot each mock
    # Get the mean of the mock
    params_shaped = np.reshape(params, (count,len(begin)))
    mean_params = []
    for k in range(len(begin)):
        mean_params.append(np.mean(params_shaped[:,k]))
    meanmock.append(mean_params)

    mu_mean = model(zz, mean_params) 
    plt.xlabel('Redshift, z')
    plt.ylabel(r'$\Delta$ Distance Modulus (Mag)')
    label, begin, legend = get_info(model.__name__, *mean_params) # grab the label for the mean values
    plt.plot(zz,mu_mean-milne, 'k', label = 'Mean - %s' %(legend) ) # plot the mean
    label1, begin1, legend1 = get_info(model.__name__, *begin) # grab the label for the true values
    plt.plot([], [], ' ', label='Truth - %s' %(legend1)) # adding the label for the truth value to the legend
    plt.legend(loc='lower left')
    plt.savefig('%s: Mock_x%s Hubble Diagram.png' % (model.__name__, count),bbox_inches='tight')
    
    plt.close()
    fig = c.plotter.plot()
    plt.savefig('%s: Mock_x%s CC.png' % (model.__name__, count),bbox_inches='tight')
    plt.close()
