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
    samples = emcee_run(zz1, mu1, mu_error1, params_begin1, nsamples, proposal, model)
    params_all.append(get_param(samples, label, model.__name__))
    return params_all, samples

c = ChainConsumer()
count = 2
zz = np.logspace(-2,0.2,20)
mu_error = np.linspace(0.1,0.1,20)
models = [wCDM]
#[FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]
meanmock = []
milne = LCDM(zz,[0,0])
for i, model in enumerate(models):
    params = []
    label, begin, legend = get_info(model.__name__)
    for j in range(count):
        mu = np.genfromtxt('Hubble Diagram/MockData/%s/%s_%d.txt' % (model.__name__,model.__name__, j))
        mockfit, samples = get_bestfit(model, zz, mu, mu_error)
        #c.add_chain(samples, parameters=label, color='b', name="Simulation validation")
        params.append(mockfit)
        mu_mock = model(zz, *mockfit)
        plt.plot(zz, mu_mock-milne, 'b', alpha = 0.2) # Plot each mock
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
    plt.text(0.1,-0.07,'Truth - %s' %(legend1))
    plt.legend(loc='best')
    plt.savefig('%s: Mock_x%s Hubble Diagram.png' % (model.__name__, count),bbox_inches='tight')
    
    plt.close()
    #fig = c.plotter.plot()
    plt.show()
