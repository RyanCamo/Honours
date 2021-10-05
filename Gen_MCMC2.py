import os
from matplotlib.markers import MarkerStyle
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer
import emcee

# This code will attempt to use emcee instead of metroplois hastings


def log_likelihood(model, zz, mu, mu_err): 
    delta = model - mu
    chit2 = np.sum(delta**2 / mu_err**2)
    B = np.sum(delta/mu_err**2)
    C = np.sum(1/mu_err**2)
    chi2 = chit2 - (B**2 / C) + np.log(C/(2* np.pi))
    #chi2 =  chit2
    # original log_likelihood ---->    -0.5 * np.sum((mu - model) ** 2 /mu_err**2) 
    return -0.5*chi2

def log_prior(parameters, begin):
    for i, (begins, parameter) in enumerate(zip(begin, parameters)):
        if abs(parameter) > 100*begins:
            return -np.inf
        return 0

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

def emceesampler(data_x, data_y, data_err, begin, nsamples, proposal_width, model):
    ndim, nwalkers = len(begin), 100
    ivar = 1. / np.random.rand(ndim)
    p0 = np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(data_x, data_y, data_err))
    sampler.run_mcmc(p0, 10000)

def metropolis_hastings(data_x, data_y, data_err, begin, nsamples, proposal_width, model):
    
    # Create an array to store the samples
    samples = np.empty((nsamples,len(begin)))
    print(len(begin))
    # Evaluate the likelihood for the starting values
    samples[0,0:] = begin
    print(samples[0,0:])
    log_posterior_old = log_prior(samples[0,0:], begin)
    if np.isinf(log_posterior_old):
        print("Starting values are outside your prior range!")
        exit()

    model_int = model(data_x, samples[0,0:]) 
    log_posterior_old += log_likelihood(model_int, data_x, data_y, data_err)
    

    # Loop over the required number of iterations
    acceptance = 0   # Update this every time you accept a sample
    for i in range(1,nsamples):
        
        # Generate a new set of parameters by drawing from a Gaussian with the proposal width
        new_params = np.random.normal(loc=samples[i-1,0:], scale=proposal_width)
        modeli = model(data_x, new_params)
        
        # Compute the log-prior and log-likelihood for the new samples 
        L_tot = log_prior(new_params, begin) + log_likelihood(modeli, data_x, data_y, data_err)
        
        # Compute the acceptance ratio
        alpha = np.exp((L_tot-log_posterior_old))
        #print('L_tot-Log_post:')
        #print(L_tot-log_posterior_old)
        #print('alpha:')
        #print(alpha)

        # Accept or reject the proposed values and store the results in samples.
        # Update the value of "acceptance" when we accept a sample so we can see how often we do it
        u = np.random.uniform()
        if u > alpha:
            samples[i,0:] = samples[i-1,0:]
        if u <= alpha:
            samples[i,0:] = new_params
            acceptance += 1  
            log_posterior_old = L_tot
    
        
        # Let's print how often we are accepting samples every thousand interations. 
        # This tells us whether we have set reasonable values for the proposal distribution
        if i%10000 == 0:
            print(str("Number of iterations=%d, Acceptance percentage:%d" % (i, int(100.0*acceptance/i))))
        
    # Return the samples for analysis
    return samples


# ---------- Import data ---------- #
# Import data
data = np.genfromtxt("HubbleDiagram/FITOPT000_MUOPT006.M0DIF",names=True,comments='#',dtype=None, skip_header=14, encoding = None)
zz = data['z']
mu = data['MUDIF'] + data['MUREF']
mu_error = data['MUDIFERR']
cov_arr = np.genfromtxt("HubbleDiagram/FITOPT000_MUOPT006.COV",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)



#ombegin, omproposal = 0.3, 0.05
#olbegin, olproposal = 0.7, 0.01
#mscrbegin, mscrproposal = 40, 1
#nsamples = int(1e5)
#samples = metropolis_hastings(zz, mu, mu_error, [ombegin, olbegin, mscrbegin], nsamples, [omproposal, olproposal, mscrproposal])

def get_params(samples, label):
    c = ChainConsumer()
    burnin = 2000
    burntin = samples[burnin:]
    c.add_chain(burntin, parameters=label, linewidth=2.0, name="MCMC").configure(summary=True)
    params = []
    for i, labelx in enumerate(label):
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    return params
