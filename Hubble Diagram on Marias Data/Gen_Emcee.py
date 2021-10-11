from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer
from numpy.core.numeric import NaN
import numpy as np
from ModelMerger import *
import emcee

def log_likelihood(params, zz, mu, mu_err, model): 
    mu_model = model(zz, params)
    delta = mu_model - mu
    chit2 = np.sum(delta**2 / mu_err**2)
    B = np.sum(delta/mu_err**2)
    C = np.sum(1/mu_err**2)
    chi2 = chit2 - (B**2 / C) + np.log(C/(2* np.pi))
    #chi2 =  np.sum((mu - model) ** 2 /mu_err**2) 
    # original log_likelihood ---->    -0.5 * np.sum((mu - model) ** 2 /mu_err**2) 
    return -0.5*chi2

def cov_log_likelihood(params, zz, mu, cov, model):
    mu_model = model(zz, params)
    delta = np.array([mu_model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
    return -0.5*chi2 

def log_prior(parameters, begin):
    for i, (begins, parameter) in enumerate(zip(begin, parameters)):
        if abs(parameter) > 10*begins:
            return -np.inf
        return 0

def L_tot(params, zz, mu, mu_err, model, begin): 
    like = cov_log_likelihood(params, zz, mu, mu_err, model)
    prior = log_prior(params, begin)
    if np.isnan(like+prior) == True:
        return -np.inf

    return like + prior


def emcee_run(data_x, data_y, data_err, begin, nsamples, proposal_width, model):
    nwalkers = 100
    ndim = len(begin)
    p0 = [np.array(begin) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]


    sampler = emcee.EnsembleSampler(nwalkers, ndim, L_tot, args=(data_x, data_y, data_err, model,begin))
    sampler.run_mcmc(p0, 400, progress=True)
    samples = sampler.get_chain(discard=50, thin=7,flat=True)

    return samples