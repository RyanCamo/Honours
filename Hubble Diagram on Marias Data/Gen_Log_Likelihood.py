import numpy as np
from ModelMerger import *

def cov_log_likelihood(model, mu, cov):
    delta = np.array([model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
    return chi2 # -0.5* (removed for grid)

