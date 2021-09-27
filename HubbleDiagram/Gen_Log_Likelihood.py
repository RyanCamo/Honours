import numpy as np
from ModelMerger import *

def log_likelihood(model, mu, cov):
    delta = np.array([model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chi2 = np.sum(deltaT * inv_cov * delta)
    return -0.5*chi2