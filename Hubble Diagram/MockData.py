import numpy as np
from ModelMergerUpdated import *

# This file creates mock data for each model and saves as a file 

# List of models to loop through from ModelMergerUpdated file
models = [FLCDM, LCDM, FwCDM, wCDM, Fwa, FCa, Chap, FGChap, GChap, DGP]

# How many Mocks 
count = 1000
zz = np.logspace(-2,0.2,20)
if __name__ == "__main__":
    for i, model in enumerate(models):
        for j in range(count):
            label, begin, legend = get_info(model.__name__)
            var = np.random.normal(scale = 0.01, size = 20)
            mu = model(zz,begin)
            mu_tot = model(zz,begin) + var
            np.savetxt('Hubble Diagram/MockData/%s/%s_%d.txt' % (model.__name__,model.__name__, j) , mu_tot, fmt="%10.4f")
