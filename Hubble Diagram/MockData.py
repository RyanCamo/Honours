import numpy as np
from ModelMergerUpdated import *

# This file creates mock data for each model and saves as a file
# 
data = np.genfromtxt("/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/DES3YR_UNBINNED.txt")#,names=True,dtype=None, encoding=None, delimiter='tab')
#zz = data[:,1]
zz = np.tile(data[:,1], 10)
#data[:,1]


# List of models to loop through from ModelMergerUpdated file
models = [FLCDM]

# How many Mocks 
count = 1
#zz = np.logspace(-2,0.2,200)
if __name__ == "__main__":
    for i, model in enumerate(models):
        for j in range(count):
            label, begin, legend = get_info(model.__name__)
            var = np.random.normal(scale = 0.1, size = 2750)
            mu = model(zz,begin)
            mu_tot = model(zz,begin) + var
            #np.savetxt('Hubble Diagram/MockData/%s/%s_%d.txt' % (model.__name__,model.__name__, j) , mu_tot, fmt="%10.4f")
            np.savetxt('Hubble Diagram/MockData/2000v200/%s2750_%d_.txt' % (model.__name__, j) , mu_tot, fmt="%10.4f")
