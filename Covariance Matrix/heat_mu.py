import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

# Import data
data = np.genfromtxt("Covariance Matrix/FITOPT000_MUOPT006.M0DIF",names=True,comments='#',dtype=None, skip_header=14, encoding=None)
mu_error = data['MUDIFERR']
mu_error_diag = np.diag(mu_error)
#print(mu_error_diag)

cov_arr = np.genfromtxt("Covariance Matrix/FITOPT000_MUOPT006.COV",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)
#print(np.diag(cov))

# Plots
fig, (ax1,ax3) = plt.subplots(2,1)

# Left Images 
sns.heatmap(mu_error_diag,ax = ax1) # Top left: Diag_mu_error on its own
ax1.set_ylabel('Not Logged data')
ax1.set_xticks([])
ax1.set_yticks([])

sns.heatmap(np.log(mu_error_diag),ax = ax3) # Bottom left: Diag_mu_error on its own - logged
ax3.set_xlabel('mu_error')
ax3.set_ylabel('Logged data')
ax3.set_xticks([])
ax3.set_yticks([])


plt.show()



plt.show()