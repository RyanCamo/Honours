import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()

# Import data
data = np.genfromtxt("Covariance Matrix/DATA_simdes5yr_binned.txt",names=True,dtype=None, encoding=None, delimiter=',')
mu_error = data['MUERR']
mu_error_diag = np.diag(mu_error)**2


cov_arr = np.genfromtxt("Covariance Matrix/COVsyst_simdes5yr_binned.txt",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)

tot = 0
for i, m in enumerate(cov_arr[0:20]):
    tot +=m**2

# Plots
fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2)

# Left Images - Covariance matrix on its own
sns.heatmap(cov,ax = ax1) # Top left: Covariance matrix on its own
ax1.set_ylabel('Not Logged data')
ax1.set_xticks([])
ax1.set_yticks([])

sns.heatmap(np.log(cov),ax = ax3) # Bottom left: Covariance matrix on its own - logged
ax3.set_xlabel('Cov')
ax3.set_ylabel('Logged data')
ax3.set_xticks([])
ax3.set_yticks([])

# Right Images - Covariance matrix with mu_error added to the diagonal
sns.heatmap(cov+mu_error_diag,ax = ax2) # Top right: Covariance matrix on its own
ax2.set_xticks([])
ax2.set_yticks([])

sns.heatmap(np.log(cov+mu_error_diag),ax = ax4) # Bottom right: Covariance matrix on its own - logged
ax4.set_xlabel('Cov + diag(mu_error)')
ax4.set_xticks([])
ax4.set_yticks([])

plt.show()