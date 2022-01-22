import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()

# Import Chains #
LCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/LCDM_CHAIN_covloglike.txt')
FLCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/FLCDM_CHAIN_covloglike.txt')

# Get Model Info #
label_FLCDM, begin_FLCDM, legend_FLCDM = get_info(FLCDM.__name__)
label_LCDM, begin_LCDM, legend_LCDM = get_info(LCDM.__name__)

# Added FLCDM Chain to plot as point on LCDM contour #
c.add_chain(FLCDM_CHAIN, parameters=label_FLCDM, linewidth=2.0, name="FLCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM
fom = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomp = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomm = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][0]

# Removes the chain so doesnt try to plot anything
c.remove_chain('FLCDM')

# The actual LCDM chain this code is about
c.add_chain(LCDM_CHAIN, parameters=label_LCDM, linewidth=2.0, name="LCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM for both params
om = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
omp = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][0]
ol = c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olp =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olm =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][0]

#### Plotting ###
fig, ax = plt.subplots(1, 1)

# Unique to this plot - plotting FLCDM best fit param
ax.scatter(fom, 1-fom, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax,r'$\Omega_m$', r'$\Omega_{\Lambda}$')

# Unique to LCDM model
ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\Omega_{\Lambda}$', fontsize = 18)

# Plot limits
ax.set_xlim(0.20,0.57)
ax.set_ylim(0.48,1.06)

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()