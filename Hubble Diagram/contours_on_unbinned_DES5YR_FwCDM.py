import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()
name = FwCDM.__name__
# Import Chains #
CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/%s_CHAIN_covloglike.txt' % name)
FLCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/FLCDM_CHAIN_covloglike.txt')
# Get Model Info #
label, begin, legend = get_info(name)
label_FLCDM, begin_FLCDM, legend_FLCDM = get_info(FLCDM.__name__)

# Added FLCDM Chain to plot as point on FwCDM contour #
c.add_chain(FLCDM_CHAIN, parameters=label_FLCDM, linewidth=2.0, name="FLCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM
fom = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomp = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomm = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][0]

# Removes the chain so doesnt try to plot anything
c.remove_chain('FLCDM')


# The actualchain this code is about
c.add_chain(CHAIN, parameters=label, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM for both params
om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
w = c.analysis.get_summary(chains=name)[r'$\omega$'][1]
wp =c.analysis.get_summary(chains=name)[r'$\omega$'][2]-c.analysis.get_summary(chains=name)[r'$\omega$'][1]
wm =c.analysis.get_summary(chains=name)[r'$\omega$'][1]-c.analysis.get_summary(chains=name)[r'$\omega$'][0]
print(omp)
print(omm)
print(wp)
print(wm)
#### Plotting ###
fig, ax = plt.subplots(1, 1)

# Unique to this plot - plotting FLCDM best fit param
ax.scatter(fom, -1, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, *label) 
#c.plotter.plot(figsize="COLUMN", chains=name,filename='Model: %s' % model) 

# Unique to FwCDM model
ax.text(0.29,-1.75,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(om,omp,omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.29,-1.75-0.11,'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(w,wp,wm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\omega$', fontsize = 18)

# Plot limits
ax.set_xlim(0.20,0.5)
#ax.set_ylim(0.48,1.06)

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()