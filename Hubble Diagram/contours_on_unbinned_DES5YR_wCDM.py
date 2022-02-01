import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()
name = wCDM.__name__
# Import Chains #
CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/%s_CHAIN_covloglike.txt' % name)
LCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/LCDM_CHAIN_covloglike.txt')

# Get Model Info #
label, begin, legend = get_info(name)
label_LCDM, begin_LCDM, legend_LCDM = get_info(LCDM.__name__)

c.add_chain(LCDM_CHAIN, parameters=label_LCDM, linewidth=2.0, name="LCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")

omlcdm = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]

# Removes the chain so doesnt try to plot anything
c.remove_chain('LCDM')

# The actual chain this code is about
c.add_chain(CHAIN[500:], parameters=label, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM for both params
om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
ol = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]
olp = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]
olm = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][0]
o = c.analysis.get_summary(chains=name)[r'$\omega$'][1]
op =c.analysis.get_summary(chains=name)[r'$\omega$'][2]-c.analysis.get_summary(chains=name)[r'$\omega$'][1]
omi =c.analysis.get_summary(chains=name)[r'$\omega$'][1]-c.analysis.get_summary(chains=name)[r'$\omega$'][0]

#### Plotting ###
fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, r'$\Omega_m$', r"$\omega$")

# Unique to this plot - plotting LCDM best fit param
ax.scatter(omlcdm, -1, marker = 'D', s = 50, c='black', label = r'$\Lambda$')
print(omlcdm)

# Unique to model
ax.text(0.58,-1.7,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(om,omp,omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.58,-1.7-0.1,'$\Omega_{\Lambda} = %10.5s^{%10.5s}_{%10.5s}$' %(ol,olp,olm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(0.58,-1.7-0.2,'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(o,op,omi), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\omega$', fontsize = 18)

# Plot limits
ax.set_xlim(0.20,0.6)
ax.set_ylim(-2,-0.5)
ax.set_xticklabels(['0.2','','0.3','','0.4','','0.5','','0.6'])
ax.set_yticklabels(['','-1.8','','-1.4','','-1.0','','-0.6'])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()