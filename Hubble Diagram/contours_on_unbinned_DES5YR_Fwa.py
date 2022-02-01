import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()
name = Fwa.__name__
# Import Chains #
CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/%s_CHAIN_covloglike.txt' % name)

# Get Model Info #
label, begin, legend = get_info(name)


# The actualchain this code is about
c.add_chain(CHAIN, parameters=label, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM for both params
om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
o0 = c.analysis.get_summary(chains=name)['$w_0$'][1]
o0p = c.analysis.get_summary(chains=name)['$w_0$'][2]-c.analysis.get_summary(chains=name)['$w_0$'][1]
o0m = c.analysis.get_summary(chains=name)['$w_0$'][1]-c.analysis.get_summary(chains=name)['$w_0$'][0]
oa = c.analysis.get_summary(chains=name)['$w_a$'][1]
oap =c.analysis.get_summary(chains=name)['$w_a$'][2]-c.analysis.get_summary(chains=name)['$w_a$'][1]
oam =c.analysis.get_summary(chains=name)['$w_a$'][1]-c.analysis.get_summary(chains=name)['$w_a$'][0]


#### Plotting ###
fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, '$w_0$', '$w_a$') 

# Unique to LCDM model
ax.text(-0.15,-2,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(om,omp,omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(-0.15,-2-0.38,'$\omega_0 = %10.5s^{%10.5s}_{%10.5s}$' %(o0,o0p,o0m), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(-0.15,-2-0.76,'$\omega_{a} = %10.5s^{%10.5s}_{%10.5s}$' %(oa,oap,oam), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\omega_0$', fontsize = 18)
ax.set_ylabel(r'$\omega_a$', fontsize = 18)

# Plot limits
ax.set_xlim(-2,0)
ax.set_ylim(-3,2)
ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()