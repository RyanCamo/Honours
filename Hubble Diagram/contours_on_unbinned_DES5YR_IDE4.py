import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()
name = IDE4.__name__
# Import Chains #
CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/Chains_OUTPUT_Geta/%s_CHAIN_covloglike.txt' % name)

# Get Model Info #
label, begin, legend = get_info(name)


# The actual LCDM chain this code is about
c.add_chain(CHAIN, parameters=label, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error on FLCDM for both params
om = c.analysis.get_summary(chains=name)[r'$\Omega_{CDM}$'][1]
#omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
#omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][0]
ol = c.analysis.get_summary(chains=name)[r'$\Omega_{DE}$'][1]
#olp =c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
#olm =c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][0]
print(om)
print(ol)
#### Plotting ###
fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, r"$\Omega_{CDM}$", r"$\Omega_{DE}$") #r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"

# Unique to LCDM model
#ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
#ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')
#ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
#ax.set_ylabel(r'$\Omega_{\Lambda}$', fontsize = 18)

# Plot limits
#ax.set_xlim(0.20,0.57)
#ax.set_ylim(0.48,1.06)

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()