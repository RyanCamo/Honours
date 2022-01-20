import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 


c = ChainConsumer()
LCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/LCDM_CHAIN_covloglike.txt')
FLCDM_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/FLCDM_CHAIN_covloglike.txt')
LCDM_pdf = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/LCDM_POSTERIOR_covloglike.txt')
FGChap_CHAIN = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/FGChap_CHAIN_covloglike.txt')
FGChap_pdf = np.loadtxt('/Users/RyanCamo/Documents/GitHub/Honours/Hubble Diagram/Chains_OUTPUT/FGChap_POSTERIOR_covloglike.txt')

label_FLCDM, begin_FLCDM, legend_FLCDM = get_info(FLCDM.__name__)
label_LCDM, begin_LCDM, legend_LCDM = get_info(LCDM.__name__)
label_FGChap, begin_FGChap, legend_FGChap = get_info(FGChap.__name__)

fig, ax = plt.subplots(1, 1)

c.add_chain(FLCDM_CHAIN, parameters=label_FLCDM, linewidth=2.0, name="FLCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")
fom = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomp = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomm = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][0]
print(fom)
print(fomp)
print(fomm)

c.remove_chain('FLCDM')

c.add_chain(LCDM_CHAIN, parameters=label_LCDM, linewidth=2.0, name="LCDM", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")
om = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
print(om)
omp = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][0]
ol = c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olp =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olm =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][0]
c.plotter.plot_contour(ax,r'$\Omega_m$', r'$\Omega_{\Lambda}$')
ax.scatter(fom, 1-fom, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')
ax.text(0.6,0.47,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.6,0.47-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\Omega_{\Lambda}$', fontsize = 18) 
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")

#c.remove_chain('LCDM')


#c.add_chain(FGChap_CHAIN, parameters=[r'$A$', r'$\alpha$'], linewidth=2.0, name="FGChap", kde=1.5, color="red",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")
#A = c.analysis.get_summary(chains="FGChap")[r'$A$'][1]
#Ap = c.analysis.get_summary(chains="FGChap")[r'$A$'][2]-c.analysis.get_summary(chains="FGChap")[r'$A$'][1]
#Am = c.analysis.get_summary(chains="FGChap")[r'$A$'][1]-c.analysis.get_summary(chains="FGChap")[r'$A$'][0]
#a = c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]
#ap =c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][2]-c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]
#am =c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]-c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][0]
#ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
#ok = 1- (om+ol)
#ax.scatter(1-fom, 0, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$', zorder = 2)
#ax.scatter(1-(om/(1-ok)), 0, marker = '*', s = 60, c='blue', label = r'$\Lambda$', zorder = 3)
#c.plotter.plot_contour(ax,r'$A$', r'$\alpha$')
#ax.text(0.87,-0.67,'$A = %10.5s\pm{%10.5s}$' %(A,Ap), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
#ax.text(0.87,-0.67-0.2,r'$\alpha = %10.5s\pm{%10.5s}$' %(a,ap), family='serif',color='black',rotation=0,fontsize=12,ha='right')
##ax.text(0.86,-0.5-0.4,r'$X=\Lambda CDM$ Best Fit', family='serif',color='black',rotation=0,fontsize=12,ha='right')
#ax.set_xlabel(r'$A$', fontsize = 18)
#ax.set_ylabel(r'$\alpha$', fontsize = 18) 
#ax.set_ylim(-1,2)
#plt.minorticks_on()
#ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")



#c.plotter.plot(figsize="COLUMN", chains=['FGChap'],filename='FGChap_TEST' )  
#c.plotter.plot(figsize="COLUMN", chains=['LCDM'],filename='LCDM_TEST' )  
#print(c.comparison.comparison_table(caption="Model comparisons!"))
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()