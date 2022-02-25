import numpy as np
from scipy.integrate import quad
from ModelMergerUpdated import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 

c = ChainConsumer()

# Import Chains for all the models DES5YR_BIN - Missing LTB & F(R)

FLCDM_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/FLCDM_DES5YR_BIN.1.txt', usecols=(2), comments='#')
LCDM_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/LCDM_DES5YR_BIN.1.txt', usecols=(2,3), comments='#')
FwCDM_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/FwCDM_DES5YR_BIN.1.txt', usecols=(2,3), comments='#')
wCDM_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/wCDM_DES5YR_BIN.1.txt', usecols=(2,3,4), comments='#')
Fwa_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/Fwa_DES5YR_BIN.1.txt', usecols=(2,3,4), comments='#')
Fwz_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/Fwz_DES5YR_BIN.1.txt', usecols=(2,3,4), comments='#')
IDE1_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/IDE1_DES5YR_BIN.1.txt', usecols=(2,3,4,5), comments='#')
IDE2_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/IDE2_DES5YR_BIN.1.txt', usecols=(2,3,4,5), comments='#')
IDE4_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/IDE4_DES5YR_BIN.1.txt', usecols=(2,3,4,5), comments='#')
FGChap_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/FGChap_DES5YR_BIN.1.txt', usecols=(2,3), comments='#')
GChap_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/GChap_DES5YR_BIN.1.txt', usecols=(2,3,4), comments='#')
Chap_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/Chap_DES5YR_BIN.1.txt', usecols=(2,3), comments='#')
FCa_COBAYA_CHAIN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/FCa_DES5YR_BIN.1.txt', usecols=(2,3,4), comments='#')

# Get Info For Each Model

label_FLCDM, begin_FLCDM, legend_FLCDM = get_info(FLCDM.__name__)
label_LCDM, begin_LCDM, legend_LCDM = get_info(LCDM.__name__)
label_FGChap, begin_FGChap, legend_FGChap = get_info(FGChap.__name__)
label_FwCDM, begin_FwCDM, legend_FwCDM = get_info(FwCDM.__name__)
label_wCDM, begin_wCDM, legend_wCDM = get_info(wCDM.__name__)
label_FCa, begin_FCa, legend_FCa = get_info(FCa.__name__)
label_IDE1, begin_IDE1, legend_IDE1 = get_info(IDE1.__name__)
label_IDE2, begin_IDE2, legend_IDE2 = get_info(IDE2.__name__)
label_IDE4, begin_IDE4, legend_IDE4 = get_info(IDE4.__name__)
label_Fwa, begin_Fwa, legend_Fwa = get_info(Fwa.__name__)
label_Fwz, begin_Fwz, legend_Fwz = get_info(Fwz.__name__)
label_GChap, begin_GChap, legend_GChap = get_info(GChap.__name__)
label_Chap, begin_Chap, legend_Chap = get_info(Chap.__name__)

#### PLOTS FOR DIFFERENT MODELS - NOTE PARAMS ARE SAVED AS: MODEL_PARAM - BESIDES FLCDM/LCDM

### FLCDM - Used as a Scatter point on some plots
c.add_chain(FLCDM_COBAYA_CHAIN, parameters=label_FLCDM, linewidth=2.0, name="FLCDM", kde=1.5, color="blue").configure(summary=True, shade_alpha=0.2,statistics="max")
fom = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomp = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]
fomm = c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="FLCDM")[r'$\Omega_m$'][0]
c.remove_chain('FLCDM')

### LCDM PLOT & points which is sometimes used as a scatter point on some plots
fig, ax = plt.subplots(1, 1)
c.add_chain(LCDM_COBAYA_CHAIN, parameters=label_LCDM, linewidth=2.0, name="LCDM", kde=1.5, color="blue",num_free_params=2).configure(summary=True, shade_alpha=0.2,statistics="max")
om = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
omp = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]
omm = c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_m$'][0]
ol = c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olp =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]
olm =c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains="LCDM")[r'$\Omega_{\Lambda}$'][0]
c.plotter.plot_contour(ax,r'$\Omega_m$', r'$\Omega_{\Lambda}$')
ax.scatter(fom, 1-fom, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')
ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\Omega_{\Lambda}$', fontsize = 18) 
ax.set_xlim(0.20,0.57)
ax.set_ylim(0.48,1.06)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.close()
c.remove_chain('LCDM')

### FwCDM PLOT
name = FwCDM.__name__
c.add_chain(FwCDM_COBAYA_CHAIN[200:], parameters=label_FwCDM, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_FwCDM)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error
FwCDM_om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
FwCDM_omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
FwCDM_omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
FwCDM_w = c.analysis.get_summary(chains=name)[r'$\omega$'][1]
FwCDM_wp =c.analysis.get_summary(chains=name)[r'$\omega$'][2]-c.analysis.get_summary(chains=name)[r'$\omega$'][1]
FwCDM_wm =c.analysis.get_summary(chains=name)[r'$\omega$'][1]-c.analysis.get_summary(chains=name)[r'$\omega$'][0]
fig, ax = plt.subplots(1, 1)

# Unique to this plot - plotting FLCDM best fit param
ax.scatter(fom, -1, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, *label_FwCDM) 

# Unique to FwCDM model
ax.text(0.24,-1.67,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(FwCDM_om,FwCDM_omp,FwCDM_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.24,-1.67-0.11,'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(FwCDM_w,FwCDM_wp,FwCDM_wm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$\Omega_m$', fontsize = 18)
ax.set_ylabel(r'$\omega$', fontsize = 18)

# Plot limits
#ax.set_xlim(0.20,0.5)
#ax.set_ylim(0.48,1.06)

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.close()
c.remove_chain('FwCDM')

### wCDM PLOT
name = wCDM.__name__
c.add_chain(wCDM_COBAYA_CHAIN[500:], parameters=label_wCDM, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_wCDM)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
wCDM_om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
wCDM_omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
wCDM_omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
wCDM_ol = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]
wCDM_olp = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]
wCDM_olm = c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_{\Lambda}$'][0]
wCDM_o = c.analysis.get_summary(chains=name)[r'$\omega$'][1]
wCDM_op =c.analysis.get_summary(chains=name)[r'$\omega$'][2]-c.analysis.get_summary(chains=name)[r'$\omega$'][1]
wCDM_omi =c.analysis.get_summary(chains=name)[r'$\omega$'][1]-c.analysis.get_summary(chains=name)[r'$\omega$'][0]

fig, ax = plt.subplots(1, 1)
# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, r'$\Omega_m$', r"$\omega$")

# Unique to this plot - plotting LCDM best fit param
ax.scatter(om, -1, marker = 'X', s = 60, c='red', label = r'$\Lambda$')

# Unique to model
ax.text(0.58,-1.7,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(wCDM_om,wCDM_omp,wCDM_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.58,-1.7-0.1,'$\Omega_{\Lambda} = %10.5s^{%10.5s}_{%10.5s}$' %(wCDM_ol,wCDM_olp,wCDM_olm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(0.58,-1.7-0.2,'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(wCDM_o,wCDM_op,wCDM_omi), family='serif',color='black',rotation=0,fontsize=12,ha='right')
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
plt.close()
#plt.show()
c.remove_chain('wCDM')


### Fwa PLOT
name = Fwa.__name__
# The actualchain this code is about
c.add_chain(Fwa_COBAYA_CHAIN[1000:], parameters=label_Fwa, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_Fwa)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
Fwa_om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
Fwa_omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
Fwa_omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
Fwa_o0 = c.analysis.get_summary(chains=name)['$w_0$'][1]
Fwa_o0p = c.analysis.get_summary(chains=name)['$w_0$'][2]-c.analysis.get_summary(chains=name)['$w_0$'][1]
Fwa_o0m = c.analysis.get_summary(chains=name)['$w_0$'][1]-c.analysis.get_summary(chains=name)['$w_0$'][0]
Fwa_oa = c.analysis.get_summary(chains=name)['$w_a$'][1]
Fwa_oap =c.analysis.get_summary(chains=name)['$w_a$'][2]-c.analysis.get_summary(chains=name)['$w_a$'][1]
Fwa_oam =c.analysis.get_summary(chains=name)['$w_a$'][1]-c.analysis.get_summary(chains=name)['$w_a$'][0]

fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax, '$w_0$', '$w_a$') 

# Unique to model
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ax.scatter(-1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.text(-1.85,-7,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(Fwa_om,Fwa_omp,Fwa_omm), family='serif',color='black',rotation=0,fontsize=12,ha='left') 
ax.text(-1.85,-7-1,'$\omega_0 = %10.5s^{%10.5s}_{%10.5s}$' %(Fwa_o0,Fwa_o0p,Fwa_o0m), family='serif',color='black',rotation=0,fontsize=12,ha='left')
ax.text(-1.85,-7-2,'$\omega_{a} = %10.5s^{%10.5s}_{%10.5s}$' %(Fwa_oa,Fwa_oap,Fwa_oam), family='serif',color='black',rotation=0,fontsize=12,ha='left')
ax.set_xlabel(r'$\omega_0$', fontsize = 18)
ax.set_ylabel(r'$\omega_a$', fontsize = 18)

# Plot limits
#ax.set_xlim(-2,0)
#ax.set_ylim(-3,2)
#ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('Fwa')

### Fwz PLOT
name = Fwz.__name__

# The actual chain this code is about
c.add_chain(Fwz_COBAYA_CHAIN[500:], parameters=label_Fwz, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_Fwz)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
Fwz_om = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
Fwz_omp = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][2]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]
Fwz_omm = c.analysis.get_summary(chains=name)[r'$\Omega_m$'][1]-c.analysis.get_summary(chains=name)[r'$\Omega_m$'][0]
Fwz_o0 = c.analysis.get_summary(chains=name)['$w_0$'][1]
Fwz_o0p = c.analysis.get_summary(chains=name)['$w_0$'][2]-c.analysis.get_summary(chains=name)['$w_0$'][1]
Fwz_o0m = c.analysis.get_summary(chains=name)['$w_0$'][1]-c.analysis.get_summary(chains=name)['$w_0$'][0]
Fwz_oz = c.analysis.get_summary(chains=name)['$w_z$'][1]
Fwz_ozp =c.analysis.get_summary(chains=name)['$w_z$'][2]-c.analysis.get_summary(chains=name)['$w_z$'][1]
Fwz_ozm =c.analysis.get_summary(chains=name)['$w_z$'][1]-c.analysis.get_summary(chains=name)['$w_z$'][0]

fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax,'$w_0$', '$w_z$') 

# Unique to model
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ax.scatter(-1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.text(-1.85,-7,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(Fwz_om,Fwz_omp,Fwz_omm), family='serif',color='black',rotation=0,fontsize=12,ha='left') 
ax.text(-1.85,-7-1,'$\omega_0 = %10.5s^{%10.5s}_{%10.5s}$' %(Fwz_o0,Fwz_o0p,Fwz_o0m), family='serif',color='black',rotation=0,fontsize=12,ha='left')
ax.text(-1.85,-7-2,'$\omega_{z} = %10.5s^{%10.5s}_{%10.5s}$' %(Fwz_oz,Fwz_ozp,Fwz_ozm), family='serif',color='black',rotation=0,fontsize=12,ha='left')
ax.set_xlabel(r'$\omega_0$', fontsize = 18)
ax.set_ylabel(r'$\omega_z$', fontsize = 18)

# Plot limits
#ax.set_xlim(-2,0)
#ax.set_ylim(-3,2)
#ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('Fwz')

### IDE1 PLOT

### FGCHAP PLOT 

c.add_chain(FGChap_COBAYA_CHAIN[100:], parameters=[r'$A$', r'$\alpha$'], linewidth=2.0, name="FGChap", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")
FGChap_A = c.analysis.get_summary(chains="FGChap")[r'$A$'][1]
FGChap_Ap = c.analysis.get_summary(chains="FGChap")[r'$A$'][2]-c.analysis.get_summary(chains="FGChap")[r'$A$'][1]
FGChap_Am = c.analysis.get_summary(chains="FGChap")[r'$A$'][1]-c.analysis.get_summary(chains="FGChap")[r'$A$'][0]
FGChap_a = c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]
FGChap_ap =c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][2]-c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]
FGChap_am =c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][1]-c.analysis.get_summary(chains="FGChap")[r'$\alpha$'][0]

fig, ax = plt.subplots(1, 1)

ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ok = 1- (om+ol)
ax.scatter(1-fom, 0, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$', zorder = 2)
c.plotter.plot_contour(ax,r'$A$', r'$\alpha$')
ax.text(0.825,-0.53,r'$A = %10.5s^{%10.5s}_{%10.5s}$' %(FGChap_A,FGChap_Ap,FGChap_Am), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.825,-0.53-0.2,r'$\alpha = %10.5s^{%10.5s}_{%10.5s}$' %(FGChap_a,FGChap_ap,FGChap_am), family='serif',color='black',rotation=0,fontsize=12,ha='right')
#ax.text(0.86,-0.5-0.4,r'$X=\Lambda CDM$ Best Fit', family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$A$', fontsize = 18)
ax.set_ylabel(r'$\alpha$', fontsize = 18) 
#ax.set_ylim(-1,2)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper left',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('FGChap')
#c.plotter.plot(figsize="COLUMN", chains=['FGChap'],filename='FGChap_TEST' )  
#c.plotter.plot(figsize="COLUMN", chains=['LCDM'],filename='LCDM_TEST' )  
#print(c.comparison.comparison_table(caption="Model comparisons!"))

### CHAP PLOT 
c.add_chain(Chap_COBAYA_CHAIN[500:], parameters=[r'$A$', r'$\Omega_{K}$'], linewidth=2.0, name="Chap", kde=1.5, color="blue",num_free_params=2, num_eff_data_points=20).configure(summary=True, shade_alpha=0.2,statistics="max")
Chap_A = c.analysis.get_summary(chains="Chap")[r'$A$'][1]
Chap_Ap = c.analysis.get_summary(chains="Chap")[r'$A$'][2]-c.analysis.get_summary(chains="Chap")[r'$A$'][1]
Chap_Am = c.analysis.get_summary(chains="Chap")[r'$A$'][1]-c.analysis.get_summary(chains="Chap")[r'$A$'][0]
Chap_ok = c.analysis.get_summary(chains="Chap")[r'$\Omega_{K}$'][1]
Chap_okp =c.analysis.get_summary(chains="Chap")[r'$\Omega_{K}$'][2]-c.analysis.get_summary(chains="Chap")[r'$\Omega_{K}$'][1]
Chap_okm =c.analysis.get_summary(chains="Chap")[r'$\Omega_{K}$'][1]-c.analysis.get_summary(chains="Chap")[r'$\Omega_{K}$'][0]

fig, ax = plt.subplots(1, 1)

#ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
c.plotter.plot_contour(ax, r'$\Omega_{K}$',r'$A$')
ax.text(0.78,0.3,r'$A = %10.5s^{%10.5s}_{%10.5s}$' %(Chap_A,Chap_Ap,Chap_Am), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.78,0.3-0.08,r'$\Omega_{K} = %10.5s^{%10.5s}_{%10.5s}$' %(Chap_ok,Chap_okp,Chap_okm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_ylabel(r'$A$', fontsize = 18)
ax.set_xlabel(r'$\Omega_{K}$', fontsize = 18) 
ax.set_ylim(0.15,1.025)
ax.set_xlim(-0.85,0.85)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper left',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('Chap')

### GCHAP PLOT 
c.add_chain(GChap_COBAYA_CHAIN[800:], parameters=[r"$A$",r"$\alpha$",r"$\Omega_{K}$"], linewidth=2.0, name="GChap", kde=1.5, color="blue",num_free_params=2).configure(summary=True, shade_alpha=0.2,statistics="max")
GChap_A = c.analysis.get_summary(chains="GChap")[r'$A$'][1]
GChap_Ap = c.analysis.get_summary(chains="GChap")[r'$A$'][2]-c.analysis.get_summary(chains="GChap")[r'$A$'][1]
GChap_Am = c.analysis.get_summary(chains="GChap")[r'$A$'][1]-c.analysis.get_summary(chains="GChap")[r'$A$'][0]
GChap_ok = c.analysis.get_summary(chains="GChap")[r'$\Omega_{K}$'][1]
GChap_okp =c.analysis.get_summary(chains="GChap")[r'$\Omega_{K}$'][2]-c.analysis.get_summary(chains="GChap")[r'$\Omega_{K}$'][1]
GChap_okm =c.analysis.get_summary(chains="GChap")[r'$\Omega_{K}$'][1]-c.analysis.get_summary(chains="GChap")[r'$\Omega_{K}$'][0]
GChap_a = c.analysis.get_summary(chains="GChap")[r"$\alpha$"][1]
GChap_ap =c.analysis.get_summary(chains="GChap")[r"$\alpha$"][2]-c.analysis.get_summary(chains="GChap")[r"$\alpha$"][1]
GChap_am =c.analysis.get_summary(chains="GChap")[r"$\alpha$"][1]-c.analysis.get_summary(chains="GChap")[r"$\alpha$"][0]

fig, ax = plt.subplots(1, 1)
ax.scatter(1-(om/(1-ok)), 0, marker = 'X', s = 60, c='red', label = r'$\Lambda$', zorder = 3)
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
c.plotter.plot_contour(ax, r'$A$',r"$\alpha$")
ax.text(0.91,-0.59 +0.3,r'$A = %10.5s^{%10.5s}_{%10.5s}$' %(GChap_A,GChap_Ap,GChap_Am), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0.91,-0.59,r'$\alpha = %10.5s^{%10.5s}_{%10.5s}$' %(GChap_a,GChap_ap,GChap_am), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(0.91,-0.59-0.3,r'$\Omega_{K} = %10.5s^{%10.5s}_{%10.5s}$' %(GChap_ok,GChap_okp,GChap_okm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$A$', fontsize = 18)
ax.set_ylabel(r"$\alpha$", fontsize = 18) 
ax.set_ylim(-1,2.0)
ax.set_xlim(0.5,0.92)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper left',frameon=False,fontsize=12)
plt.show()
#plt.close()
c.remove_chain('GChap')
exit()
### FCa PLOT 
c.add_chain(FCa_COBAYA_CHAIN[500:], parameters=[r"$\Omega_m$",r"$q$",r"$n$"], linewidth=2.0, name="FCa", kde=1.5, color="blue",num_free_params=3).configure(summary=True, shade_alpha=0.2,statistics="max")
FCa_om = c.analysis.get_summary(chains="FCa")[r"$\Omega_m$"][1]
FCa_omp = c.analysis.get_summary(chains="FCa")[r"$\Omega_m$"][2]-c.analysis.get_summary(chains="FCa")[r"$\Omega_m$"][1]
FCa_omm = c.analysis.get_summary(chains="FCa")[r"$\Omega_m$"][1]-c.analysis.get_summary(chains="FCa")[r"$\Omega_m$"][0]
FCa_q = c.analysis.get_summary(chains="FCa")[r"$q$"][1]
FCa_qp =c.analysis.get_summary(chains="FCa")[r"$q$"][2]-c.analysis.get_summary(chains="FCa")[r"$q$"][1]
FCa_qm =c.analysis.get_summary(chains="FCa")[r"$q$"][1]-c.analysis.get_summary(chains="FCa")[r"$q$"][0]
FCa_n = c.analysis.get_summary(chains="FCa")[r"$n$"][1]
FCa_np =c.analysis.get_summary(chains="FCa")[r"$n$"][2]-c.analysis.get_summary(chains="FCa")[r"$n$"][1]
FCa_nm =c.analysis.get_summary(chains="FCa")[r"$n$"][1]-c.analysis.get_summary(chains="FCa")[r"$n$"][0]

fig, ax = plt.subplots(1, 1)
ax.scatter(1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.axvline(1,color = 'k', ls = ':', linewidth=1, zorder = 1)
c.plotter.plot_contour(ax, r"$q$",r"$n$")
ax.text(2.9,-2.23,r'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(FCa_om,FCa_omp,FCa_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(2.9,-2.23-0.3,r'$q = %10.5s^{%10.5s}_{%10.5s}$' %(FCa_q,FCa_qp,FCa_qm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(2.9,-2.23-0.6,r'$n = %10.5s^{%10.5s}_{%10.5s}$' %(FCa_n,FCa_np,FCa_nm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r'$q$', fontsize = 18)
ax.set_ylabel(r"$n$", fontsize = 18) 
ax.set_ylim(-3,0.5)
ax.set_xlim(0.1,3)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper left',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('FCa')


### IDE1 PLOT
name = IDE1.__name__

# The actual chain this code is about
c.add_chain(IDE1_COBAYA_CHAIN[1000:], parameters=label_IDE1, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_IDE1)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
IDE1_om = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE1_omp = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE1_omm = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][0]
IDE1_ol = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE1_olp = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE1_olm = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][0]
IDE1_w = c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE1_wp =c.analysis.get_summary(chains=name)[r"$\omega$"][2]-c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE1_wm =c.analysis.get_summary(chains=name)[r"$\omega$"][1]-c.analysis.get_summary(chains=name)[r"$\omega$"][0]
IDE1_e = c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE1_ep =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][2]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE1_em =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][0]

fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax,r"$\omega$", r"$\varepsilon$") 

# Unique to model
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ax.scatter(-1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.text(0,-0.225,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(IDE1_om,IDE1_omp,IDE1_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(0,-0.225-0.075,r'$\Omega_x = %10.5s^{%10.5s}_{%10.5s}$' %(IDE1_ol,IDE1_olp,IDE1_olm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(0,-0.225-0.15,r'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(IDE1_w,IDE1_wp,IDE1_wm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(0,-0.225-0.225,r'$\varepsilon = %10.5s^{%10.5s}_{%10.5s}$' %(IDE1_e,IDE1_ep,IDE1_em), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r"$\omega$", fontsize = 18)
ax.set_ylabel(r"$\varepsilon$", fontsize = 18)

# Plot limits
ax.set_xlim(-2.0,0.05)
ax.set_ylim(-0.5,0.5)
#ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
#plt.show()
plt.close()
c.remove_chain('IDE1')


### IDE2 PLOT
name = IDE2.__name__

# The actual chain this code is about
c.add_chain(IDE2_COBAYA_CHAIN[2000:], parameters=label_IDE2, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_IDE1)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
IDE2_om = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE2_omp = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE2_omm = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][0]
IDE2_ol = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE2_olp = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE2_olm = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][0]
IDE2_w = c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE2_wp =c.analysis.get_summary(chains=name)[r"$\omega$"][2]-c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE2_wm =c.analysis.get_summary(chains=name)[r"$\omega$"][1]-c.analysis.get_summary(chains=name)[r"$\omega$"][0]
IDE2_e = c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE2_ep =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][2]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE2_em =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][0]

fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax,r"$\omega$", r"$\varepsilon$") 

# Unique to model
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ax.scatter(-1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.text(1.9,-0.225,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(IDE2_om,IDE2_omp,IDE2_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(1.9,-0.225-0.075,r'$\Omega_x = %10.5s^{%10.5s}_{%10.5s}$' %(IDE2_ol,IDE2_olp,IDE2_olm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(1.9,-0.225-0.15,r'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(IDE2_w,IDE2_wp,IDE2_wm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(1.9,-0.225-0.225,r'$\varepsilon = %10.5s^{%10.5s}_{%10.5s}$' %(IDE2_e,IDE2_ep,IDE2_em), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r"$\omega$", fontsize = 18)
ax.set_ylabel(r"$\varepsilon$", fontsize = 18)

# Plot limits
ax.set_xlim(-5.0,2)
ax.set_ylim(-0.5,0.5)
#ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()
#plt.close()
c.remove_chain('IDE2')


### IDE4 PLOT
name = IDE4.__name__

# The actual chain this code is about
c.add_chain(IDE4_COBAYA_CHAIN[500:], parameters=label_IDE4, linewidth=2.0, name=name, kde=1.5, color="blue",num_free_params=len(begin_IDE1)).configure(summary=True, shade_alpha=0.2,statistics="max")

# Gets best fit and plus/minus error 
IDE4_om = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE4_omp = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]
IDE4_omm = c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{m}$"][0]
IDE4_ol = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE4_olp = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][2]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]
IDE4_olm = c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][1]-c.analysis.get_summary(chains=name)[r"$\Omega_{x}$"][0]
IDE4_w = c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE4_wp =c.analysis.get_summary(chains=name)[r"$\omega$"][2]-c.analysis.get_summary(chains=name)[r"$\omega$"][1]
IDE4_wm =c.analysis.get_summary(chains=name)[r"$\omega$"][1]-c.analysis.get_summary(chains=name)[r"$\omega$"][0]
IDE4_e = c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE4_ep =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][2]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]
IDE4_em =c.analysis.get_summary(chains=name)[r"$\varepsilon$"][1]-c.analysis.get_summary(chains=name)[r"$\varepsilon$"][0]

fig, ax = plt.subplots(1, 1)

# Selectings which params to plot and on what axis
c.plotter.plot_contour(ax,r"$\omega$", r"$\varepsilon$") 

# Unique to model
ax.axhline(0,color = 'k', ls = ':', linewidth=1, zorder = 1)
ax.scatter(-1, 0, marker = 'x', s = 70, c='black', zorder = 3)
ax.text(1.9,-0.225,'$\Omega_m = %10.5s^{%10.5s}_{%10.5s}$' %(IDE4_om,IDE4_omp,IDE4_omm), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
ax.text(1.9,-0.225-0.075,r'$\Omega_x = %10.5s^{%10.5s}_{%10.5s}$' %(IDE4_ol,IDE4_olp,IDE4_olm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(1.9,-0.225-0.15,r'$\omega = %10.5s^{%10.5s}_{%10.5s}$' %(IDE4_w,IDE4_wp,IDE4_wm), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.text(1.9,-0.225-0.225,r'$\varepsilon = %10.5s^{%10.5s}_{%10.5s}$' %(IDE4_e,IDE4_ep,IDE4_em), family='serif',color='black',rotation=0,fontsize=12,ha='right')
ax.set_xlabel(r"$\omega$", fontsize = 18)
ax.set_ylabel(r"$\varepsilon$", fontsize = 18)

# Plot limits
ax.set_xlim(-5.0,2)
ax.set_ylim(-0.5,0.5)
#ax.set_xticklabels(['','',-1.5,'',-1.0,'',-0.5,'',''])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
ax.legend(loc='upper right',frameon=False,fontsize=12)
plt.show()
#plt.close()
c.remove_chain('IDE4')