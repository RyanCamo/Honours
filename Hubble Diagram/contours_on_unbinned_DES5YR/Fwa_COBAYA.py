from getdist import plots

analysis_settings = {'ignore_rows': '500'}
g=plots.get_subplot_plotter(chain_dir=r'/Users/RyanCamo/Desktop', analysis_settings=analysis_settings)
roots = ['Fwa_DES5YR_BIN']
params = ['om','w0', 'wa']
g.triangle_plot(roots, params, filled=True,title_limit=1)
g.export()
