from getdist import plots

analysis_settings = {'ignore_rows': '1000'}
g=plots.get_subplot_plotter(chain_dir=r'/Users/RyanCamo/Desktop', analysis_settings=analysis_settings)
roots = ['IDE1_DES5YR_BIN']
params = ['cdm', 'ol', 'w', 'e']
g.triangle_plot(roots, params, filled=True,title_limit=1)
g.export()
