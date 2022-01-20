from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

fig, axes = plt.subplots()
plt.minorticks_on()
axes.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")
#axes.xaxis.set_minor_locator(AutoMinorLocator())
#axes.yaxis.set_minor_locator(AutoMinorLocator())
axes.legend()
plt.show()