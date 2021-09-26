import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

cov_arr = np.genfromtxt("FITOPT000_MUOPT006.COV",comments='#',dtype=None, skip_header=1)
cov = cov_arr.reshape(20,20)

#fig, ax =plt.subplots()
sns.heatmap(cov)
plt.xticks([])
plt.yticks([])
plt.show()