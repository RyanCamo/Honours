from scipy.stats import multivariate_normal
import numpy as np
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt


c = ChainConsumer()
for i in range(1000):
   # Generate some data centered at a random location with uncertainty
   # equal to the scatter
   mean = [3, 8]
   cov = [[1.0, 0.5], [0.5, 2.0]]
   mean_scattered = multivariate_normal.rvs(mean=mean, cov=cov)
   data = multivariate_normal.rvs(mean=mean_scattered, cov=cov, size=20)
   posterior = multivariate_normal.logpdf(data, mean=mean_scattered, cov=cov)
   print(posterior)
   print(data)
   c.add_chain(data, posterior=posterior, parameters=["$x$", "$y$"], color='r', name="Simulation validation")
fig = c.plotter.plot()
plt.show()