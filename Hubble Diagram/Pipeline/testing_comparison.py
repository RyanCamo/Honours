from scipy.stats import norm
from chainconsumer import ChainConsumer
import numpy as np

n = 10000
d1 = norm.rvs(size=n)
p1 = norm.logpdf(d1)
#print(np.max(p1))
#print(p1)
AIC1 = -2*np.max(p1) + 2*4
#print(AIC)
BIC1 = -2*np.max(p1) + 4*np.log(n)
#rint(BIC)
p2 = norm.logpdf(d1, scale=1.1)

AIC2 = -2*np.max(p1) + 2*14
BIC2 = -2*np.max(p1) + 14*np.log(n)

delAIC = AIC2-AIC1
delBIC = BIC2-BIC1
print(delAIC)
print(delBIC)


c = ChainConsumer()
c.add_chain(d1, posterior=p1, name="Model A", num_eff_data_points=n, num_free_params=4)
c.add_chain(d1, posterior=p2, name="Model B", num_eff_data_points=n, num_free_params=5)
c.add_chain(d1, posterior=p2, name="Model C", num_eff_data_points=n, num_free_params=4)
c.add_chain(d1, posterior=p1, name="Model D", num_eff_data_points=n, num_free_params=14)
table = c.comparison.comparison_table(caption="Model comparisons!")
print(table)


