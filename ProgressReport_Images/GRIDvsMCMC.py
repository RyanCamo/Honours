import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 2)
theta = np.radians(np.linspace(0,360*5,1000))
var = np.random.normal(scale = 1, size = 1000)
test =-1000 + 2000*np.random.random((89,89))
print(test)
theta1 = theta + var
r0 = theta**2
x_0 = r0*np.cos(theta)
y_0 = r0*np.sin(theta)

r1 = theta1**2
x_1 = r1*np.cos(theta)
y_1 = r1*np.sin(theta)

#plt.figure()
# MCMC
ax[1].plot(x_0,y_0, linewidth =3)
ax[1].scatter(x_1,y_1,s=6, color='k')
ax[1].scatter(test[0],test[1],s=6, color='k')
ax[1].set_xlim(-1100,1100)
ax[1].set_ylim(-1100,1100)
ax[1].set_yticks([])
ax[1].set_xticks([])
ax[1].set_aspect(1.0)

# Grid
x, y = np.meshgrid(np.linspace(-1000,1000, 33), np.linspace(-1000, 1000, 33))
ax[0].scatter(x,y,s=6, color='k')
ax[0].plot(x_0,y_0,linewidth =3)
ax[0].set_xlim(-1100,1100)
ax[0].set_ylim(-1100,1100)
ax[0].set_yticks([])
ax[0].set_xticks([])
ax[0].set_aspect(1.0)
plt.show()