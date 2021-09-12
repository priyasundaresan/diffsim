import numpy as np
import matplotlib.pyplot as plt
import pprint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable

results = np.load('results.npy')
ini_p1, ini_p2, final_p1, final_p2, loss = results.T

param1_lab = 'Stiffness'
#param2_lab = 'Mass'
param2_lab = 'Gravity'

def normalize(arr):
    return (arr - np.min(arr))/np.ptp(arr)

fig, axes = plt.subplots(2, figsize=(8, 7))
ax = axes[0]
ax2 = axes[1]
#ax.set_aspect("equal")

ini_p1_norm = normalize(ini_p1)
ini_p2_norm = normalize(ini_p2)

cmap = lambda p1,p2 : (p1, 0, p2)
cmap_loss = plt.get_cmap('RdYlGn_r')

norm = plt.Normalize(loss.min(), loss.max())

loss_min, loss_max = np.amin(loss), np.amax(loss)
for i in range(len(final_p1)):
    ax.scatter(final_p1[i], final_p2[i], color=cmap(ini_p1_norm[i],ini_p2_norm[i]))
    #sc = ax2.scatter(final_p1[i], final_p2[i], color=plt.get_cmap('RdYlGn')(1-(loss[i] - loss_min)/(loss_max - loss_min)))
    sc = ax2.scatter(final_p1[i], final_p2[i], color=cmap_loss(norm(loss[i])))
ax.set_xlabel("Final Estimated %s Multiplier"%param1_lab)
ax.set_ylabel("Final Estimated %s Multiplier"%param2_lab)
ax.set_xlim(0,np.amax(final_p1)+(np.amax(final_p1) - np.amin(final_p1))/10)
ax.set_ylim(0,np.amax(final_p2)+(np.amax(final_p2) - np.amin(final_p2))/10)
ax2.set_xlim(0,np.amax(final_p1)+(np.amax(final_p1) - np.amin(final_p1))/10)
ax2.set_ylim(0,np.amax(final_p2)+(np.amax(final_p2) - np.amin(final_p2))/10)

sm =  ScalarMappable(norm=norm, cmap=cmap_loss)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2)
cbar.ax.set_title("Loss", fontsize=10)

fig.suptitle("Est. Cloth Params vs. Initial Guesses")
plt.subplots_adjust(left=0.1, right=0.625, top=0.9)
cax = fig.add_axes([0.7,0.55,0.3,0.3])
cp1 = np.linspace(0,1)
cp2 = np.linspace(0,1)
Cp1, Cp2 = np.meshgrid(cp1,cp2)
C0 = np.zeros_like(Cp1)
# make RGB image, p1 to red channel, p2 to blue channel
Legend = np.dstack((Cp1, C0, Cp2))
# parameters range between 0 and 1
cax.imshow(Legend, origin="lower", extent=[0,1,0,1])
cax.set_xlabel("p1: %s"%param1_lab.lower())
cax.set_xticklabels(np.around(np.linspace(ini_p1[0], ini_p1[-1], 6),2))
cax.set_yticklabels(np.around(np.linspace(ini_p2[0], ini_p2[-1], 6),2))
cax.set_ylabel("p2: %s"%param2_lab.lower())
cax.set_title("Initial Guess Legend", fontsize=10)

plt.savefig('cloth_params.png')
plt.show()
