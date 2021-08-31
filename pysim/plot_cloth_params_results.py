import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

results = np.load('results.npy')
ini_p1, ini_p2, final_p1, final_p2 = results.T

param1_lab = 'Stiffness'
param2_lab = 'Mass'

#colors = (ini_p1+ini_p2 - np.min(ini_p1+ini_p2))/np.ptp(ini_p1+ini_p2)
#cmap = 'Spectral'
#plt.imshow((colors).reshape(1,len(ini_p1)), cmap=cmap, aspect=2)
#plt.title('Initial Guess Colormap: (stiffness multiplier, gravity multiplier)')
#plt.yticks(np.arange(0))
#plt.xticks(np.arange(0,len(ini_p1),3), list(zip(ini_p1,ini_p2))[::3])
#plt.savefig('colormap.png')
#plt.clf()
##plt.show()
#
#plt.scatter(final_p1, final_p2, c=ini_p1+ini_p2, cmap=cmap)
#plt.xlabel("Final Estimated %s Multiplier"%param1_lab)
#plt.ylabel("Final Estimated %s Multiplier"%param2_lab)
#plt.title("Est. Cloth Params as a Function of Initial Guesses")
#plt.savefig('cloth_params.png')
##plt.show()

def normalize(arr):
    return (arr - np.min(arr))/np.ptp(arr)

fig, ax = plt.subplots()
ax.set_aspect("equal")

ini_p1_norm = normalize(ini_p1)
ini_p2_norm = normalize(ini_p2)

cmap = lambda p1,p2 : (p1, 0, p2)

for i in range(len(final_p1)):
    circle = plt.Circle((final_p1[i], final_p2[i]), 0.01, color=cmap(ini_p1_norm[i],ini_p2_norm[i]))
    ax.add_artist(circle)
ax.set_xlabel("Final Estimated %s Multiplier"%param1_lab)
ax.set_ylabel("Final Estimated %s Multiplier"%param2_lab)
ax.set_xlim(0,np.amax(final_p1)+(np.amax(final_p1) - np.amin(final_p1))/10)
ax.set_ylim(0,np.amax(final_p2)+(np.amax(final_p2) - np.amin(final_p2))/10)

plt.title("Est. Cloth Params vs. Initial Guesses")
plt.subplots_adjust(left=0.1, right=0.65, top=0.85)
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
cax.set_xticklabels(np.around(np.linspace(ini_p1[0], ini_p1[-1], 3),2))
cax.set_yticklabels(np.around(np.linspace(ini_p2[0], ini_p2[-1], 6),2))
cax.set_ylabel("p2: %s"%param2_lab.lower())
cax.set_title("Initial Guess Legend", fontsize=10)

plt.savefig('cloth_params.png')
plt.show()
