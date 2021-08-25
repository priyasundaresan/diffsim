import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

results = np.load('results.npy')
ini_stiff, ini_grav, final_stiff, final_grav = results.T

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#sp = ax.scatter(data[:,2],data[:,3],data[:,0], s=20, c=data[:,1])
#plt.colorbar(sp)
#plt.show()

colors = (ini_stiff+ini_grav - np.min(ini_stiff+ini_grav))/np.ptp(ini_stiff+ini_grav)
cmap = 'Spectral'
plt.imshow((colors).reshape(1,len(ini_stiff)), cmap=cmap, aspect=2)
plt.title('Initial Guess Colormap: (stiffness multiplier, gravity multiplier)')
plt.yticks(np.arange(0))
plt.xticks(np.arange(0,len(ini_stiff),3), list(zip(ini_stiff,ini_grav))[::3])
plt.savefig('colormap.png')
plt.clf()
#plt.show()

plt.scatter(final_stiff, final_grav, c=ini_stiff+ini_grav, cmap=cmap)
plt.xlabel("Final Estimated Stiffness Multiplier")
plt.ylabel("Final Estimated Gravity Multiplier")
plt.title("Est. Cloth Params as a Function of Initial Guesses")
plt.savefig('cloth_params.png')
#plt.show()

#
## text is left-aligned
#for i,(x,y) in enumerate(zip(final_stiff, final_grav)):
#    plt.annotate('(%.1f, %.1f)'%(ini_stiff[i], ini_grav[i]), # this is the text
#             (x,y), # these are the coordinates to position the label
#             textcoords="offset points", # how to position the text
#             xytext=(0,10), # distance from text to points (x,y)
#             ha='center') # horizontal alignment can be left, right or center
#    #plt.text(x,y,'(%.1f, %.1f)'%(ini_stiff[i], ini_grav[i]))
#plt.show()


