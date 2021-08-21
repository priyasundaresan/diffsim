import glob
from PIL import Image
import os

#os.system('ffmpeg -i video.mp4 images/%05d.jpg')
os.system('cd images && gifski --fps 8 -o clip.gif *.png && cd ..')

#fp_in = "images/*.png"
#fp_out = "images/rollout.gif"
#img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
#img.save(fp=fp_out, format='GIF', append_images=imgs,
#         save_all=True, duration=600, loop=0)
