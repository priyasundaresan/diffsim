import glob
from PIL import Image
import os

#os.system('ffmpeg -i video.mp4 images/%05d.jpg')
os.system('cd images && gifski -o clip.gif *.png && cd ..')

#fp_in = "images/*.jpg"
#fp_out = "images/rollout.gif"
#img, *imgs = [Image.open(f).convert('P') for f in sorted(glob.glob(fp_in))]
#img.save(fp=fp_out, format='GIF', append_images=imgs,
#         save_all=True, duration=50, loop=0)
