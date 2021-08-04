import glob
from PIL import Image

fp_in = "images/*.jpg"
fp_out = "images/rollout.gif"
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0)
