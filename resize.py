"""
Adds padding to images to be 400x400
Author: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
with minor adjustments by Nadine Adnane & Tanjuma Haque
Date: 12/7/19
"""

from PIL import Image, ImageOps
import cv2
from imutils import paths

desired_size = 400
base_path = "malaria/cell_images"
image_paths = list(paths.list_images(base_path))

for im_pth in image_paths:

    im = Image.open(im_pth)
    old_size = im.size  

    ratio = float(desired_size)/max(old_size)

    new_size = tuple([int(x) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))

    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    print(padding)
    new_im = ImageOps.expand(im, padding, fill="black")
    new_im.save(im_pth)

print("All done!~ ^-^")