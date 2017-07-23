import numpy as np
import os
from PIL import Image
import scipy.misc as im

input_dir = "./rotate_img/"
output_dir = "./rotate_processed/"
except_dir = "./Exception/"
crop_pixel = 5

if (os.path.exists(input_dir) == False):
    print("No input directory")

try:
    os.stat(output_dir)
    os.stat(except_dir)
except:
    os.mkdir(output_dir)
    os.mkdir(except_dir)

file_list = os.listdir(input_dir)

cnt_success = 0
cnt_fail = 0

for file in file_list:
    img = Image.open(os.path.join(input_dir, file))
    s = np.shape(img)

    if s == (100, 100, 3):
        region = (0 + crop_pixel, 0 + crop_pixel, s[0] - crop_pixel, s[1] - crop_pixel)
        img = img.crop(region)
        im.imsave(output_dir + file.title(), img)
        cnt_success += 1
    else:
        im.imsave(except_dir + file.title(), img)
        cnt_fail += 1

print("Done")

print("success={}, fail={}".format(cnt_success, cnt_fail))
