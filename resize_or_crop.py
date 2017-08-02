import numpy as np
import scipy.misc as im
import os
import argparse
import math
import PIL.Image as Image
import time

# pass input and output directory via command line
parser = argparse.ArgumentParser(prog="pre_process")
parser.add_argument("--input", default="./input/")
parser.add_argument("--output", default="./output/")
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--resize", type=tuple, default=(90, 90))
parser.add_argument("--method", default="crop")

# use vars() to get all the args (cannot directly get them via parse_args())
args = vars(parser.parse_args())
input_dir = args["input"]
output_dir = args["output"]
batch_size = args["batch_size"]
resize = args["resize"]
method = args["method"]
except_dir = "./except/"

if (os.path.exists(input_dir) == False):
    print("No input directory")
    exit()
try:
    os.stat(output_dir)

except:
    os.mkdir(output_dir)

try:
    os.stat(except_dir)

except:
    os.mkdir(except_dir)

file_list = os.listdir(input_dir)
batch_num = math.ceil(file_list.__len__() / batch_size)

print("{} batch files will be generate.".format(int(batch_num)))

batch_file = []
batch_id = 0

t1 = time.time()

if method == "resize":
    # resize method
    for file in file_list:

        img = im.imread(os.path.join(input_dir, file))
        img = im.imresize(img, resize)
        # save image file
        # im.imsave(output_dir + file.title(), img)
        batch_file.append(img)

        if batch_file.__len__() % batch_size == 0:
            np.save(output_dir + "batch_data" + str(batch_id), batch_file)
            print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))
            batch_id += 1
            batch_file = []
            t1 = time.time()
    if batch_file.__len__() != 0:
        np.save(output_dir + "batch_data" + str(batch_id), batch_file)
        print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))

######################################################################################
elif method == "crop":
    # crop method

    w_target, h_target = resize
    for file in file_list:

        img = Image.open(os.path.join(input_dir, file))
        w, h = img.size

        if w < w_target or h < h_target:
            img.save(except_dir + file.title())
            continue

        region = (
            (w - w_target) / 2, (h - h_target) / 2, (w + w_target) / 2, (h + h_target) / 2)

        img = img.crop(region)
        batch_file.append(np.asarray(img))

        if batch_file.__len__() % batch_size == 0:
            np.save(output_dir + "batch_data" + str(batch_id), batch_file)
            print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))
            batch_id += 1
            batch_file = []
            t1 = time.time()

    if batch_file.__len__() != 0:
        np.save(output_dir + "batch_data" + str(batch_id), batch_file)
        print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))

print("done")
