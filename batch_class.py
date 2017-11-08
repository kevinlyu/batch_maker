import numpy as np
import os
import math
import time
from PIL import Image
import scipy.misc as im


class BatchMaker:
    def __init__(self, input_dir, output_dir, batch_size=300, output_size=(100, 100), mode="resize"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.mode = mode
        self.output_size = output_size


    def process(self):

        file_list = os.listdir(self.input_dir)
        batch_num = math.ceil(file_list.__len__() / self.batch_size)
        print("{} batch files will be generate.".format(int(batch_num)))

        batch_file = []
        batch_id = 0
        t1 = time.time()

        if self.mode is "resize":
            for file in file_list:

                img = im.imread(os.path.join(self.input_dir, file))
                img = im.imresize(img, self.output_size)
                batch_file.append(img)

                if batch_file.__len__() % self.batch_size == 0:
                    np.save(self.output_dir + "batch" + str(batch_id), batch_file)
                    print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))
                    batch_id += 1
                    batch_file = []
                    t1 = time.time()

            if batch_file.__len__() != 0:
                np.save(self.output_dir + "batch" + str(batch_id), batch_file)
            print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))

        elif self.mode is "crop":
            w_target, h_target = self.output_size
            for file in file_list:

                img = Image.open(os.path.join(self.input_dir, file))
                w, h = img.size

                if w < w_target or h < h_target:
                    print("size error")
                    continue

                region = (
                    (w - w_target) / 2, (h - h_target) / 2, (w + w_target) / 2, (h + h_target) / 2)

                img = img.crop(region)
                batch_file.append(np.asarray(img))

                if batch_file.__len__() % self.batch_size == 0:
                    np.save(self.output_dir + "batch" + str(batch_id), batch_file)
                    print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))
                    batch_id += 1
                    batch_file = []
                    t1 = time.time()

            if batch_file.__len__() != 0:
                np.save(self.output_dir + "batch" + str(batch_id), batch_file)

            print("batch {} is saved, spent {:.3f} s".format(batch_id, time.time() - t1))
        else:
            print("Please set mode to 'resize' or 'crop' ")
