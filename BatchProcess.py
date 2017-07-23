import numpy as np
import os
from PIL import Image
import math
import time
import scipy.misc as im

class BatchMaker:
    def __init__(self, input_dir="./img", output_dir="./processed", batch_size=100):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.t1 = None

    def process(self):

        if (os.path.exists(self.input_dir) == False):
            print("No input directory")

        try:
            os.stat(self.output_dir)
        except:
            os.mkdir(self.output_dir)

        file_list = os.listdir(self.input_dir)
        batch_num = math.ceil(file_list.__len__() / self.batch_size)

        batch_file = []

        cnt = 0

        print("[Start creating batches, {} batch files will be created]".format(int(batch_num)))
        self.t1 = time.time()
        for file in file_list:

            img = Image.open(os.path.join(self.input_dir, file))
            img = np.array(img)
            batch_file.append(img)
            cnt += 1

            if (cnt % self.batch_size == 0):
                np.save(self.output_dir + "/batch_data" + str(int(cnt / self.batch_size)), batch_file)

                print("batch data{} is saved, spent {:.3f}s".format(str(int(cnt / self.batch_size)),
                                                                    time.time() - self.t1))
                batch_file = []
                self.t1 = time.time()

        # data left, will be saved this line
        np.save(self.output_dir + "/batch_data" + str(int(cnt / self.batch_size) + 1), batch_file)
        print("batch data{} is saved, spent {:.3f}s".format(int(cnt / self.batch_size) + 1, time.time() - self.t1))

        print("[Done]")

    def process_with_reshape(self, reshape=(100, 100)):

        if (os.path.exists(self.input_dir) == False):
            print("No input directory")

        try:
            os.stat(self.output_dir)
        except:
            os.mkdir(self.output_dir)

        file_list = os.listdir(self.input_dir)
        batch_num = math.ceil(file_list.__len__() / self.batch_size)

        batch_file = []

        cnt = 0

        print("[Start creating batches, {} batch files will be created]".format(int(batch_num)))
        self.t1 = time.time()
        for file in file_list:

            img = Image.open(os.path.join(self.input_dir, file))
            img = np.array(img)
            img = im.imresize(img, reshape)
            batch_file.append(img)
            cnt += 1

            if (cnt % self.batch_size == 0):
                np.save(self.output_dir + "/batch_data" + str(int(cnt / self.batch_size)), batch_file)

                print("batch data{} is saved, spent {:.3f}s".format(str(int(cnt / self.batch_size)),
                                                                    time.time() - self.t1))
                batch_file = []
                self.t1 = time.time()

        # data left, will be saved this line
        np.save(self.output_dir + "/batch_data" + str(int(cnt / self.batch_size) + 1), batch_file)
        print("batch data{} is saved, spent {:.3f}s".format(int(cnt / self.batch_size) + 1, time.time() - self.t1))

        print("[Done]")


class BatchReader:
    def __init__(self, input_dir="./processed"):
        self.input_dir = input_dir

    def get_batch(self, selected_batch=[1]):
        print("[Start loading batch files]")

        batch = []
        for id in selected_batch:

            if (not os.path.exists(self.input_dir + "/batch_data" + str(id) + ".npy")):
                print("batch_data" + str(id) + " does not exist.")
                continue

            data = np.load(self.input_dir + "/batch_data" + str(id) + ".npy")
            batch.extend(data)
            print("batch data{} loaded. contains {} data".format(id, np.shape(batch)[0]))

        print("[Done]")
        return batch
