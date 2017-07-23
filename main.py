from BatchProcess import *
import matplotlib.pyplot as plt


m = BatchMaker(input_dir="./img/", batch_size=20)
m.process_with_reshape(reshape=(100, 100))

r = BatchReader()
data = r.get_batch(selected_batch=[1])

plt.imshow(data[10])
plt.show()
