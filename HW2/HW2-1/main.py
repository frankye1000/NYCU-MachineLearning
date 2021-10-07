import numpy as np
import discrete_, continue_
from loadMnist import load_images, load_labels


train_data  = load_images("data/train-images.idx3-ubyte")
train_label = load_labels("data/train-labels.idx1-ubyte")
test_data   = load_images("data/t10k-images.idx3-ubyte")
test_label  = load_labels("data/t10k-labels.idx1-ubyte")
            

option = input('Toggle option (0:discrete / 1:continuous): ')

if option == "0":
    discrete_.run(train_data, train_label, test_data, test_label)
else:
    continue_.run(train_data, train_label, test_data, test_label)
