import numpy as np
import idx2numpy
import Constants
import csv
import math

# ================ HYPER PARAMETERS ================
# Right at the top for easy access and scope
INPUT_NODE_COUNT = 784
OUTPUT_NODE_COUNT = 10
HIDDEN_NODE_COUNT = 16
HIDDEN_LAYER_COUNT = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.001


# ================ LOAD THE DATASET ================

train = idx2numpy.convert_from_file("./MNIST_datasets/train-images-idx3-ubyte")

train_labels = idx2numpy.convert_from_file("./MNIST_datasets/train-labels-idx1-ubyte")

test = idx2numpy.convert_from_file("./MNIST_datasets/t10k-images-idx3-ubyte")

test_labels = idx2numpy.convert_from_file("./MNIST_datasets/t10k-labels-idx1-ubyte")

# Resize dataset into n by 784, and convert values from 0-255 to 0-1
train = (train.reshape(60000, -1).astype('float')) /255
test = (test.reshape(10000, -1).astype('float')) /255

# One hot encode the labels
def one_hot_encode(x: np.ndarray):
        ret_arr = np.zeros((x.size, OUTPUT_NODE_COUNT))
        ret_arr[np.arange(x.size), x] = 1
        return ret_arr

train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)


# ================= CREATE THE MLP =================

class MLP:
    
    def __init__(self):
        pass
    
    def reLu(self, x: int):
        x if x > 0 else 0
        
    '''
    Softmax takes in a 2-d array of shape (num_samples, num_classes) and performs
    softmax on the matrix row-by-row
    '''
    def soft_max(self, x: np.ndarray):
        x_max = np.max(x, axis=1, keepdims=True) # keepdims gives us (n,1) instead of (n,)
        e_x = np.exp(x - x_max) # subtract x_max here to prevent overflow / preserve numerical stability (does not affect the softmax math)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
        

    
    