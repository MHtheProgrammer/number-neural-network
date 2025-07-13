import numpy as np
import idx2numpy
import Constants
import csv
import math

# ================ HYPER PARAMETERS ================
# Right at the top for easy access and scope
LAYER_SIZES = [784, 16, 16, 10] ## The num of nodes in each layer, first is input last is output
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
def one_hot_encode(x: np.ndarray, num_classes: int):
        ret_arr = np.zeros((x.size, num_classes))
        ret_arr[np.arange(x.size), x] = 1
        return ret_arr

train_labels = one_hot_encode(train_labels, LAYER_SIZES[-1])
test_labels = one_hot_encode(test_labels, LAYER_SIZES[-1])


# ================= CREATE THE MLP =================

'''
Takes an array and returns an array where negative numbers are replaced with 0.
EG: ReLU activation function
'''
def relu(x: np.ndarray):
    return np.maximum(0, x)
    
'''
Takes an array and returns an array of same shape where each element is 1 or 0 based on condition Xi > 0
'''
def relu_derivative(x: np.ndarray):
    return x > 0 # Boolean mask, array of same shape with True False values which are same as 1s and 0s

'''
Softmax takes in a 2-d array of shape (num_samples, num_classes) and performs
softmax on the matrix row-by-row
'''
def soft_max(x: np.ndarray):
    x_max = np.max(x, axis=1, keepdims=True) ## keepdims gives us (n,1) instead of (n,)
    e_x = np.exp(x - x_max) ## subtract x_max here to prevent overflow / preserve numerical stability (does not affect the softmax math)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

'''
Initializes model parameters into a dict where the weights are indexed via
Wi, where i is the output layer number. Weights between input and 1st hidden layer is W1, and output layer is Wn. No W0
bi is biases with i referring to the layer
HE normal initialization used for weights, 0s for biases
'''
def initialize_model_parameters(layer_sizes: list):
    parameters = {}
    
    for l in range (1, layer_sizes): ## Every layer needs weights and biases except input
        input_size = layer_sizes[l-1]
        output_size = layer_sizes[l]
        parameters[f'W{l}'] = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size) ## HE normal initialization
        parameters[f'b{l}'] = np.zeros((output_size, 1)) ## 0s for the biases
        
    return parameters

'''
Take the input matrix of shape (input_size, num_samples), and the parameters dict
Return y-hat of size (num_classes, num_samples), and cache which stores intermediate
values in form Zi and Ai, where the 1st hidden layer is Z1 and A1 (no Z0 but there is A0 for inputs)
'''
def forward_propagation(X: np.ndarray, parameters: dict, layer_sizes: list):
    A = X
    cache = {'A0': A}
    L = len(layer_sizes)
    
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = W @ A + b
        A = relu(Z) if l < (L-1) else soft_max(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    return A, cache
    
'''
Takes softmaxxed Y_hat shape (num_classes, num_samples) and Y one hot encoded of same shape
Computes cross-entropy loss
Returns scalar loss value, averaged over num_samples
'''
def compute_loss(Y_hat: np.ndarray, Y: np.ndarray):
    num_samples = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / num_samples ## Add 1e-8 so we don't log(0) which is undefined
    return loss

'''
Takes as parameters
    X: Input data of shape (flattened_input_size, num_samples)
    Y: Correct output one hot encoded of shape (num_classes, num_samples)
    parameters: dictionary holding the weights and biases (W1, b1, W2, b2, ...)
    cache: dictionary holding the activations and pre-activations from forward pass (A1, Z1, ...)
    layer_sizes: list of the layer sizes
This function runs backpropagation to calculate the gradients for weights and biases
It DOES NOT update the weights and biases via gradient descent.
Returns a dict containing the gradients as {dW1, db1, dW2, ...}
'''
def backward_propagation(X: np.ndarray, Y: np.ndarray, parameters: dict, cache: dict, layer_sizes: list):
    gradients = {}
    L = len(layer_sizes)
    num_samples = X.shape[1]
    Y_hat = cache[f'A{L-1}']
    
    dA = Y_hat - Y ## Derivative dL/dA for output nodes
    
    # Loop through the layers starting from the end and going backwards
    for l in reversed(range(1, L)):
        A_prev = cache[f'A{l-1}']
        Z = cache[f'Z{l}']
        W = parameters[f'W{l}']
        
        if l == L-1:
            dZ = dA ## For softmax the derivative is already included in Y_hat - Y
        else:
            dZ = dA * relu_derivative(Z)
            
        # At this point dZ contains dL/dZ = dL/dA * dA/dZ. shape (nodes_in_layer, num_samples)
        gradients[f'dW{l}'] = (dZ @ A_prev.T) / num_samples ## dZ/dW = A_prev. shape (nodes_in_layer, num_samples)
        gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / num_samples ## dZ/db = 1, so just sum dZ across samples
        dA = W.T @ dZ ## Update dL/dA for the next layer
        
    return gradients
