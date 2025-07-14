import numpy as np
import idx2numpy
import json

# ================ HYPER PARAMETERS ================
# Right at the top for easy access
LAYER_SIZES = [784, 16, 16, 10] ## The num of nodes in each layer, first is input last is output
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EXPORT_FILE_PATH = "./model_parameters.json"

# ================ LOAD THE DATASET ================

train_data = idx2numpy.convert_from_file("./MNIST_datasets/train-images-idx3-ubyte")

train_labels = idx2numpy.convert_from_file("./MNIST_datasets/train-labels-idx1-ubyte")

test_data = idx2numpy.convert_from_file("./MNIST_datasets/t10k-images-idx3-ubyte")

test_labels = idx2numpy.convert_from_file("./MNIST_datasets/t10k-labels-idx1-ubyte")

# Resize dataset into n by 784, and convert values from 0-255 to 0-1
train_data = (train_data.reshape(60000, -1).astype('float')) /255
test_data = (test_data.reshape(10000, -1).astype('float')) /255

# One hot encode the labels
'''
Takes array of labels ex: [1, 3, 5, ...] and one hot encodes them -> [[0, 1, 0, ...], [0, 0, 0, 1, 0, ...], [...], ...]
'''
def one_hot_encode(x: np.ndarray, num_classes: int):
        ret_arr = np.zeros((x.size, num_classes))
        ret_arr[np.arange(x.size), x] = 1
        return ret_arr

train_labels = one_hot_encode(train_labels, LAYER_SIZES[-1])


# ================= CREATE THE MLP =================

def relu(x: np.ndarray):
    '''
    Takes an array and returns an array where negative numbers are replaced with 0.
    EG: ReLU activation function
    '''
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
    '''
    Takes an array and returns an array of same shape where each element is 1 or 0 based on condition Xi > 0
    '''
    return x > 0 # Boolean mask, array of same shape with True False values which are same as 1s and 0s

def soft_max(x: np.ndarray):
    '''
    Softmax takes in a 2-d array of shape (num_samples, num_classes) and performs
    softmax on the matrix row-by-row
    '''
    x_max = np.max(x, axis=1, keepdims=True) ## keepdims gives us (n,1) instead of (n,)
    e_x = np.exp(x - x_max) ## subtract x_max here to prevent overflow / preserve numerical stability (does not affect the softmax math)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def initialize_model_parameters(layer_sizes: list):
    '''
    Initializes model parameters into a dict where the weights are indexed via
    Wi, where i is the output layer number. Weights between input and 1st hidden layer is W1, and output layer is Wn. No W0
    bi is biases with i referring to the layer
    HE normal initialization used for weights, 0s for biases
    '''
    parameters = {}
    
    for l in range (1, layer_sizes): ## Every layer needs weights and biases except input
        input_size = layer_sizes[l-1]
        output_size = layer_sizes[l]
        parameters[f'W{l}'] = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size) ## HE normal initialization
        parameters[f'b{l}'] = np.zeros((output_size, 1)) ## 0s for the biases
        
    return parameters

def forward_propagation(X: np.ndarray, parameters: dict, layer_sizes: list):
    '''
    Take the input matrix of shape (input_size, num_samples), and the parameters dict
    Return y-hat of size (num_classes, num_samples), and cache which stores intermediate
    values in form Zi and Ai, where the 1st hidden layer is Z1 and A1 (no Z0 but there is A0 for inputs)
    '''
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

def compute_loss(Y_hat: np.ndarray, Y: np.ndarray):
    '''
    Takes softmaxxed Y_hat shape (num_classes, num_samples) and Y one hot encoded of same shape
    Computes cross-entropy loss
    Returns scalar loss value, averaged over num_samples
    '''
    num_samples = Y.shape[1]
    loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / num_samples ## Add 1e-8 so we don't log(0) which is undefined
    return loss

def backward_propagation(X: np.ndarray, Y: np.ndarray, parameters: dict, cache: dict, layer_sizes: list):
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

def update_parameters(parameters: dict, gradients: dict, learning_rate: float):
    '''
    Takes parameters {W1, b1, etc}, gradients {dW1, db1, etc}, and a learning rate
    Updates the model parameters by moving them in negative direction of the gradient * learning_rate
    '''
    for key in parameters:
        parameters[key] -= learning_rate * gradients['d' + key] ## EX: W1 -= learning_rate * dW1
    return parameters

def run_gradient_descent(X: np.ndarray, Y: np.ndarray, parameters: dict, layer_sizes: list, epochs: int, batch_size: int, learning_rate: float):
    '''
    Takes the following arguments
    X: the training data flattened out into 2d matrix
    Y: labels for training data one hot encoded
    parameters: dict of model parameters {W1, b1, etc}
    layer_sizes: list of layer sizes
    epochs: number of times to run through training data
    batch_size: num of samples to use for each iteration of learning
    learning_rate: constant value for learning rate

    This function will take the parameters and run gradient descent to start tweaking the parameters.
    Returns the updated parameters dict
    '''
    num_samples = X.shape[1]
    
    for epoch in range(epochs):
        # Randomize order so we get different batches every epoch
        random_order = np.random.permutation(num_samples)
        X_shuffled = X[:, random_order]
        Y_shuffled = Y[:, random_order]
        
        # Split data into batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_shuffled[:, batch_start:batch_end]
            Y_batch = Y_shuffled[:, batch_start:batch_end]
            
            # Run forward and backwards propagation, then take a step down the gradient
            Y_hat, cache = forward_propagation(X_batch, parameters, layer_sizes)
            gradients = backward_propagation(X_batch, Y_batch, parameters, cache, layer_sizes)
            parameters = update_parameters(parameters, gradients, learning_rate)
            
            # Compute and print the loss to console so we can see if its learning (debug purposes)
            loss = compute_loss(Y_hat, Y_batch)
            print(f'Loss on epoch {epoch} batch {batch_start/batch_size}= {loss}')
            
    return parameters

def predict(X: np.ndarray, parameters: dict, layer_sizes: list):
    '''
    Given a matrix of input size (num_samples, flattened_input_size) run predictions for every sample
    Returns an array of predictions NOT one hot, eg [1, 3, 5, 2, 9, etc]
    '''
    Y_hat, _ = forward_propagation(X, parameters, layer_sizes)
    return np.argmax(Y_hat, axis=0)

def get_accuracy(y_hat, y):
    '''
    Given y_hat and y, two arrays of ints. Return the accuracy of the predictions
    '''
    return np.mean(y_hat == y) ## Create an array of True/False (1, 0) then get the mean of that array which is same as successes/total

def export_parameters_to_file(parameters: dict, filepath: str):
    """
    Writes the parameters dict {W1, b1, etc} to a file as a JSON.
    """
    parameters_as_lists = {key: value.tolist() for key, value in parameters.items()}
    with open(filepath, 'w') as file:
        json.dump(parameters_as_lists, file)
            
            
# ================ RUN THE TRAINING ================

# Create and tweak our model parameters
parameters = initialize_model_parameters(LAYER_SIZES)
parameters = run_gradient_descent(train_data, train_labels, parameters, LAYER_SIZES, EPOCHS, BATCH_SIZE, LEARNING_RATE)

# Test the model on the test data
predictions = predict(test_data, parameters, LAYER_SIZES)
accuracy = get_accuracy(predictions, test_labels)
print(f'Model Accuracy: {accuracy}')

# Save the model to file
export_parameters_to_file(parameters, EXPORT_FILE_PATH)