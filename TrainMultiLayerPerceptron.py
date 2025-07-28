import numpy as np
import idx2numpy
import json

# ================ HYPER PARAMETERS ================
# Right at the top for easy access
LAYER_SIZES = [784, 16, 16, 10] ## The num of nodes in each layer, first is input last is output
MAX_EPOCHS = 350
EARLY_STOP_EPOCH_COUNT = 100 ## If this many epochs pass without improvement to loss, then stop training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EXPORT_FILE_PATH = "./model_parameters.json"

# ================ LOAD THE DATASET ================

train_data = idx2numpy.convert_from_file("./MNIST_datasets/train-images-idx3-ubyte")

train_labels = idx2numpy.convert_from_file("./MNIST_datasets/train-labels-idx1-ubyte")

test_data = idx2numpy.convert_from_file("./MNIST_datasets/t10k-images-idx3-ubyte")

test_labels = idx2numpy.convert_from_file("./MNIST_datasets/t10k-labels-idx1-ubyte")

# Resize dataset into n by 784, and convert values from 0-255 to 0-1
train_data = (train_data.reshape(60000, -1).astype('float32')) /255
test_data = (test_data.reshape(10000, -1).astype('float32')) /255

# One hot encode the labels
'''
Takes array of labels ex: [1, 3, 5, ...] and one hot encodes them -> [[0, 1, 0, ...], [0, 0, 0, 1, 0, ...], [...], ...]
'''
def one_hot_encode(x: np.ndarray, num_classes: int):
        ret_arr = np.zeros((x.size, num_classes), dtype=np.float32)
        ret_arr[np.arange(x.size), x] = 1.0
        return ret_arr

train_labels = one_hot_encode(train_labels, LAYER_SIZES[-1])

# Use shape (feature_size, num_samples) for data to make matrix multiplication easier
train_data = train_data.T
test_data = test_data.T
# and shape (num_classes, num_samples) for labels
train_labels = train_labels.T


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
    Softmax takes in a 2-d array of shape (num_classes, num_samples) and performs
    softmax on the matrix row-by-row
    '''
    x_max = np.max(x, axis=0, keepdims=True) ## keepdims gives us (1,n) instead of (n,)
    e_x = np.exp(x - x_max) ## subtract x_max here to prevent overflow / preserve numerical stability (does not affect the softmax math)
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def initialize_model_parameters(layer_sizes: list):
    '''
    Initializes model parameters into a dict where the weights are indexed via
    Wi, where i is the output layer number. Weights between input and 1st hidden layer is W1, and output layer is Wn. No W0
    bi is biases with i referring to the layer
    HE normal initialization used for weights, 0s for biases
    '''
    parameters = {}
    
    for l in range (1, len(layer_sizes)): ## Every layer needs weights and biases except input
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

def run_gradient_descent(X: np.ndarray, Y: np.ndarray, parameters: dict, layer_sizes: list, max_epochs: int, batch_size: int, learning_rate: float, early_stop: int):
    '''
    Takes the following arguments
    X: the training data flattened out into 2d matrix
    Y: labels for training data one hot encoded
    parameters: dict of model parameters {W1, b1, etc}
    layer_sizes: list of layer sizes
    max_epochs: number of times to run through training data
    batch_size: num of samples to use for each iteration of learning
    learning_rate: constant value for learning rate
    early_stop: num of epochs where training has to be ineffective to trigger early stop

    This function will take the parameters and run gradient descent to start tweaking the parameters.
    Returns the updated parameters dict
    '''
    num_samples = X.shape[1]
    epochs_without_improvement = 0
    best_loss = 10000.0 ## arbitrarly sufficient large value
    
    for epoch in range(max_epochs):
        # Randomize order so we get different batches every epoch
        random_order = np.random.permutation(num_samples)
        X_shuffled = X[:, random_order]
        Y_shuffled = Y[:, random_order]
        
        epoch_loss = 0
        
        # Split data into batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_shuffled[:, batch_start:batch_end]
            Y_batch = Y_shuffled[:, batch_start:batch_end]
            
            # Run forward and backwards propagation, then take a step down the gradient
            Y_hat, cache = forward_propagation(X_batch, parameters, layer_sizes)
            gradients = backward_propagation(X_batch, Y_batch, parameters, cache, layer_sizes)
            parameters = update_parameters(parameters, gradients, learning_rate)
            
            # Compute the loss and add it to loss in this epoch
            loss = compute_loss(Y_hat, Y_batch)
            epoch_loss += loss * X_batch.shape[1] ## Accounts for diff sized final batch
        
        # Check if loss is decreasing this epoch
        epoch_loss /= num_samples ## Average loss across samples
        if epoch_loss < (best_loss - 1e-4):
            best_loss = epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # If 100 epochs pass without improvement, end training
        if epochs_without_improvement >= 100:
            return parameters
        
        # print loss on this epoch
        print(f'Loss on epoch {epoch}: {epoch_loss}')
            
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
    
def import_parameters_from_file(filepath: str) -> dict:
    """
    Reads the parameters from a JSON file and returns them as a dictionary.
    """
    with open(filepath, 'r') as file:
        parameters_as_lists = json.load(file)
    return {key: np.array(value) for key, value in parameters_as_lists.items()}
            
def import_test_data_from_file(filepath: str) -> np.ndarray:
    """
    Reads the test data from a JSON file and returns it as a numpy array.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return np.array(data)
            
def export_training_data_to_file(data: np.ndarray, labels: np.ndarray, filepath: str):
    """
    Exports training data and labels to a JSON file.
    """
    with open(filepath, 'w') as file:
        json.dump({
            'data': data.tolist(),
            'labels': np.argmax(labels, 1).tolist()}, file)
# ================ RUN THE TRAINING ================

# Training model

# Debugging purposes
print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(test_labels.shape)
'''
# Create and tweak our model parameters
parameters = initialize_model_parameters(LAYER_SIZES)
parameters = run_gradient_descent(train_data, train_labels, parameters, LAYER_SIZES, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOP_EPOCH_COUNT)

# Test the model on the test data
predictions = predict(test_data, parameters, LAYER_SIZES)
accuracy = get_accuracy(predictions, test_labels)
print(f'Model Accuracy: {accuracy}')

# Save the model to file
export_parameters_to_file(parameters, EXPORT_FILE_PATH)
'''

# Predicting on a test sample with imported parameters

imported_parameters = import_parameters_from_file("09547_accuracy_parameters.json")
for key, value in imported_parameters.items():
    print(f'{key}: {value.shape}')  # Print shapes of imported parameters for debugging
imported_test_sample = import_test_data_from_file("test_sample.json").T.astype('float32') /255
print(imported_test_sample.shape)
prediction = predict(imported_test_sample, imported_parameters, LAYER_SIZES)
print(f'Prediction for the test sample: {prediction}')


# Exporting training data to file
# export_training_data_to_file(test_data.T[:100], one_hot_encode(test_labels.T[:100], LAYER_SIZES[-1]), "training_data.json")