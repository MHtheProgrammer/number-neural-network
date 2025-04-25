import numpy as np
import idx2numpy
from DrawnDigit import DrawnDigit
import Constants
from Node import Node
import csv
import math

class MLP:
    
    def __init__(self, use_existing_model_parameters: bool):
        # Initialize the nodes (will store activation values)
        self.input_nodes = np.full(shape=(Constants.INPUT_NODE_COUNT), fill_value=0.0)
        self.hidden_nodes = np.full(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER), fill_value=0.0)
        self.output_nodes = np.full(shape=(Constants.OUTPUT_NODE_COUNT), fill_value=0.0)
        
        # Initialize or read in the weights and biases
        if (use_existing_model_parameters):
            self.read_weights_and_biases()
        else:
            self.initialize_weights_and_biases()
    
    
    def train(self):
        """
        Running this function will train the model on the datasets defined in Constants.
        The weights and biases are stored in a csv file.
        """
        # Read in the training data and labels
        self.data : np.ndarray = idx2numpy.convert_from_file(Constants.TRAIN_DATA_LOCATION)
        self.labels: np.ndarray = idx2numpy.convert_from_file(Constants.TRAIN_LABEL_LOCATION)
        
        # The pixel data is stored as ints from 0-255, we first normalize the data to 0-1
        self.data = self.data/255
        
        # Create some matrices to store our weights and biases
        self.sum_input_weights = np.empty(shape=(Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER), dtype=float)
        self.sum_hidden_weights = np.empty(shape=(Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER), dtype=float)
        self.sum_output_weights = np.empty(shape=(Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT), dtype=float)
        self.sum_hidden_biases = np.empty(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER), dtype=float)
        self.sum_output_biases = np.empty(shape=(Constants.OUTPUT_NODE_COUNT), dtype=float)
        
        # Run the learning algorithm
        count = 0
        data_len = self.data.shape[0]
        while count < data_len:
            error = 0.0
            # Run the mini batch, up to as much as batch_size
            if data_len - count < Constants.BATCH_SIZE:
                break # eventually change this to run 1 last batch
            elif data_len == count:
                break
            else:
                error = self.run_batch(count)
            
            # Print error for this batch
            print("Error in batch " + str(count/Constants.BATCH_SIZE) + ": " + str(error))
            
            # Continue to next batch
            count += Constants.BATCH_SIZE
        
        # Now we can test our model on new data
        self.test_data : np.ndarray = idx2numpy.convert_from_file(Constants.TEST_DATA_LOCATION)
        self.test_labels : np.ndarray = idx2numpy.convert_from_file(Constants.TEST_LABEL_LOCATION)
        error = self.test_model()
        print("Test Error: " + str(error))
        
        # Now store our learned model parameters
        self.write_weights_and_biases_to_csv()
            
        
    def test_model(self):
        
        sum_error = 0.0
        
        # Run the test data predictions
        for i in range(self.test_labels.size):
            self.predict_new_number(self.test_data[i])
                        
            self.y = np.zeros(Constants.OUTPUT_NODE_COUNT)
            self.y[self.test_labels[i]] += 1
        
            # Get the error
            sum_error += self.mean_squared_error()
        
        return sum_error/self.test_labels.size
    
        
    def read_weights_and_biases(self):
        """
        This function will read in the weights and biases stored in the csv file. It will also validate that
        the matrices stored match the sizes set in Constants.py. If they don't, or if it fails to read in the
        data, we return None.
        
        Returns: None on failure
        """
        with open(Constants.WEIGHTS_AND_BIASES_CSV_LOCATION, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Read in the first value, which should be the number of hidden layers
            layer_count = int(next(csv_reader)[0])
            if (layer_count != Constants.HIDDEN_LAYER_COUNT):
                return None
            
            # Read in the weights between inputs and 1st hidden layer
            matrix_size = int(next(csv_reader)[0])
            self.input_weights = np.empty(shape=(Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
            for i in range(matrix_size):
                self.input_weights[i] = np.array(next(csv_reader))
                
            # Read in the weights between hidden layers
            self.hidden_weights = np.empty(shape=(Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
            for i in range(Constants.HIDDEN_LAYER_COUNT - 1):
                matrix_size = int(next(csv_reader)[0])
                for j in range(matrix_size):
                    self.hidden_weights[i][j] = np.array(next(csv_reader))
            
            # Read in weights between final hidden layer and output layer
            self.output_weights = np.empty(shape=(Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT))
            matrix_size = int(next(csv_reader)[0])
            for i in range(matrix_size):
                self.output_weights[i] = np.array(next(csv_reader))
            
            # Read in biases for hidden layers
            self.hidden_biases = np.empty(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER), dtype=float)
            for i in range(layer_count):
                self.hidden_biases[i] = np.array(next(csv_reader))
            
            # Read in biases for output layer
            self.output_biases = np.array(next(csv_reader), dtype=float)            
             

    def initialize_weights_and_biases(self):
        """
        Initialize our weights and biases.
        Weights with Xavier/Gorat Uniform distribution
        Biases set to 0
        """
        # Set the range of weight distribution according to the Gorat method
        input_x_range = math.sqrt(6/(Constants.INPUT_NODE_COUNT + Constants.NODES_PER_HIDDEN_LAYER))
        hidden_x_range = math.sqrt(6/(Constants.NODES_PER_HIDDEN_LAYER + Constants.NODES_PER_HIDDEN_LAYER))
        output_x_range = math.sqrt(6/(Constants.NODES_PER_HIDDEN_LAYER + Constants.OUTPUT_NODE_COUNT))
        
        # Now create the arrays with uniform random distribution using the ranges
        self.input_weights = np.random.uniform(low=(-1.0 * input_x_range), high=input_x_range, size=(Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        self.hidden_weights = np.random.uniform(low=(-1.0 * hidden_x_range), high=hidden_x_range, size=(Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
        self.output_weights = np.random.uniform(low=(-1.0 * output_x_range), high=output_x_range, size=(Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT))
        
        # Now create the arrays for biases
        self.hidden_biases = np.full(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER), fill_value=0.0)
        self.output_biases = np.full(shape=(Constants.OUTPUT_NODE_COUNT), fill_value=0.0)
        
        
    def write_weights_and_biases_to_csv(self):
        """
        Writes the weights and biases to the csv location specified in Constants.
        """
        with open(Constants.WEIGHTS_AND_BIASES_CSV_LOCATION, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the number of layers not including input (3 would be 3 weight matrices and 3 bias vectors)
            csv_writer.writerow([Constants.HIDDEN_LAYER_COUNT])
            
            # Write the weights between input and 1st hidden layer
            csv_writer.writerow([Constants.INPUT_NODE_COUNT])
            csv_writer.writerows(self.input_weights)
            
            # Write the weights between hidden layers
            for matrix in self.hidden_weights:
                csv_writer.writerow([Constants.NODES_PER_HIDDEN_LAYER])
                csv_writer.writerows(matrix)
            
            # Write the weights between final hidden layer and output layer
            csv_writer.writerow([Constants.NODES_PER_HIDDEN_LAYER])
            csv_writer.writerows(self.output_weights)
            
            # Write the biases for hidden layers
            csv_writer.writerows(self.hidden_biases)
            
            # Write the biases for output layer
            csv_writer.writerow(self.output_biases)
    
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))
    
    
    def run_batch(self, start_index):
        '''
        start_index: The index of the input data to begin the batch at
        
        Run a batch of predictions, and keep an average of results for backpropagation.
        By the end of this function, it will put the average into the normal self parameters
        EX: average output activations will be places in self.output_nodes
        
        Also populates self.y which will contain the average for the expected outputs taken from labels
        '''
        
        sum_error = 0.0
        
        # Add the values into the sums each time
        for i in range(Constants.BATCH_SIZE):
            # Run inputs through and get activations
            self.predict_new_number(self.data[start_index + i])
            correct_number = self.labels[start_index + i]
            
            # Set array with correct number
            self.y = np.zeros(Constants.OUTPUT_NODE_COUNT)
            self.y[correct_number] += 1
            
            # Calculate the error
            sum_error += self.mean_squared_error()
            
            # Get derivatives
            self.calculate_and_sum_derivatives()
            
        # Update weights and biases
        self.backpropagate()
        
        # Reset sum matrices to 0s
        self.sum_input_weights = np.zeros_like(self.sum_input_weights)
        self.sum_hidden_weights = np.zeros_like(self.sum_hidden_weights)
        self.sum_output_weights = np.zeros_like(self.sum_output_weights)
        self.sum_hidden_biases = np.zeros_like(self.sum_hidden_biases)
        self.sum_output_biases = np.zeros_like(self.sum_output_biases)
        
        return sum_error/Constants.BATCH_SIZE
            

    def backpropagate(self):
        '''
        Update the weights and biases based on the sums generated by run_batch.
        '''
        self.sum_input_weights /= float(Constants.BATCH_SIZE)
        self.sum_hidden_weights /= float(Constants.BATCH_SIZE)
        self.sum_output_weights /= float(Constants.BATCH_SIZE)
        self.sum_hidden_biases /= float(Constants.BATCH_SIZE)
        self.sum_output_biases /= float(Constants.BATCH_SIZE)
        
        self.input_weights += self.sum_input_weights * (-1.0 * Constants.LEARNING_RATE)
        self.hidden_weights += self.sum_hidden_weights * (-1.0 * Constants.LEARNING_RATE)
        self.output_weights += self.sum_output_weights * (-1.0 * Constants.LEARNING_RATE)
        self.hidden_biases += self.sum_hidden_biases * (-1.0 * Constants.LEARNING_RATE)
        self.output_biases += self.sum_output_biases * (-1.0 * Constants.LEARNING_RATE)
    
        
    def predict_new_number(self, input: np.ndarray):
        '''
        This function runs the prediction, filling out all the activations.
        It takes in an array of 728 or a 28x28 matrix.
        '''
        # Create vectorized sigmoid function
        sigmoid_v = np.vectorize(self.sigmoid)
        
        # Flatten the input and check the size
        self.input_nodes = input.flatten()
        if (self.input_nodes.size != Constants.INPUT_NODE_COUNT):
            print("Attempting to predict with an input array of incorrect size")
            exit(1)
            
        # Calculate activations, it will be weights_transposed * previous_activations + biases_vector
        # Find the 1st hidden layers activations
        self.hidden_nodes[0] = sigmoid_v((self.input_weights.transpose() @ self.input_nodes) + self.hidden_biases[0])
        
        # Now calculate the other hidden layers
        for i in range(1, Constants.HIDDEN_LAYER_COUNT):
            self.hidden_nodes[i] = sigmoid_v((self.hidden_weights[i-1].transpose() @ self.hidden_nodes[i-1]) + self.hidden_biases[i])
        
        # Now calculate output layer activation
        self.output_nodes = sigmoid_v((self.output_weights.transpose() @ self.hidden_nodes[Constants.HIDDEN_LAYER_COUNT-1]) + self.output_biases)
        
    
    def calculate_and_sum_derivatives(self):
        '''
        Will calculate derivatives and add them to the corresponding self.sum_xyz
        
        First lets explain the variables:
        C = the cost function
        L = the layer we are on, L-1 would be the layer to the left
        a = the activation of a node, aLj would be the activation of node j in layer L. Given by equation aLj = sigmoid(zLj)
        w = the weight between 2 nodes, wLjk would be the weight between node j on the left in layer L-1, and node k on the right in layer L
        b = the bias of a node, bLj would be the bias of node j in layer L
        z = the linear function, eg: zLj = wL0j * a(L-1)0 + wL1j * a(L-1)1 + ... + wLnj * a(L-1)n + bLj
        
        In this function we have to do a few things:
        1) calculate the dC/dw for all weights
        2) calculte the dC/db for all biases
        3) update the sums for weights and biases
        
        How do we do it:
        1) For any layer L, and weight wLjk the derivative is dC/dwLjk = dzLk/dwLjk * daLk/dzLk * dC/daLk.
        simplifying the derivative we get dC/dwLjk = aL-1j * aLk * (1 - aLk) * dc/daLk
        2) Same pretty much except 1st term is dzLk/dbLk
        
        to get vector dc/daL, for the output layer its simply 2(aL - y)
        but for any previous layer, dc/daL = wL+1 * aL+1 * (1 - aL+1) * dc/daL+1
        This field requires us to calculate the previous layers dc/daL, so we will just store the values
        as we go backwards
        
        '''
        
        # We will first calculate the dC/da values, starting from outputs and going back
        dC_da_output = np.subtract(self.output_nodes, self.y) * 2
        dC_da_hidden = np.zeros((Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        
        # For each layer calc dc/da, iterating right to left
        for L in range(Constants.HIDDEN_LAYER_COUNT - 1, -1, -1):
            
            # Calculate values needed for dC/da
            wL_1 = self.output_weights if L == (Constants.HIDDEN_LAYER_COUNT - 1) else self.hidden_weights[L]
            aL_1 = self.output_nodes if L == (Constants.HIDDEN_LAYER_COUNT - 1) else self.hidden_nodes[L + 1]
            one_minus_aL_1 = np.subtract(np.full((aL_1.size), 1.0), aL_1)
            dC_daL_1 = dC_da_output if L == (Constants.HIDDEN_LAYER_COUNT - 1) else dC_da_hidden[L + 1]
            
            # Now multiply them
            dC_da_hidden[L] = np.matmul(wL_1, aL_1 * one_minus_aL_1 * dC_daL_1)
        
        # Now we calculate the weights and biases
        for L in range(Constants.HIDDEN_LAYER_COUNT, -1, -1):
            
            # Weights first, also L_m1 means Layer L-1
            aL_m1 = self.input_nodes if L == 0 else self.hidden_nodes[L-1]
            aL = self.output_nodes if L == Constants.HIDDEN_LAYER_COUNT else self.hidden_nodes[L]
            one_minus_aL = np.subtract(np.full(aL.size, 1.0), aL)
            dc_daL = dC_da_output if L == Constants.HIDDEN_LAYER_COUNT else dC_da_hidden[L]
            
            # Now multiply the values
            multiplied_vectors = aL * one_minus_aL * dc_daL
            two_dimensional_mult = np.array([multiplied_vectors])
            two_dimensional_aL_m1 = np.transpose(np.array([aL_m1]))
            sum_weights = (np.matmul(two_dimensional_aL_m1, two_dimensional_mult))
            if L == Constants.HIDDEN_LAYER_COUNT:
                self.sum_output_weights += sum_weights
            elif L == 0:
                self.sum_input_weights += sum_weights
            else:
                self.sum_hidden_weights[L-1] += sum_weights
            
            # Now we do the biases
            sum_biases = aL * one_minus_aL * dc_daL
            if L == Constants.HIDDEN_LAYER_COUNT:
                self.sum_output_biases += sum_biases
            else:
                self.sum_hidden_biases[L] += sum_biases
            
    
    def mean_squared_error(self):
        return np.sum(np.power(self.output_nodes - self.y, 2))