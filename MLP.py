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
        data: np.ndarray = idx2numpy.convert_from_file(Constants.TRAIN_DATA_LOCATION)
        labels: np.ndarray = idx2numpy.convert_from_file(Constants.TRAIN_LABEL_LOCATION)
        
        # The pixel data is stored as ints from 0-1, we first normalize the data
        data = data/255
        
        self.predict_new_number(data[0])
        print(self.output_nodes)
        
        
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
            self.hidden_biases = np.empty(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
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
        input_x_range = math.sqrt(6/(Constants.NODES_PER_HIDDEN_LAYER))
        hidden_x_range = math.sqrt(6/(Constants.INPUT_NODE_COUNT + Constants.NODES_PER_HIDDEN_LAYER))
        output_x_range = math.sqrt(6/(Constants.NODES_PER_HIDDEN_LAYER + Constants.OUTPUT_NODE_COUNT))
        
        # Now create the arrays with uniform random distribution using the ranges
        self.input_weights = np.random.uniform(low=(-1.0 * input_x_range), high=input_x_range, size=(Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        self.hidden_weights = np.random.uniform(low=(-1.0 * hidden_x_range), high=hidden_x_range, size=(Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
        self.output_weights = np.random.uniform(low=(-1.0 * output_x_range), high=output_x_range, size=(Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT))
        
        # Now create the arrays for biases
        self.hidden_biases = np.full(shape=(Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER), fill_value=0)
        self.output_biases = np.full(shape=(Constants.OUTPUT_NODE_COUNT), fill_value=0)
        
        
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
        return 1.0 / (1.0 + np.exp(-x))
    
    
    def predict_new_number(self, input: np.ndarray):
        '''
        This function runs the prediction, filling out all the activations.
        It takes in an array of 728 or a 28x28 matrix.
        '''
        # Create vectorized sigmoid function
        sigmoid_v = np.vectorize(self.sigmoid)
        
        # Flatten the input and check the size
        input = input.flatten()
        if (input.size != Constants.INPUT_NODE_COUNT):
            print("Attempting to predict with an input array of incorrect size")
            exit(1)
            
        # Find the 1st hidden layers activations
        # It will be weights transposed * previous activations + bias vector
        self.hidden_nodes[0] = sigmoid_v((self.input_weights.transpose() @ input) + self.hidden_biases[0])
        
        # Now calculate the other hidden layers
        for i in range(Constants.HIDDEN_LAYER_COUNT - 1):
            self.hidden_nodes[i+1] = sigmoid_v((self.hidden_weights[i].transpose() @ self.hidden_nodes[i]) + self.hidden_biases[i+1])
        
        # Now calculate output layer activation
        self.output_nodes = sigmoid_v((self.output_weights.transpose() @ self.hidden_nodes[Constants.HIDDEN_LAYER_COUNT-1]) + self.output_biases)