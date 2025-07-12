import numpy as np
import idx2numpy
import Constants
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
        self.test_data = self.test_data / 255
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
        input_x_range = math.sqrt(2/Constants.INPUT_NODE_COUNT)
        hidden_x_range = math.sqrt(2/Constants.NODES_PER_HIDDEN_LAYER)
        output_x_range = math.sqrt(2/Constants.NODES_PER_HIDDEN_LAYER)
        
        # Now create the arrays with uniform random distribution using the ranges
        self.input_weights = np.random.normal(loc=0, scale=input_x_range, size=(Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        self.hidden_weights = np.random.normal(loc=0, scale=hidden_x_range, size=(Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
        self.output_weights = np.random.normal(loc=0, scale=output_x_range, size=(Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT))
        
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
    
    
    def run_batch(self, start_index):
        '''
        start_index: The index of the input data to begin the batch at
        
        Run a batch of predictions, and keep an average of results for backpropagation.
        By the end of this function, it will put the average into the normal self parameters
        EX: average output activations will be places in self.output_nodes
        
        Also populates self.y which will contain the average for the expected outputs taken from labels
        '''
        
        # Set sum matrices to 0s (our gradient, accumulating sum for each example)
        self.sum_dc_dw_input = np.zeros((Constants.INPUT_NODE_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        self.sum_dc_dw_hidden = np.zeros((Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
        self.sum_dc_dw_output = np.zeros((Constants.NODES_PER_HIDDEN_LAYER, Constants.OUTPUT_NODE_COUNT))
        self.sum_dc_db_hidden = np.zeros((Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        self.sum_dc_db_output = np.zeros(Constants.OUTPUT_NODE_COUNT)
        
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
            
            # Get gradient and add it to sum
            self.calculate_and_sum_gradients()
            
        # Update weights and biases
        self.update_weights_and_biases()
        
        return sum_error/Constants.BATCH_SIZE
     
            
    def update_weights_and_biases(self):
        self.input_weights = np.subtract(self.input_weights, np.divide(np.multiply(self.sum_dc_dw_input, Constants.LEARNING_RATE), Constants.BATCH_SIZE))
        self.hidden_weights = np.subtract(self.hidden_weights, np.divide(np.multiply(self.sum_dc_dw_hidden, Constants.LEARNING_RATE), Constants.BATCH_SIZE))
        self.output_weights = np.subtract(self.output_weights, np.divide(np.multiply(self.sum_dc_dw_output, Constants.LEARNING_RATE), Constants.BATCH_SIZE))
        self.hidden_biases = np.subtract(self.hidden_biases, np.divide(np.multiply(self.sum_dc_db_hidden, Constants.LEARNING_RATE), Constants.BATCH_SIZE))
        self.output_biases = np.subtract(self.output_biases, np.divide(np.multiply(self.sum_dc_db_output, Constants.LEARNING_RATE), Constants.BATCH_SIZE))
    
        
    def predict_new_number(self, input: np.ndarray):
        '''
        This function runs the prediction, filling out all the activations.
        It takes in an array of 728 or a 28x28 matrix.
        '''
        # Create vectorized sigmoid function
        relu_v = np.vectorize(self.relu)
        
        # Flatten the input and check the size
        self.input_nodes = input.flatten()
        if (self.input_nodes.size != Constants.INPUT_NODE_COUNT):
            print("Attempting to predict with an input array of incorrect size")
            exit(1)
            
        # Calculate activations, it will be weights_transposed * previous_activations + biases_vector
        # Find the 1st hidden layers activations
        self.hidden_nodes[0] = relu_v((self.input_weights.transpose() @ self.input_nodes) + self.hidden_biases[0])
        
        # Now calculate the other hidden layers
        for i in range(1, Constants.HIDDEN_LAYER_COUNT):
            self.hidden_nodes[i] = relu_v((self.hidden_weights[i-1].transpose() @ self.hidden_nodes[i-1]) + self.hidden_biases[i])
        
        # Now calculate output layer activation
        self.output_nodes = relu_v((self.output_weights.transpose() @ self.hidden_nodes[Constants.HIDDEN_LAYER_COUNT-1]) + self.output_biases)
    
    
    def relu(self, x):
        return x if x > 0 else 0
    
    def relu_prime(self, x):
        return 1 if x > 0 else 0
        
    
    def calculate_and_sum_gradients(self):
        '''
        Variables Used:
        aL = Activation of layer L
        WL = Weight matrix which leads into layer L (m*n matrix where m is the # input nodes, n is # output nodes)
        C = Cost
        sig' = derivative of sigmoid function
        z = linear function = w1a1 + w2a2 + ... + wnan + b
        T = transposed, eg WT is W transposed
        
        '''
        # Create sig' vectors for all nodes
        relu_prime_v = np.vectorize(self.relu_prime)
        relu_prime_output = relu_prime_v(self.output_nodes)
        relu_prime_hidden = relu_prime_v(self.hidden_nodes)
        
        # Calculate dC/da of outputs first first
        dc_da_output = 2 * (self.output_nodes - self.y)
        
        # Create matrix to hold dc_das for hidden layers
        dc_da_hidden = np.zeros((Constants.HIDDEN_LAYER_COUNT, Constants.NODES_PER_HIDDEN_LAYER))
        
        # Now the dC/da for layer behind outputs
        dc_da_hidden[Constants.HIDDEN_LAYER_COUNT-1] = np.multiply(dc_da_output, relu_prime_output) @ np.transpose(self.output_weights)
        
        # Now for the hidden layers
        for i in range(Constants.HIDDEN_LAYER_COUNT - 2, -1, -1):
            dc_da_hidden[i] = np.multiply(dc_da_hidden[i+1], relu_prime_hidden[i+1]) @ np.transpose(self.hidden_weights[i])

        # Done calculating dc/da values
        # Now we calculate dc/dw
        
        # First the output weights
        dc_dw_output = np.outer(self.hidden_nodes[Constants.HIDDEN_LAYER_COUNT-1], np.multiply(dc_da_output, relu_prime_output))
    
        # Now the hidden weights
        dc_dw_hidden = np.zeros((Constants.HIDDEN_LAYER_COUNT - 1, Constants.NODES_PER_HIDDEN_LAYER, Constants.NODES_PER_HIDDEN_LAYER))
        for i in range(Constants.HIDDEN_LAYER_COUNT-2, -1, -1):
            dc_dw_hidden[i] = np.outer(self.hidden_nodes[i], np.multiply(dc_da_hidden[i+1], relu_prime_hidden[i+1]))

        # Now the input weights
        dc_dw_input = np.outer(self.input_nodes, np.multiply(dc_da_hidden[0], relu_prime_hidden[0]))

        # Done calculating dc/dw
        # Now calculate dc/db
        dc_db_output = np.multiply(dc_da_output, relu_prime_output)
        dc_db_hidden = np.multiply(dc_da_hidden, relu_prime_hidden)
        
        # Now add the gradient to the ongoing sum
        self.sum_dc_dw_output = np.add(self.sum_dc_dw_output, dc_dw_output)
        self.sum_dc_dw_hidden = np.add(self.sum_dc_dw_hidden, dc_dw_hidden)
        self.sum_dc_dw_input = np.add(self.sum_dc_dw_input, dc_dw_input)
        self.sum_dc_db_output = np.add(self.sum_dc_db_output, dc_db_output)
        self.sum_dc_db_hidden = np.add(self.sum_dc_db_hidden, dc_db_hidden)
  
    
    def mean_squared_error(self):
        return np.sum(np.power(self.output_nodes - self.y, 2))