TRAIN_DATA_LOCATION = "./MNIST_datasets/train-images-idx3-ubyte"
TRAIN_LABEL_LOCATION = "./MNIST_datasets/train-labels-idx1-ubyte"
TEST_DATA_LOCATION = "./MNIST_datasets/t10k-images-idx3-ubyte"
TEST_LABEL_LOCATION = "./MNIST_datasets/t10k-labels-idx1-ubyte"

WEIGHTS_AND_BIASES_CSV_LOCATION = "./weights-and-biases.csv"

# MNIST dataset has a 28x28 pixel image, 784 pixels total
INPUT_NODE_COUNT = 784
# 10 possible output for digits 0-9
OUTPUT_NODE_COUNT = 10

HIDDEN_LAYER_COUNT = 2
NODES_PER_HIDDEN_LAYER = 16
