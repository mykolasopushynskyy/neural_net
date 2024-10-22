import time
import numpy
import scipy.special as sci

MNIST_TRAIN_100_CSV = './resources/mnist_train_100.csv'
MNIST_TEST_10_CSV = './resources/mnist_test_10.csv'

MNIST_TRAIN = './resources/mnist_train.csv'
MNIST_TEST = './resources/mnist_test.csv'

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.lr = learning_rate

        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation_function = lambda x: sci.expit(x)

    def train(self, inputs_list, targets_list):
        numpy_inputs = numpy.array(inputs_list, ndmin=2).T
        numpy_targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, numpy_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = numpy_targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs *
                                         (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                         (1.0 - hidden_outputs)), numpy.transpose(numpy_inputs))


    def query(self, inputs_list):
        numpy_inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, numpy_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

if __name__ == "__main__":
    start_time = time.time()

    # network configs
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.25
    epochs = 5

    train_file = MNIST_TRAIN
    test_file = MNIST_TEST

    # create network
    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # open training data
    # CSV contains 100 handwritten numbers in the following format:
    # mnist_train_100.csv
    # number 1, 28x28 pixel data [0..255] of greyscale image
    # number 2, 28x28 pixel data [0..255] of greyscale image
    # ...
    training_data_file = open(train_file)
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train network according to dataset
    for epoch in range(epochs):
        for i, record in enumerate(training_data_list):
            try:
                record_value = record.split(',')

                inputs = (numpy.asarray(record_value[1:], numpy.dtype(float)) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(output_nodes) + 0.01

                targets[int(record_value[0])] = 0.99
                network.train(inputs, targets)
            except ValueError as e:
                pass

    # test network with test dataset
    testing_data_file = open(test_file)
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    data_length = 0
    success_rate = 0
    for i, record in enumerate(testing_data_list):
        try:
            record_value = record.split(',')
            expected = int(record_value[0])

            # test inputs for neural network
            inputs = (numpy.asarray(record_value[1:], numpy.dtype(float)) / 255.0 * 0.99) + 0.01
            # test outputs for neural network
            outputs = network.query(inputs)
            # actual output of nn
            actual = numpy.argmax(outputs)

            output = [[i, round(float(v[0]), 4)] for i, v in enumerate(outputs)]
            # print(f"{expected == actual}: expected={expected}, actual={actual}, output={output}")

            data_length += 1
            success_rate += 1 if expected == actual else 0
        except ValueError as e:
            pass

    end_time = time.time()
    execution_time = end_time - start_time

    print()
    print(f"execution={execution_time:.5f}s, data length: {data_length}, efficiency_rate={success_rate / data_length}, "
          f"success={success_rate}, fail={data_length - success_rate}")