import time
import experiments
import numpy
import scipy.special as sci

import util

MNIST_TRAIN = './resources/mnist_train.csv'
MNIST_TEST = './resources/mnist_test.csv'

LEARNING_PLOTS_FOLDER = '.plots'

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        self.lr = learning_rate
        self.mr = []

        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation_function = lambda x: sci.expit(x)

    def __str__(self):
        return f"Neural network with LR: {self.lr}"

    def back_propagate(self, inputs_list, targets_list):
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

        hidden_inputs = numpy.dot(self.wih, numpy_inputs) + 1   #biased neuron
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs) + 1   #biased neuron
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def mistakes(self, fails: int):
        # adapt learning rate
        if len(self.mr) > 0:
            threshold = self.mr[-1] - fails
            if threshold < 5:
                self.lr = self.lr / 2

        self.mr.append(fails)

if __name__ == "__main__":
    start_time = time.time()

    # network configs
    input_nodes = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.06    # 0.065
    epochs = 50

    train_file = MNIST_TRAIN
    test_file = MNIST_TEST

    # create network
    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # ==================================================================================================================
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
        mistakes = 0
        for i, record in enumerate(training_data_list):
            try:
                record_value = record.split(',')

                inputs = (numpy.asarray(record_value[1:], numpy.dtype(float)) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(output_nodes) + 0.01

                targets[int(record_value[0])] = 0.99
                network.back_propagate(inputs, targets)

                # identify mistakes to identify epoch number
                expected = int(record_value[0])
                actual = numpy.argmax(network.query(inputs))
                if not (expected == actual):
                    mistakes += 1

            except ValueError as e:
                mistakes+=1

        network.mistakes(mistakes)
        print(f"Epoch {epoch + 1}, mistakes {mistakes}, LR {network.lr}")

    # Save plot of learning for different learning rate
    x_axis=range(1, epochs + 1)
    x_axis_title="Epochs"
    y_axis=network.mr
    y_axis_title="Mistakes"
    plot_file=f"{LEARNING_PLOTS_FOLDER}/LR_{network.lr}_{time.time_ns()}.png"
    plot_title=str(network)

    util.save_plot(x_axis, y_axis, plot_file, x_axis_title, y_axis_title, plot_title)

    end_time = time.time()
    execution_time = end_time - start_time

    print()
    print(f"Training complete: data length: {len(training_data_list)}, time={execution_time:2}s, "
          f"epochs={epochs}, LR={network.lr}")

    # ==================================================================================================================
    # Test MNIST test file
    data_length, success_rate = experiments.test_mnist(network, MNIST_TEST)
    print()
    print(f"MNIST test file: data length: {data_length}, efficiency_rate={success_rate / data_length}, "
          f"success={success_rate}, fail={data_length - success_rate}")

    # ==================================================================================================================
    # Test my handwriting
    data_length, success_rate = experiments.test_my_handwriting(network)
    print()
    print(f"My handwriting testing: data length: {data_length}, efficiency_rate={success_rate / data_length}, "
          f"success={success_rate}, fail={data_length - success_rate}")