import time
import os
import re
import numpy
import scipy.special as sci
from PIL import Image
import util
from neural_network import NeuralNetwork

FAILED_RECOGNITION_FOLDER = '.failed'
LABEL_KEY = "label"
DATA_KEY = "data"
MY_HANDWRITING_TEST_PATH = './resources/handwriting/png_labeled/'

def test_my_handwriting(neural_network: NeuralNetwork):
    # failing rate of image recognition
    correct_answers = 0

    # get images of my handwriting
    my_handwriting_images = util.read_images(MY_HANDWRITING_TEST_PATH)

    for j, image_record in enumerate(my_handwriting_images):
        try:
            image_data = image_record[DATA_KEY]
            label_data = image_record[LABEL_KEY]

            # get numpy array of image
            normalized_input = (numpy.asarray(image_data, numpy.dtype(float)) / 255.0 * 0.99) + 0.01
            # query neural network
            outputs = neural_network.query(normalized_input)
            # actual output of nn
            actual_result = numpy.argmax(outputs)

            correct_answers += 1 if label_data == actual_result else 0

            if not label_data == actual_result:
                image_path = f'./{FAILED_RECOGNITION_FOLDER}/{j}_expected_{label_data}_actual_{actual_result}.png'
                fail_image_data = numpy.asarray(image_data, numpy.uint8)
                util.save_grayscale_image(numpy.reshape(fail_image_data, shape=(28, 28)), image_path)
        except ValueError as e:
            print("Error during testing: ", e)

    return len(my_handwriting_images), correct_answers

def test_mnist(neural_network: NeuralNetwork, mnist_test_file_path: str):
    # test network with mnist test dataset
    testing_data_file = open(mnist_test_file_path)
    testing_data_list = testing_data_file.readlines()
    testing_data_file.close()

    # test network with mnist data
    correct_answers = 0
    for i, record in enumerate(testing_data_list):
        try:
            record_value = record.split(',')
            expected_result = int(record_value[0])

            # test inputs for neural network
            inputs = (numpy.asarray(record_value[1:], numpy.dtype(float)) / 255.0 * 0.99) + 0.01
            # test outputs for neural network
            outputs = neural_network.query(inputs)
            # actual output of nn
            actual_result = numpy.argmax(outputs)

            correct_answers += 1 if expected_result == actual_result else 0
        except ValueError as e:
            print("Error during testing: ", e)

    return len(testing_data_list), correct_answers