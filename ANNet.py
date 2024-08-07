import numpy as np
import matplotlib.pyplot as plt
import mnist
import math
from colour import Color
from PIL import Image
import copy
import os
import glob
import random
from enum import Enum
from typing import Union
from types import SimpleNamespace
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

try:
    from utils.activation import activate, gradient, ActivationFunction
    from utils.opimizer.adam import AdamOptimizer
except ModuleNotFoundError:
    from .utils.activation import activate, gradient, ActivationFunction
    from .utils.opimizer.adam import AdamOptimizer

# TODO - CLEAN UP CODE
# TODO - Move normalization
# TODO - CHECK HOW TO SET SETTINGS CONVENIENTLY
# TODO - MOVE VISUALIZATION
# TODO - RE WRITE TRAINING AND REMOVE DEPRECATED
# TODO - ADD SAVE METHOD
# TODO - ADD LOAD METHOD
# TODO - CHECK OPTIMIZER


class Normalization(Enum):
    MINMAX = 1
    ZSCORE = 2
    MINMAX_ALL = 3
    MINMAX_OLD = 4


class Initialization(Enum):
    EPSILON = 1
    HE = 2


class ANNet:
    def __init__(self, network_settings: dict = None):
        # Settings
        activation_func = ActivationFunction.SIGMOID
        norm_method = Normalization.ZSCORE
        init_method = Initialization.HE               # Init method
        output_func = activation_func

        # Params
        self.params = SimpleNamespace(alpha=10,  # Learning rate
                                      default_hidden_layer_size=15,  # Default number of neurons in hidden layer
                                      epsilon=0.12,  # Initialization value for weight matrices
                                      use_optimizer=False)

        self.default_hidden_layer_size = 15  # Default number of neurons in hidden layer
        self.epsilon = 0.12                  # Initialization value for weight matrices
        self.alpha = 10                      # Learning rate
        self.use_optimizer = False

        # Network
        self.Theta = []  # Holding weight matrices
        self.network = SimpleNamespace(weight=[],
                                       bias=[],
                                       activation_func=activation_func,  # Activation function [DEFAULTS TO sigmoid)
                                       output_func=output_func,  # Make output function to default ot activation func
                                       input_shape=[],  # Original shape of input data
                                       output_layer_size=None,  # Output layer size
                                       input_layer_size=None,  # Input layer size
                                       network_architecture=None,  # Network architecture
                                       init_method=init_method  # Init method
                                       )

        # Normalization
        self.normalization = SimpleNamespace(feature_mean_vector=[],  # (list) average values in data per feature
                                             feature_var_vector=[],  # (list) variance values in data per feature
                                             feature_min_vector=[],  # (list) min values in data per feature
                                             feature_max_vector=[],  # (list) max values in data per feature
                                             data_min=[],            # (float) min value in input data
                                             data_max=[],            # (float) max value in input data
                                             norm_method=norm_method)  # Normalization method
        # Data
        self.data = SimpleNamespace(num_of_train_samples = None,     # (int) Number of train samples)
                                    num_of_test_samples=None,  # (int) Number of labels in test data-set
                                    train_data=np.array([]),  # Hold train data
                                    test_data=np.array([]),  # Holds test data
                                    train_labels=[],  # Labels for train-data
                                    test_labels=[])  # Labels for test-data
        # Visualization
        self.orig_images = []                # List holding the original images
        self.orig_test_images = []           # List holding the original test images
        self.gif_created = False             # Boolean variable if gif was ever created
        self.is_mnist = False                # Boolean variable stating if mnist data-set is used

        if network_settings is not None:
            self.set_network_settings(network_settings)

        # Optimizer
        self.optimizer = AdamOptimizer(self.network.weight, learning_rate=self.params.alpha)

    def __getattr__(self, item):
        network_attribute = ['activation_func', 'bias', 'input_layer_size', 'input_shape', 'network_architecture',
                             'output_func', 'output_layer_size', 'weight']
        param_attribute = ['alpha', 'default_hidden_layer_size', 'epsilon', 'use_optimizer']
        norm_attribute = ['feature_mean_vector', 'feature_var_vector', 'feature_min_vector',
                          'feature_max_vector', 'data_min', 'data_max', 'norm_method']
        data_attribute = ['num_of_train_samples', 'num_of_test_samples', 'train_data',
                          'test_data', 'train_labels', 'test_labels']
        if item in network_attribute:
            d = eval(f'self.network.{item}')
        elif item in param_attribute:
            d = eval(f'self.params.{item}')
        elif item in norm_attribute:
            d = eval(f'self.normalization.{item}')
        elif item in data_attribute:
            d = eval(f'self.data.{item}')
        else:
            raise AttributeError
        return d

    """
    def __setattr__(self, key, value):
        network_attribute = ['activation_func', 'bias', 'input_layer_size', 'input_shape', 'network_architecture',
                             'output_func', 'output_layer_size', 'weight']
        param_attribute = ['alpha', 'default_hidden_layer_size', 'epsilon', 'use_optimizer']
        norm_attribute = ['feature_mean_vector', 'feature_var_vector', 'feature_min_vector',
                          'feature_max_vector', 'data_min', 'data_max', 'norm_method']
        data_attribute = ['num_of_train_samples', 'num_of_test_samples', 'train_data',
                          'test_data', 'train_labels', 'test_labels']
        if key in network_attribute:
            exec(f'self.network.{key} = value')
        elif key in param_attribute:
            exec(f'self.params.{key} = value')
        elif key in norm_attribute:
            exec(f'self.normalization.{key} = value')
        elif key in data_attribute:
            exec(f'self.data.{key} = value')
        else:
            exec(f'self.{key} = value')
    """

    def set_alpha(self, alpha):
        self.params.alpha = alpha
        self.optimizer.learning_rate = alpha

    def init_network_params(self, network_size):
        """ Takes a list of integers and assigns it to the network_size class variable and creates the weight matrices
            (theta) corresponding to the network architecture. The weights of the artificial network will be initialized
            to random values in the interval (-self.params.epsilon, self.params.epsilon)

        :param network_size: (list)
        :return theta: (list) List of npArrays containing the network weights
        """

        theta = []
        self.network.network_architecture = network_size

        if self.network.init_method == Initialization.EPSILON:
            for i in range(0, len(network_size)-1):
                n = (network_size[i]+1) * network_size[i+1]
                t = np.random.uniform(-self.params.epsilon, self.params.epsilon, size=(1, n))
                theta.extend(t)
            self.Theta = theta
        else:
            for i in range(len(self.network.network_architecture) - 1):
                n = (network_size[i] + 1) * network_size[i + 1]
                input_size = self.network.network_architecture[i]

                # He initialization
                t = np.random.randn(1, n) * np.sqrt(2.0 / input_size)
                theta.append(t)
            self.Theta = theta

        self.network.weight = []
        self.network.bias = []

        for i in range(len(network_size) - 1):
            input_size = network_size[i]
            output_size = network_size[i + 1]

            # He initialization
            w = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            b = np.random.randn(1, output_size)

            self.network.weight.append(w)
            self.network.bias.append(b)

        return theta

    def set_activation_function(self, activation_function: Union[int, str]):
        if isinstance(activation_function, int):
            try:
                self.network.activation_func = ActivationFunction(activation_function)
            except ValueError:
                raise ValueError(f"Integer {activation_function} is not a valid value for Activation Function")
        elif isinstance(activation_function, str):
            try:
                self.network.activation_func = ActivationFunction[activation_function.upper()]
            except KeyError:
                raise ValueError(f"String '{activation_function}' is not a valid activation function")
        else:
            raise TypeError("Input must be an integer or a string")

    def set_init_function(self, init: Union[int, str]):
        if isinstance(init, int):
            try:
                self.network.init_method = Initialization(init)
            except ValueError:
                raise ValueError(f"Integer {init} is not a valid value for Init Function")
        elif isinstance(init, str):
            try:
                self.network.init_method = Initialization[init.upper()]
            except KeyError:
                raise ValueError(f"String '{init}' is not a valid Init function")
        else:
            raise TypeError("Input must be an integer or a string")

    def set_output_activation(self, output: Union[int, str]):
        if isinstance(output, int):
            try:
                self.network.output_func = ActivationFunction(output)
            except ValueError:
                raise ValueError(f"Integer {output} is not a valid value for Activation Function")
        elif isinstance(output, str):
            try:
                self.network.output_func = ActivationFunction[output.upper()]
            except KeyError:
                raise ValueError(f"String '{output}' is not a valid activation function")
        else:
            raise TypeError("Input must be an integer or a string")

    def set_normalization_method(self, norm_method: Union[int, str]):
        if isinstance(norm_method, int):
            try:
                self.normalization.norm_method = Normalization(norm_method)
            except ValueError:
                raise ValueError(f"Integer {norm_method} is not a valid value for Activation Function")
        elif isinstance(norm_method, str):
            try:
                self.normalization.norm_method = Normalization[norm_method.upper()]
            except KeyError:
                raise ValueError(f"String '{norm_method}' is not a valid activation function")
        else:
            raise TypeError("Input must be an integer or a string")

    @staticmethod
    def create_gif():
        """ Will fetch all *.png files located under the temp folder and creates an animated gif to visualize the
            training sequence.

        :return:
        """

        def extract_number(filename):
            return int(filename.split('/')[-1].split('.')[0])  # Assuming Unix-like path separator '/'

        print("Creating gif ... ")
        cur_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cur_path, 'data', 'data_dump', 'images')
        image_directory = glob.glob(temp_path + '/*.png')

        # Sort image paths based on numeric part of filename
        image_directory_sorted = sorted(image_directory, key=extract_number)

        images = []
        i = 0
        for im_dir in image_directory_sorted:
            print(str(i/len(image_directory)))
            if i < 1000:
                images.append(Image.open(im_dir))
            i += 1
        print("Images loaded")
        images[0].save(os.path.join(cur_path, 'data', 'data_dump', 'movie.gif'),
                       save_all=True, append_images=images, duration=10)

    def set_network_architecture(self, network_architecture: list):
        """ Assigns the provided network_architecture variable to the internal class-variable self.network_architecture

        :param network_architecture: (list) List of integer corresponding to the desired size of the input layer,
                                            hidden layers and the output layer.
        :return:
        """
        self.network.network_architecture = network_architecture

    def set_mnist_data(self):
        """ Imports the mnist data-set and assigns it to the train- & test-data. The mnist data-set is an open source
            data set consisting of 60000 training images of hand-written digits. The method will also assign the
            internal class-variable self.is_mnist to true.

        :return:
        """

        # Extract data
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        self.orig_images = train_images
        self.orig_test_images = test_images

        # Assign data
        self.set_train_data(train_images)
        self.set_train_labels(train_labels)
        self.set_test_data(test_images)
        self.set_test_labels(test_labels)
        self.is_mnist = True

    def normalize_data(self, data, data_axis: int = None):
        """ Takes the provided data and normalize it using the min and the max values of the data. If the min and max
            is already provided, the stored max/min value will be used.

        :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
        :param data_axis: (int) Determines which axis should be considered to be the data axis
        :return data: (npArray) Normalized data
        """
        data = np.array(data)
        data = data.astype('float64')

        # Set data and feature axis
        if data_axis is None:
            if data.shape[0] > data.shape[1]:
                data_axis = 0
                feat_axis = 1
            else:
                data_axis = 1
                feat_axis = 0
        else:
            feat_axis = 1

        # Assign mean per feature
        if 'self.normalization.feature_mean_vector' not in locals():
            feature_mean_vector = []
            for iFeat in range(0, data.shape[feat_axis]):
                feature_mean_vector.append(np.mean(data[:, iFeat]))
            self.normalization.feature_mean_vector = feature_mean_vector

        # Extract variance per feature
        if 'self.normalization.feature_var_vector' not in locals():
            feature_var_vector = []
            for iFeat in range(0, data.shape[feat_axis]):
                d = data[:, iFeat]
                feature_var_vector.append(np.var(d))
            self.normalization.feature_var_vector = feature_var_vector

        # Extract min values per feature
        if 'self.normalization.feature_min_vector' not in locals():
            self.normalization.data_min = np.min(np.min(data))
            feature_min_vector = []
            for iFeat in range(0, data.shape[feat_axis]):
                feature_min_vector.append(np.min(data[:, iFeat]))
            self.normalization.feature_min_vector = feature_min_vector

        # Extract max values per feature
        if 'self.normalization.feature_max_vector' not in locals():
            self.normalization.data_max = np.max(np.max(data))
            feature_max_vector = []
            for iFeat in range(0, data.shape[feat_axis]):
                feature_max_vector.append(np.max(data[:, iFeat]))
            self.normalization.feature_max_vector = feature_max_vector

        # ----- Normalize data -----

        # MINMAX
        if self.normalization.norm_method == Normalization.MINMAX:
            for iFeat in range(0, data.shape[feat_axis]):
                d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
                d_norm = ((2 * (d - self.normalization.feature_min_vector[iFeat])) /
                          (self.normalization.feature_max_vector[iFeat] -
                           self.normalization.feature_min_vector[iFeat]) - 1)

                if feat_axis:
                    data[:, iFeat] = d_norm
                else:
                    data[iFeat, :] = d_norm
        # Z-SCORE
        elif self.normalization.norm_method == Normalization.ZSCORE:
            for iFeat in range(0, data.shape[feat_axis]):
                d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
                d_norm = (d - self.normalization.feature_mean_vector[iFeat]) / self.normalization.feature_var_vector[iFeat]

                if feat_axis:
                    data[:, iFeat] = d_norm
                else:
                    data[iFeat, :] = d_norm
        # MIN MAX ALL
        elif self.normalization.norm_method == Normalization.MINMAX_ALL:
            for iFeat in range(0, data.shape[0]):
                d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
                d_norm = ((2*(d - self.normalization.data_min)) /
                          (self.normalization.data_max - self.normalization.data_min) - 1)

                if feat_axis:
                    data[:, iFeat] = d_norm
                else:
                    data[iFeat, :] = d_norm
        # DEPRECATED
        elif self.normalization.norm_method == Normalization.MINMAX_OLD:
            for iData in range(0, data.shape[0]):
                d = np.array(data[iData, :])
                d_norm = (2*(d - self.normalization.data_min)) / (self.normalization.data_max
                                                                  - self.normalization.data_min) - 1
                data[iData, :] = d_norm

        return data

    def reverse_normalization(self, data):
        """ Takes the provided data and un-normalizes it using the min and the max values of the data.

        TODO - IMPLEMENT FOR ALL NORMALIZATION METHODS

        :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
        :return data: (npArray) Un-Normalized data
        """
        data = np.array(data)
        data = data.astype('float64')

        # ----- Reverse Normalize data -----
        for iData in range(0, data.shape[0]):
            d = np.array(data[iData, :])
            d_unnorm = (((d + 1) * (self.normalization.data_max - self.normalization.data_min)) / 2
                        + self.normalization.data_min)
            data[iData, :] = d_unnorm
        return data

    def set_train_data(self, data):
        """ Assigns the provided data to the training data-set. The data will be normalized using the normalize_data
            method. The method will also assign the class variables input_layer_size, num_of_train_samples, input_shape
            based on the properties of the provided data.

        :param data: (npArray) Data to be set to the training data-set
        :return:
        """
        train_data = []

        if len(data.shape) == 3:
            self.network.input_layer_size = data.shape[1] * data.shape[2]
            self.data.num_of_train_samples = data.shape[0]
            self.network.input_shape = [data.shape[1], data.shape[2]]
            # Reshape data
            for i in range(0, len(data)):
                train_data.append(np.array(np.matrix.flatten(data[i])))
        elif len(data.shape) == 2:
            self.network.input_layer_size = data.shape[1]
            self.data.num_of_train_samples = data.shape[0]
            self.network.input_shape = [1, data.shape[1]]
            train_data = data
        train_data = self.normalize_data(np.array(train_data))
        self.data.train_data = train_data

    def set_test_data(self, data):
        """ Assigns the provided data to the test data-set. The data will be normalized using the normalize_data
            method. The method will also assign the class variables input_layer_size, num_of_train_samples, input_shape
            based on the properties of the provided data.

        :param data: (npArray) Either a numpy array or list with test samples
        :return:
        """

        test_data = []
        self.data.num_of_test_samples = data.shape[0]
        if len(data.shape) == 3:
            self.network.input_shape = [data.shape[1], data.shape[2]]
            # Reshape data
            for i in range(0, len(data)):
                test_data.append(np.matrix.flatten(data[i]))
        elif len(data.shape) == 2:
            self.network.input_shape = [1, data.shape[1]]
        test_data = self.normalize_data(np.array(test_data))
        self.data.test_data = np.array(test_data)

    def set_train_labels(self, labels):
        """ Assigns the provided labels to the labels of the training data-set. The method will also set the
            class-variable output_layer_size based on the unique set of labels in the provided data.

        :param labels: (npArray/list) Either a numpy array or list with labels corresponding to the true classifications
                                      of the training data-set
        :return:
        """
        self.data.train_labels = labels
        if 'self.network.output_layer_size' not in locals():
            self.network.output_layer_size = len(np.unique(labels))

    def set_test_labels(self, labels):
        """ Assigns the provided labels to the labels of the test data-set. The method will also set the
            class-variable output_layer_size based on the unique set of labels in the provided data.

        :param labels: (npArray/list) Either a numpy array or list with labels corresponding to the true classifications
                                      of the training data-set
        :return:
        """
        self.data.test_labels = labels
        if 'self.network.output_layer_size' not in locals():
            self.network.output_layer_size = len(np.unique(labels))

    def draw_and_predict(self):
        """ The method will randomly pick a sample in the test data-set and makes a prediction by doing a forward
            propagation using the current state of the neural network. If the network data is the mnist data-set
            the drawn sample will be visualised and the predicted value will be printed.

        :return:
        """
        sample_idx = random.randint(0, self.data.test_data.shape[0])
        x = self.data.test_data[sample_idx, :]
        y = self.data.test_labels[sample_idx]
        y_hat = np.zeros((self.network.output_layer_size, 1))
        y_hat[y] = 1
        h, c, a, z = ANNet.forward_propagation(self, x, y_hat)
        prediction = np.argmax(h)
        print("Network predicts value: " + str(prediction))
        im = x.reshape(self.network.input_shape[0], self.network.input_shape[1])
        plt.imshow(im)
        plt.show()

    def init_weights(self):
        # Initialize weights if they have not been initialized by user.
        if not self.Theta:
            if self.network.network_architecture is None:
                print("Network weights has not been initialized by user!, default hidden layer  of size " +
                      str(self.params.default_hidden_layer_size) + " has been applied")
                input_layer_size = self.network.input_layer_size
                output_layer_size = self.network.output_layer_size
                network_architecture = [input_layer_size, self.params.default_hidden_layer_size, output_layer_size]
                self.network.network_architecture = network_architecture
            self.init_network_params(self.network.network_architecture)

    def set_network_settings(self, settings: dict):
        if "normalization_method" in settings.keys():
            self.set_normalization_method(settings["normalization_method"])

        if "activation_function" in settings.keys():
            self.set_activation_function(settings["activation_function"])

        if "network_architecture" in settings.keys():
            self.set_network_architecture(settings["network_architecture"])

        if "init_method" in settings.keys():
            self.set_init_function(settings["init_method"])

        if "output_activation" in settings.keys():
            self.set_output_activation(settings["output_activation"])

    def forward(self, x):
        """ Performs the forward propagation through the neural network.

        :param x: (numpy array) Input data of shape (m, input_size), where m is the number of examples.
        :return:
            activations: (list) List of activation values for each layer.
            zs: (list) List of z values (weighted input to activation functions) for each layer.
            output: (numpy array) Final output of the network.
        """
        activations = [x]  # Store the input layer activations
        zs = []            # Store the z values for each layer

        for i in range(len(self.network.weight)):
            # Compute the z value (weighted sum of inputs + bias)
            z = np.dot(activations[-1], self.network.weight[i]) + self.network.bias[i]
            zs.append(z)  # Store the z value
            # Apply the activation function (sigmoid)
            a = activate(self.network.activation_func, z) if i != len(self.network.weight) - 1 else (
                activate(self.network.output_func, z))
            activations.append(a)  # Store the activations

        return activations, zs, activations[-1]

    def back(self, activations, z_list, y):
        """ Performs the backpropagation through the neural network to update weights and biases.

        :param activations: (list) List of activation values for each layer.
        :param z_list: (list) List of z values (weighted input to activation functions) for each layer.
        :param y: (numpy array) True labels of shape (m,), where m is the number of examples.
        :return: None
        """
        # Reshape y to match the output shape
        if activations[-1].shape != y.shape:
            y = y.reshape(-1, 1)

        # Calculate the error at the output layer (difference between prediction and true label)
        output_error = (activations[-1] - y)
        # Calculate delta for the output layer
        deltas = [output_error * gradient(self.network.output_func, z_list[-1])]

        # Initialize gradients for biases and weights for each layer
        d_bias = [deltas[-1].mean(axis=0)]
        d_weight = [np.dot(activations[-2].T, deltas[-1])]

        # Iterate backwards through the network layers
        for l in reversed(range(len(self.network.weight) - 1)):
            z = z_list[l]
            zp = gradient(self.network.activation_func, z)
            delta = np.dot(deltas[-1], self.network.weight[l + 1].T) * zp
            deltas.append(delta)
            d_bias.append(delta.mean(axis=0))
            d_weight.append(np.dot(activations[l].T, delta))

        # Reverse the gradients to align with the network layer order
        d_weight.reverse()
        d_bias.reverse()

        # Update the weights and biases using the calculated gradients
        for i in range(len(self.network.weight)):
            if not self.params.use_optimizer:
                self.network.weight[i] -= self.params.alpha * d_weight[i]
            self.network.bias[i] -= self.params.alpha * d_bias[i]

        if self.params.use_optimizer:
            self.optimizer.set_parameters(self.network.weight)
            self.network.weight = self.optimizer.step(d_weight)

    def forward_propagation(self, x, y=None):
        """ Will make predictions by forward propagate the provided input data x through the neural network stored in
            within the class. If the annotations (labels) are provided through the y-input variable, the cost (c) of the
            network will be calculated post the forward propagation using the least-square method. Else the cost will be
            set to None. Along with the predictions (h) per provided samples and the total cost (c), the method will
            return a list of all activation layers (a_list) and a list of all sigmoid layers (z_list)

        :param x:  (npArray) Input data to be propagated through the neural network, Either one or more samples.
        :param y:  (npArray) [optional] Annotations to be used to calculate the network cost

        :return h, (npArray)   Prediction array
                c, (float)     Network cost
                a_list, (list) List of all activation layers
                z_list: (list) List of sigmoid layers
        """
        num_of_samples = len(x) if len(x.shape) > 1 else 1  # Sets the number of samples based on the dimensions of the
        #                                                     provided input data (x)
        bias = np.ones((num_of_samples, 1))                 # Create bias units
        a = x                                               # Sets first activation layer to the provided input data

        a_list = []  # List containing all activation in the forward propagation
        if num_of_samples > 1:
            a = np.vstack([bias.T, a.T])
            a = a.T
        else:
            a = np.append(bias, a)
        a_list.append(a)

        z_list = []  # List to hold all sigmoid values
        for i, theta in enumerate(self.Theta):   # Forward propagate through all layers
            # Reshape parameter matrix
            t = theta.reshape(self.network.network_architecture[i]+1, self.network.network_architecture[i+1])

            z = np.matmul(a, t)  # z = a*t
            z_list.append(z)     # Store the sigmoid values

            a = activate(self.network.activation_func, z) if i != len(self.Theta) - 1 else (
                activate(self.network.output_func, z))

            # add bias unit
            if num_of_samples > 1:
                a = np.column_stack([bias, a])
            else:
                a = np.append(bias, a)

            a_list.append(a)     # Store activation layer

        h = a  # Prediction layer is equal to the final activation layer
        # pop bias unit
        if len(h.shape) > 1:
            h = h[:, 1:]
        else:
            h = h[1:]
        a_list.pop(-1)  # Last layer in a is equivalent to h

        # Calculate cost (Least Square Method)
        if y is None:
            # labels has not been provided, cost can not be calculated
            c = None
        else:
            c = (1 / (2 * num_of_samples)) * np.sum((h - y) ** 2)  # Cost with LSM

        return h, c, a_list, z_list

    def back_propagation(self, delta, z_mat, a_mat):
        # Init weight gradient matrices
        theta_grad = []
        for iLayer, theta in enumerate(self.Theta):
            theta = theta.reshape(self.network.network_architecture[iLayer] + 1,
                                  self.network.network_architecture[iLayer + 1])
            theta_grad.append(np.zeros(theta.shape))

        for iLayer in range(len(a_mat) - 1, -1, -1):
            z = z_mat[iLayer]
            if not iLayer == len(a_mat) - 1:
                index = iLayer + 1
                theta = self.Theta[index]
                t = theta.reshape(self.network.network_architecture[index] + 1,
                                  self.network.network_architecture[index + 1])
                t = t[1:, :]

                delta_weight = np.dot(t, delta.T)
                sig_z = gradient(self.network.activation_func, z)
                delta = delta_weight * sig_z.T
                delta = delta.T
            else:
                delta = delta * gradient(self.network.output_func, z)

            a = a_mat[iLayer]
            th_grad = np.dot(a.T, delta)
            theta_grad[iLayer] += th_grad

        # Update weights from the weight gradients
        if self.params.use_optimizer and False:
            self.optimizer.set_parameters(self.Theta)
            self.Theta = self.optimizer.step(theta_grad)
        else:
            for i, theta_val in enumerate(theta_grad):
                theta_grad[i] = (1 / len(delta)) * theta_val
                t = self.params.alpha * theta_grad[i]
                self.Theta[i] -= t.flatten()

    def train_network(self, x=None, y=None, num_of_iterations=1000, visualize_training=False):
        """ The method will initialize a training session with a number of training iterations determined by the
            variable num_of_iterations (int). The network will be trained using the x data as input data and the
            y data as the annotations. If x or y are not provided the input data and the annotation data will be
            set to the internal class-variables self.train_data and self.train_labels. The network is trained by
            forward propagating the input data and then back propagate the errors using gradient decent. If the
            boolean variable visualize_training is set to true, the training session will be visualized and the
            output will be stored in a temp-folder under the current directory path.

        :param x:                  (npArray)  Input data to be propagated through the network
        :param y:                  (npArray)  Training labels used for calculating the network errors for
                                              back-propagation
        :param num_of_iterations:  (int)      Number of iterations to be executed in the training session
        :param visualize_training: (boolean)  Boolean flag to indicate if the training session should be visualized
        :return:
        """
        historic_prediction = []  # holds the predicted values for every iteration, used for visualization
        historic_theta = []       # holds the wight matrices for every iteration, used for visualization

        # Set input and label data if not provided
        if (x is None) or (y is None):
            x = self.data.train_data
            y = self.data.train_labels

        num_of_samples = len(x)   # Samples in the training set

        # Initialize weights if they have not been initialized by user.
        if not self.Theta:
            if self.network.network_architecture is None:
                print("Network weights has not been initialized by user!, default hidden layer  of size " +
                      str(self.params.default_hidden_layer_size) + " has been applied")
                input_layer_size = self.network.input_layer_size
                output_layer_size = self.network.output_layer_size
                network_architecture = [input_layer_size, self.params.default_hidden_layer_size, output_layer_size]
                self.network.network_architecture = network_architecture
            self.init_network_params(self.network.network_architecture)

        # Preparing label data, every sample will be represented by a vector with the size equal to the number of
        # unique classes in the training data. The correct classification will be marked by a one while the false
        # classifications are marked with a zero for all samples. The label matrix will be of the size
        # [num_of_train_samples x num_of_unique_classes]
        if len(y.shape) < 2:
            y_mat = y
            # y_mat = np.zeros((y.shape[0], self.output_layer_size))
            # for i, val in enumerate(y):
            #    y_mat[i][int(val)] = 1
        elif y.shape[0] == 1 or y.shape[1] == 1:
            y_mat = np.zeros((y.shape[0], self.network.output_layer_size))
            for i, val in enumerate(y):
                y_mat[i][int(val)] = 1
        else:
            y_mat = y

        cost = np.zeros([num_of_iterations, 1])      # Holding cost value per iteration
        accuracy = np.zeros([num_of_iterations, 1])  # Holding accuracy value per iteration
        for iteration in range(0, num_of_iterations):
            # Init weight gradient matrices
            theta_grad = []
            for iLayer, theta in enumerate(self.Theta):
                theta = theta.reshape(self.network.network_architecture[iLayer]+1,
                                      self.network.network_architecture[iLayer+1])
                theta_grad.append(np.zeros(theta.shape))

            h_mat, c, a_mat, z_mat = ANNet.forward_propagation(self, x, y_mat)  # Forward propagate
            if len(y_mat.shape) < 2:
                delta = (h_mat.transpose() - y).transpose()
            else:
                delta = -(y_mat-h_mat)
            for iLayer in range(len(a_mat)-1, -1, -1):
                z = z_mat[iLayer]
                if not iLayer == len(a_mat)-1:
                    index = iLayer + 1
                    theta = self.Theta[index]
                    t = theta.reshape(self.network.network_architecture[index] + 1,
                                      self.network.network_architecture[index + 1])
                    t = t[1:, :]

                    delta_weight = np.dot(t, delta.T)
                    sig_z = gradient(self.network.activation_func, z)
                    delta = delta_weight * sig_z.T
                    delta = delta.T
                else:
                    delta = delta * gradient(self.network.output_func, z)

                a = a_mat[iLayer]
                th_grad = np.dot(a.T, delta)
                theta_grad[iLayer] += th_grad

            # Update weights from the weight gradients
            if self.params.use_optimizer:
                self.optimizer.set_parameters(self.Theta)
                self.Theta = self.optimizer.step(theta_grad)
            else:
                for i, theta_val in enumerate(theta_grad):
                    theta_grad[i] = (1/num_of_samples)*theta_val
                    t = self.params.alpha * theta_grad[i]
                    self.Theta[i] -= t.flatten()

            # Save the weight values for animation
            historic_theta.append(copy.deepcopy(self.Theta))

            # Calculate and print cost
            h, c, _, _ = ANNet.forward_propagation(self, x, y_mat)
            historic_prediction.append(h)
            itr = 0
            num_of_errors = 0
            for p in h:
                prediction = np.argmax(p)
                y = np.argmax(y_mat[itr])
                if not prediction == y:
                    num_of_errors += 1
                itr += 1
            # ------------------------------ print iteration --------------------------------
            cost[iteration] = c
            accuracy[iteration] = 1 - num_of_errors/num_of_samples
            print("Iteration (" + str(iteration) + "/" + str(num_of_iterations) + "), " +
                  "Cost of network: " + str(c) +
                  " , number of false predictions: " + str(num_of_errors) +
                  " , accuracy: " + str(1 - num_of_errors/num_of_samples))

        if visualize_training:

            # # ===== Dump data to json =====
            if self.is_mnist:
                picture_idx = random.randint(0, len(self.data.test_labels))
                h_test, c_test, a_test, z_test = ANNet.forward_propagation(self, self.data.test_data[picture_idx, :],
                                                                           self.data.test_labels[picture_idx])
                picture = self.orig_test_images[picture_idx, :, :]
            else:
                picture, a_test, z_test, h_test = None, None, None, None

            # create dump json file
            cur_path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(cur_path, 'data', 'data_dump', 'training')

            if not os.path.exists(data_path):
                # Create dump folder
                os.makedirs(data_path)
            else:
                # clean dump folder
                for file in glob.glob(os.path.join(data_path, '*.npy')):
                    os.remove(file)

            # Dump data
            if self.is_mnist:
                data = {'picture': picture,
                        'theta': self.Theta,
                        'network_size': self.network.network_architecture,
                        'a': a_test,
                        'z': z_test,
                        'prediction': h_test}
                np.save(os.path.join(data_path, 'data.npy'), data)  # save to npy file

            # ----- Plot Cost/Accuracy progression -----
            plt.close()
            plt.plot(cost)
            plt.plot(accuracy)
            plt.pause(0.001)
            plt.ion()
            plt.show()
            self.visualize_training(x, cost, accuracy, historic_theta, historic_prediction)

    def visualize_training(self, x, cost, accuracy, historic_theta, historic_prediction):
        """ The method will visualize the training session in a (2x2) subplot displaying cost/accuracy, sample
            predictions and neural network for the whole training session. The output will be stored with .png
            files stored in a temp folder.


        :param x:                   (npArray) Training input data set
        :param cost:                (list)    List holding the cost values per iteration
        :param accuracy:            (list)    List holding the accuracy values per iteration
        :param historic_theta:      (list)    List holding all weight matrices per iteration
        :param historic_prediction: (list)    List holding all predictions for every iteration
        :return:
        """

        fig, axs = plt.subplots(2, 2)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        # mng.window.state('zoomed')
        temp_folder = self.create_temp_folder()  # Create a temp-folder to store the training visualisation content

        # Extract min/max weight
        absolute_weight = False
        max_weight = []
        min_weight = []
        if absolute_weight:
            max_weight = -np.inf
            min_weight = np.inf
            for theta in historic_theta:
                for t in theta:
                    max_v = np.max(np.max(t))
                    min_v = np.min(np.min(t))
                    if max_weight < max_v:
                        max_weight = max_v
                    if min_weight > min_v:
                        min_weight = min_v

        picture_idx = []
        num_of_pics_m = 23
        num_of_pics_n = 45
        tot_num_of_pics = num_of_pics_m * num_of_pics_n
        if self.is_mnist:
            picture_idx = random.sample(range(0, len(self.data.train_labels)), tot_num_of_pics)

        image_directory = []
        num_of_digits = self.determine_number_of_digits(len(historic_prediction))

        n_start = 0
        n_end = len(historic_prediction)
        ask_user = True
        if ask_user:
            try:
                n_start = int(input("Select starting index for rendering"))
                n_end = int(input("Select ending index for rendering"))
            except ValueError:
                print("Provided input is not valid, only integers.")

        for iteration in range(n_start, n_end):
            title_str = 'Cost: ' + str(cost[iteration]) + ' , Accuracy: ' + str(accuracy[iteration]) + \
                        ' , Iteration: ' + str(iteration)
            fig.suptitle(title_str)
            print(iteration)
            h = historic_prediction[iteration]

            # ----- Prediction plot [0, 0]-----
            outcome = []
            col = []
            col_true = []
            if self.is_mnist:
                output_classes = list(set(self.data.train_labels))

                num_of_false_predictions = np.zeros((len(output_classes),), dtype=int)
                num_of_true_predictions = np.zeros((len(output_classes),), dtype=int)
                outcome = np.ones((len(h),), dtype=int)
                for i_sample in range(0, len(h)):
                    y = self.data.train_labels[i_sample]
                    prediction = np.argmax(h[i_sample, :])
                    if prediction != y:
                        outcome[i_sample] = 0
                        num_of_false_predictions[y] += 1
                    else:
                        num_of_true_predictions[y] += 1

                axs[0, 0].bar(output_classes, num_of_false_predictions, color='#ff5959', edgecolor='white')
                axs[0, 0].bar(output_classes, num_of_true_predictions, color='#34ebd6', edgecolor='white',
                              bottom=num_of_false_predictions)
                axs[0, 0].set_xlabel('Output classes')
                axs[0, 0].set_ylabel('Number of false vs number of true')
            else:
                for i_sample in range(0, h.shape[0]):
                    col.append('#ff5959' if np.argmax(h[i_sample, :]) else '#34ebd6')
                    col_true.append('#ff5959' if self.data.train_labels[i_sample] else '#34ebd6')

                axs[0, 0].scatter(x[:, 0], x[:, 1], c=col)
                axs[0, 0].axis('equal')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])

            # ----- Cost / Accuracy plot [1,0] -----
            axs[1, 0].plot(cost[0: iteration], label='Cost')
            axs[1, 0].plot(accuracy[0: iteration], label='Accuracy')
            axs[1, 0].legend()

            # ----- Reference plot  [0, 1]-----
            if self.is_mnist:
                classification_result = outcome[picture_idx]
                border_thickness = 4

                m = self.network.input_shape[0] + 2*border_thickness
                n = self.network.input_shape[1] + 2*border_thickness
                validation_image = np.zeros((num_of_pics_m * m,
                                             num_of_pics_n * n,
                                             3))
                def_im = np.zeros((m, n, 3))

                # Set frame
                im_true = def_im.copy()
                border_color_channel = 1  # Green
                im_true[0:border_thickness - 1, 0:n, border_color_channel] = 1      # top
                im_true[m - border_thickness + 1:m, 0:n, border_color_channel] = 1  # right
                im_true[0:m, 0:border_thickness - 1, border_color_channel] = 1      # left
                im_true[0:m, n - border_thickness + 1:n, border_color_channel] = 1  # bottom

                im_false = def_im.copy()
                border_color_channel = 0  # Red
                im_false[0:border_thickness - 1, 0:n, border_color_channel] = 1  # top
                im_false[m - border_thickness + 1:m, 0:n, border_color_channel] = 1  # right
                im_false[0:m, 0:border_thickness - 1, border_color_channel] = 1  # left
                im_false[0:m, n - border_thickness + 1:n, border_color_channel] = 1  # bottom

                idx = 0
                for i in range(0, num_of_pics_m):
                    for j in range(0, num_of_pics_n):
                        sample_idx = picture_idx[idx]
                        mnist_im = self.orig_images[sample_idx]
                        if classification_result[idx]:
                            im = im_true
                        else:
                            im = im_false

                        # Set image
                        im[border_thickness:m-border_thickness,
                           border_thickness:n-border_thickness, 0] = mnist_im / 255
                        im[border_thickness:m - border_thickness,
                           border_thickness:n - border_thickness, 1] = mnist_im / 255
                        im[border_thickness:m - border_thickness,
                           border_thickness:n - border_thickness, 2] = mnist_im / 255

                        # ----- Set overall picture -----
                        validation_image[i*m:(i+1)*m, j*n:(j+1)*n, :] = im
                        idx += 1
                axs[0, 1].imshow(validation_image)
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 1].set_title(str(round(tot_num_of_pics / len(self.data.train_labels) * 1000) / 10)
                                    + '% of the data')
            else:
                axs[0, 1].scatter(x[:, 0], x[:, 1], c=col_true)
                axs[0, 1].axis('equal')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])

            # ----- Network plot [1, 1]-----
            # if self.is_mnist:
            if absolute_weight:
                self.draw_network(historic_theta[iteration], axs[1, 1], max_weight, min_weight)
            else:
                self.draw_network(historic_theta[iteration], axs[1, 1])

            # ----- Save figure -----
            im_dir = self.append_with_zeros(num_of_digits, iteration) + '.png'
            im_dir = os.path.join(temp_folder, im_dir)
            image_directory.append(im_dir)
            plt.savefig(im_dir)

            # ----- Clear figures -----
            axs[0, 0].clear()
            axs[0, 1].clear()
            axs[1, 1].clear()
            axs[1, 0].clear()
        plt.close(fig)

    @staticmethod
    def create_temp_folder():
        """ The method will create a folder called temp in the current file-directory if it already does no exist. If
            the folder already exists it will flush it by deleting all .png files located in the folder.

        :return temp_folder: (str) path to temp folder
        """
        current_path = os.path.dirname(os.path.realpath(__file__))  # Get current directory path
        temp_folder = os.path.join(current_path, 'data', 'data_dump', 'images')  # Temp folder for storing images
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)                                # Create if not already exist

        # flush folder from png
        files = os.listdir(temp_folder)                             # List all files in temp folder
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(temp_folder, file))          # Delete if file is png
        return temp_folder

    @staticmethod
    def determine_number_of_digits(num_of_iterations):
        """ Determines the number of digits needed in the image naming to hold all iterations. This is used to get the
            images naturally sorted based on name even in a windows file-explorer.

        :param num_of_iterations: (int) Total number of iterations
        :return num_of_digits:    (int) Total number of needed digits in the file-naming
        """
        return math.ceil(math.log10(num_of_iterations + 0.001))

    @staticmethod
    def append_with_zeros(num_of_digits, iteration):
        """ Will give the file-name of the current iteration. The file will be named to enable natrual sorting even in
            a windows OS. For instance, if total number of digits == 3 and current iteration is 7 the file-name will be
            "007"

        :param num_of_digits:  (int) Total number of needed digits in the file-naming
        :param iteration:      (int) Current iteration
        :return appended_name: (str) File name
        """
        number_of_zeros = num_of_digits - math.ceil(math.log10(iteration + 0.001))

        appended_name = ''
        for i in range(0, number_of_zeros):
            appended_name += '0'
        appended_name += str(iteration)
        return appended_name

    def draw_network(self, theta, axs, max_weight=None, min_weight=None):
        """ The method will take list of weight matrices (npArray) and a plot axis and display the the network
            given by the weight matrices. If the max&min weight values (float) are given, they will be used as
            max and min for the color scaling of the weight lines, otherwise the the method will use the max
            and min value in the weight matrices. This is used to enable absolute color scaling through
            a series of training iterations. The method will display all neurons, along with bias neurons as-well
            as the weight network between the neurons. The color of the neurons and the weights will be set by
            their individual values. The method will also display the sum of the network per layer to enable
            visualisation of changes in the network between iterations.

        :param theta:      (list)            List of weight matrices to be displayed
        :param axs:        (matplotlib axes) axis where the figure should be displayed
        :param max_weight: (float)           maximum value used for color scaling
        :param min_weight: (float)           minimum value used for color scaling
        :return:
        """
        neuron_radius = 1                       # Neuron radius
        neuron_distance = 15                    # Distance between neurons in each layer
        layer_distance = 10 * neuron_distance   # Distance between each layer

        # Reconstruct weight matrices
        network_architecture = []
        i_layer = len(theta)
        for i_layer, t in enumerate(theta):
            theta[i_layer] = t.reshape(self.network.network_architecture[i_layer] + 1,
                                       self.network.network_architecture[i_layer + 1])
            network_architecture.append(theta[i_layer].shape[0])
        network_architecture.append(theta[i_layer].shape[1])

        # Calculate position lists
        use_max_y = True
        max_y = (max(network_architecture)-1)*(neuron_distance + 2*neuron_radius)

        x = 0                  # x-position of the first neural layer
        neuron_positions = []  # List holding the positions of all neurons
        min_y_pos = []         # List holding the minimum y-value for each layer
        for N_neurons in network_architecture:
            p = []  # Position list storing the positions of the neurons of the current layer

            # Find vertical start position
            if use_max_y:
                # The width in the y-dimension will be equal for all layers
                y0 = max_y / 2
                delta_y = max_y / (N_neurons - 1)
            else:
                # The width in the y-dimension will be different for all layers. The y-positions of the neurons will
                # distributed according to the neuron_radius and neuron_distance variables

                if N_neurons % 2:
                    n = math.floor(N_neurons / 2)
                    y0 = (2 * neuron_radius + neuron_distance) * n
                else:
                    n = N_neurons / 2
                    y0 = (neuron_distance / 2 + neuron_radius) + (neuron_distance + 2 * neuron_radius) * (n - 1)

                delta_y = (neuron_distance + 2 * neuron_radius)

            # Position the neurons of the current layer
            y = y0
            for i in range(0, N_neurons):
                pos_tuple = (x, y)
                p.append(pos_tuple)
                y -= delta_y

            neuron_positions.append(p)  # Store the neuron positions of the current layer
            x += layer_distance         # update the x-position for the new neuron layer
            min_y_pos.append(y)         # Store the minimal y-position of the current layer

        # ======== Get colormap ========
        blue = Color("blue")
        n_col_steps = 1000
        colors = list(blue.range_to(Color("red"), n_col_steps))  # Colormap to be used for scaling the colors of
        #                                                          the neurons and the weights

        # ===== Plot neurons =====
        for iLayer, n_pos in enumerate(neuron_positions):
            if iLayer == len(neuron_positions) - 1:
                output_layer = True
                input_layer = False
                t = theta[iLayer - 1]
            elif not iLayer:
                output_layer = False
                input_layer = True
                t = []
            else:
                output_layer = False
                input_layer = False
                t = theta[iLayer - 1]

            if output_layer:
                neuron_value = []
                for i in range(0, t.shape[0]):
                    neuron_value.append(np.sum(t[i, :]))
                min_neuron_val = np.min(neuron_value)
                max_neuron_val = np.max(neuron_value)
            elif not input_layer:
                neuron_value = []
                for i in range(0, t.shape[1]):
                    neuron_value.append(np.sum(t[:, i]))
                min_neuron_val = np.min(neuron_value)
                max_neuron_val = np.max(neuron_value)
            else:
                neuron_value = []
                min_neuron_val = None
                max_neuron_val = None

            for iNeuron, p in enumerate(n_pos):
                # Get synapse color
                if input_layer:
                    # Input layer
                    c = str(colors[0])
                else:
                    if iNeuron:
                        col_idx = math.floor(((neuron_value[iNeuron - 1] - min_neuron_val) /
                                              (max_neuron_val - min_neuron_val)) * (n_col_steps - 1))
                        c = str(colors[col_idx])
                    else:
                        c = str(colors[0])

                draw_circle = plt.Circle((p[0], p[1]), radius=neuron_radius, color=c)
                axs.add_patch(draw_circle)

        # ===== Plot Connections =====

        # extract min/max values, used if max&min weight values are not provided by user
        if not max_weight or not min_weight:
            max_weight = -np.inf
            min_weight = np.inf
            for t in theta:
                max_v = np.max(np.max(t))
                min_v = np.min(np.min(t))
                if max_weight < max_v:
                    max_weight = max_v
                if min_weight > min_v:
                    min_weight = min_v

        # Dynamically set the alpha values
        alpha = []
        for iLayer, t in enumerate(theta):
            num_of_weights = t.shape[0] * t.shape[1]
            alp = (1 / num_of_weights) * 100
            alp = max([0, alp])
            alp = min([1, alp])
            alpha.append(alp)

        for iLayer, t in enumerate(theta):
            if iLayer == len(theta)-1:
                output_layer = True
            else:
                output_layer = False
            neurons_positions_prim_layer = neuron_positions[iLayer]
            neurons_positions_sec_layer = neuron_positions[iLayer+1]

            # Set alpha value
            a = alpha[iLayer]
            for i, p1 in enumerate(neurons_positions_prim_layer):
                for j, p2 in enumerate(neurons_positions_sec_layer):
                    if j or output_layer:
                        connection_weight = t[i, j-1]
                        col_idx = math.floor(((connection_weight - max_weight) /
                                              (max_weight - min_weight)) * (n_col_steps - 1))
                        c = str(colors[col_idx])
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, alpha=a)

        # ===== write theta sum text =====
        y_pos = min(min_y_pos) + 2*neuron_distance
        x_pos = layer_distance/4
        for t in theta:
            theta_sum = np.sum(np.sum(t))
            theta_sum = np.floor(theta_sum*1000)/1000
            s = 'weight sum: ' + str(theta_sum)
            plt.text(x_pos, y_pos, s)
            x_pos += layer_distance

        axs.set_xticks([])  # Delete x-ticks
        axs.set_yticks([])  # Delete y-ticks
