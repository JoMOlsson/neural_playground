import numpy as np
import matplotlib.pyplot as plt
import mnist
import copy
import os
import glob
import random
from enum import Enum
from typing import Union
from types import SimpleNamespace

try:
    from utils.activation import activate, gradient, ActivationFunction
    from utils.normalization import Normalization, normalize, denormalize
    from utils.opimizer.adam import AdamOptimizer
    from utils.visual.visualize import animate_training, create_gif_from_dump
except ModuleNotFoundError:
    from .utils.activation import activate, gradient, ActivationFunction
    from .utils.normalization import Normalization, normalize, denormalize
    from .utils.opimizer.adam import AdamOptimizer
    from .utils.visual.visualize import animate_training, create_gif_from_dump

# TODO - CLEAN UP CODE
# TODO - CHECK HOW TO SET SETTINGS CONVENIENTLY
# TODO - RE WRITE TRAINING AND REMOVE DEPRECATED


class Initialization(Enum):
    EPSILON = 1
    HE = 2

class NetworkType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


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
                                       init_method=init_method,  # Init method
                                       network_type=NetworkType.CLASSIFICATION
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
        self.visual = SimpleNamespace(orig_images = [],                # List holding the original images
                                      orig_test_images=[],  # List holding the original test images
                                      is_mnist=False)  # Boolean variable stating if mnist data-set is used

        if network_settings is not None:
            self.set_network_settings(network_settings)

        # Optimizer
        self.optimizer = AdamOptimizer(self.network.weight, learning_rate=self.params.alpha)

    def __getattr__(self, item):
        """ Finds the desired attribute by searching in the class NameSpaces

        :param item (str) Name of desired attribute to be fetched

        """
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

    def save(self, save_dir: str = '.', file_name: str = 'network'):
        """ Saves the current state of the network to the directory specified in save_dir

        :param save_dir: (str) Desired path where saved network should be stored
        :param file_name: (str) Name of file
        """
        data = {
            'theta': self.Theta,
            'network': self.network,
            'params': self.params,
            'data': self.data,
            'normalization': self.normalization,
            'optimizer': self.optimizer
                }
        np.save(os.path.join(save_dir, f'{file_name}.npy'), data)  # save to npy file

    def load_network(self, load_dir: str):
        """ Loads a previously saved network

        :param load_dir: (str) Directory to model to be loaded

        """
        data = np.load(load_dir, allow_pickle=True)
        self.network = data.item().get('network')
        self.params = data.item().get('params')
        self.data = data.item().get('data')
        self.normalization = data.item().get('normalization')
        self.optimizer = data.item().get('optimizer')
        self.Theta = data.item().get('theta')

    def set_alpha(self, alpha: float):
        """ Sets the learning rate parameter

        :param alpha: (float) Learning rate

        """
        self.params.alpha = alpha
        self.optimizer.learning_rate = alpha

    def init_network_params(self, network_size: list = None):
        """ Given a network architecture which is a list of integers determining the desired number of neurons per layer
            the method initializes the network layers with the init method specified in self.network.init_method.
            If the input variable network_size is provided, the method will assign the internal network_architecture
            variable to the given list.

        :param network_size: (list) List of integers determining how many neurons every layer should have
        """

        theta = []
        if not network_size:
            network_size = self.network.network_architecture
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
        self.optimizer = AdamOptimizer(self.network.weight, learning_rate=self.params.alpha)

    def init_weights(self):
        """ Deprecated method for setting the weights

        """
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
        self.optimizer = AdamOptimizer(self.network.weight, learning_rate=self.params.alpha)

    def set_network_settings(self, settings: dict):
        """ Given a settings dictionary the method will assign the containing values to
            the model.

        :param settings (dict) A dictionary with network settings
        """
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

    def set_activation_function(self, activation_function: Union[int, str]):
        """ Assigns the given activation function to the model.
            The input can either be a string value or an integer in the ActivationFunction Enum class.

        :param activation_function: (int, str) Desired activation function
        """
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
        """ Assigns the given activation function to the model.
            The input can either be a string value or an integer in the InitFunction Enum class.

        :param init: (int, str) Desired init function
        """

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
        """ Assigns the given output function to the model.
            The input can either be a string value or an integer in the ActivationFunction Enum class.


        :param output: (int, str) Desired output function
        """

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
        """ Assigns the given normalization function to the model.
            The input can either be a string value or an integer in the NormalizationFunction Enum class.

        :param norm_method: (int, str) Desired Normalization function
        """

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
        create_gif_from_dump()

    def set_network_architecture(self, network_architecture: list):
        """ Assigns the provided network_architecture variable to the internal class-variable self.network_architecture

        :param network_architecture: (list) List of integer corresponding to the desired size of the input layer,
                                            hidden layers and the output layer.
        :return:
        """
        self.network.network_architecture = network_architecture
        self.network.output_layer_size = network_architecture[-1]
        self.network.input_layer_size = network_architecture[0]

    def set_mnist_data(self):
        """ Imports the mnist data-set and assigns it to the train- & test-data. The mnist data-set is an open source
            data set consisting of 60000 training images of handwritten digits. The method will also assign the
            internal class-variable self.is_mnist to true.

        :return:
        """

        # Extract data
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        self.visual.orig_images = train_images
        self.visual.orig_test_images = test_images

        # Assign data
        self.set_train_data(train_images)
        self.set_train_labels(train_labels)
        self.set_test_data(test_images)
        self.set_test_labels(test_labels)
        self.visual.is_mnist = True

    def normalize_data(self, data, data_axis: int = None):
        """ Takes the provided data and normalize it using the min and the max values of the data. If the min and max
            is already provided, the stored max/min value will be used.

        :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
        :param data_axis: (int) Determines which axis should be considered to be the data axis
        :return data: (npArray) Normalized data
        """
        data = normalize(self.normalization.norm_method, data, self.normalization, data_axis)
        return data

    def de_normalize_data(self, data):
        """ Takes the provided data and un-normalizes it using the min and the max values of the data.

        :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
        :return data: (npArray) Un-Normalized data
        """
        data = denormalize(self.normalization.norm_method, data, self.normalization)

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

    @staticmethod
    def calc_mse(true_value: np.ndarray, prediction: np.ndarray):
        """ Calculates the mean square error between the true value and the prediction

        :param true_value (np.ndarray) The target value
        :param prediction (npp.ndarray) Prediction array

        """
        return np.mean((true_value - prediction) ** 2)

    def print_progress(self,
                       target: np.ndarray,
                       output: np.ndarray,
                       iteration: int,
                       total_iterations: int):
        """ Evaluates and prints the current progress to the console.

        :param target: (np.ndarray) Input data
        :param output: (np.ndarray) Output
        :param iteration: (int) Current iteration
        :param total_iterations: (int) Number of training iterations
        """
        mse = self.calc_mse(target, output)
        prompt = f"Iteration ({iteration} / {total_iterations}), cost: {mse}"

        # Calculate number of errors if network is of type classification
        if self.network.network_type == NetworkType.CLASSIFICATION:
            num_of_errors = 0
            for itr, p in enumerate(output):
                prediction = np.argmax(p)
                true_val = np.argmax(target[itr])
                if prediction != true_val:
                    num_of_errors += 1

            prompt += f", number of false predictions: {num_of_errors}, accuracy: {1 - num_of_errors / len(target)}"
        print(prompt)


    def train(self, x: np.ndarray, y: np.ndarray, num_iterations: int = 1000, print_every: int = 100):
        """ Trains the network given som input and desired output

        :param x: (np.ndarray) Input data
        :param y: (np.ndarray) Output
        :param num_iterations: (int) Number of training iterations
        :param print_every: (int) Determines how frequent the training status should be printed
        """
        if not self.optimizer.dimensions_match(self.network.weight):
            self.optimizer = AdamOptimizer(self.network.weight, learning_rate=self.params.alpha)

        # Reshape data if dimension mismatch
        if len(x.shape) < 2:
            x = x.reshape((x.shape[0], 1))

        # TODO - ADD visualization
        for iteration in range(num_iterations):
            activations, zs, h = self.forward(x=x)
            if iteration % print_every == 0:
                self.print_progress(y, h, iteration, num_iterations)
            self.back(activations, zs, y=y)

    def forward(self, x):
        """ Performs the forward propagation through the neural network.

        :param x: (numpy array) Input data of shape (m, input_size), where m is the number of examples.
        :return:
            activations: (list) List of activation values for each layer.
            zs: (list) List of z values (weighted input to activation functions) for each layer.
            output: (numpy array) Final output of the network.
        """
        # Reshape data if dimension mismatch
        if len(x.shape) < 2:
            x = x.reshape((x.shape[0], 1))

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
        for layer in reversed(range(len(self.network.weight) - 1)):
            z = z_list[layer]
            zp = gradient(self.network.activation_func, z)
            delta = np.dot(deltas[-1], self.network.weight[layer + 1].T) * zp
            deltas.append(delta)
            d_bias.append(delta.mean(axis=0))
            d_weight.append(np.dot(activations[layer].T, delta))

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

    def train_network(self, x=None, y=None, num_of_iterations=1000, visualize_training=False, print_every: int = 100):
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
        :param print_every: (int)
        :return:
        """
        historic_prediction = []  # holds the predicted values for every iteration, used for visualization
        historic_theta = []       # holds the wight matrices for every iteration, used for visualization
        cost = np.zeros([num_of_iterations, 1])      # Holding cost value per iteration
        accuracy = np.zeros([num_of_iterations, 1])  # Holding accuracy value per iteration

        # Set input and label data if not provided
        if (x is None) or (y is None):
            x = self.data.train_data
            y = self.data.train_labels

        num_of_samples = len(x)   # Samples in the training set

        # Initialize weights if they have not been initialized by user.
        self.init_weights()

        # Optimizer
        if not self.optimizer.dimensions_match(self.network.weight):
            self.optimizer = AdamOptimizer(self.Theta, learning_rate=self.params.alpha)


        # Preparing label data, every sample will be represented by a vector with the size equal to the number of
        # unique classes in the training data. The correct classification will be marked by a one while the false
        # classifications are marked with a zero for all samples. The label matrix will be of the size
        # [num_of_train_samples x num_of_unique_classes]
        if self.network.network_type == NetworkType.REGRESSION:
            y_mat = y
        else:
            if len(y.shape) < 2:
                y_mat = y
            elif y.shape[0] == 1 or y.shape[1] == 1:
                y_mat = np.zeros((y.shape[0], self.network.output_layer_size))
                for i, val in enumerate(y):
                    y_mat[i][int(val)] = 1
            else:
                y_mat = y

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
                self.Theta = self.optimizer.step(theta_grad, flatten=True)
            else:
                for i, theta_val in enumerate(theta_grad):
                    theta_grad[i] = (1/num_of_samples)*theta_val
                    t = self.params.alpha * theta_grad[i]
                    self.Theta[i] -= t.flatten()

            if self.network.network_type != NetworkType.REGRESSION:
                # Calculate and print cost
                h, c, _, _ = ANNet.forward_propagation(self, x, y_mat)

                num_of_errors = 0
                for itr, p in enumerate(h):
                    prediction = np.argmax(p)
                    y = np.argmax(y_mat[itr])
                    if not prediction == y:
                        num_of_errors += 1
                # ------------------------------ print iteration --------------------------------
                if visualize_training:
                    cost[iteration] = c
                    accuracy[iteration] = 1 - num_of_errors/num_of_samples

                    # Save the weight values for animation
                    historic_theta.append(copy.deepcopy(self.Theta))
                    historic_prediction.append(h)
                if iteration % print_every == 0:
                    print("Iteration (" + str(iteration) + "/" + str(num_of_iterations) + "), " +
                          "Cost of network: " + str(c) +
                          " , number of false predictions: " + str(num_of_errors) +
                          " , accuracy: " + str(1 - num_of_errors/num_of_samples))

        if visualize_training:

            # # ===== Dump data to json =====
            if self.visual.is_mnist:
                picture_idx = random.randint(0, len(self.data.test_labels))
                h_test, c_test, a_test, z_test = ANNet.forward_propagation(self, self.data.test_data[picture_idx, :],
                                                                           self.data.test_labels[picture_idx])
                picture = self.visual.orig_test_images[picture_idx, :, :]
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
            if self.visual.is_mnist:
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
            animate_training(x, cost, accuracy, historic_theta, historic_prediction,
                             network_architecture=self.network.network_architecture,
                             train_labels=self.data.train_labels,
                             is_mnist=self.visual.is_mnist)
