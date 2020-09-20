import numpy as np
from math import e
import matplotlib.pyplot as plt
import random
import mnist
import math
import imageio
from colour import Color
from PIL import Image
import copy
import os
import glob



def sigmoid(z):
    # Accepts float, int or np-arrays
    g = 1 / (1 + e**(-z))
    return g


def sigmoid_gradient(z):
    g = sigmoid(z)*(1 - sigmoid(z))
    return g


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class ANNet:

    def __init__(self):
        self.default_hidden_layer_size = 15  # Default number of neurons in hidden layer
        self.epsilon = 0.12                  # Initialization value for weight matrices
        self.alpha = 10                      # Learning rate
        self.network_size = []               # Default network size
        self.Theta = []                      # Holding weight matrices
        self.feature_mean_vector = []        # (list) average values in data per feature
        self.feature_var_vector = []         # (list) variance values in data per feature
        self.data_min = None                 # (float) min value in input data
        self.feature_min_vector = []         # (list) min values in data per feature
        self.data_max = None                 # (float) max value in input data
        self.feature_max_vector = []         # (list) max values in data per feature
        self.num_of_train_samples = None     # (int) Number of train samples
        self.input_shape = []                # Original shape of input data
        self.num_of_test_samples = None      # (int) Number of labels in test data-set
        self.train_data = np.array([])       # Hold train data
        self.test_data = np.array([])        # Holds test data
        self.train_labels = []               # Labels for train-data
        self.test_labels = []                # Labels for test-data
        self.output_layer_size = []          # Output layer size
        self.input_layer_size = None         # Input layer size
        self.network_architecture = None     # Network architecture
        self.gif_created = False             # Boolean variable if gif was ever created

    def init_network_params(self, network_size):
        theta = []
        self.network_size = network_size
        for i in range(0, len(network_size)-1):
            n = (network_size[i]+1) * network_size[i+1]
            t = np.random.uniform(-self.epsilon, self.epsilon, size=(1, n))
            theta.extend(t)
        self.Theta = theta
        return theta

    def create_gif(self):
        self.gif_created = True
        print("Creating gif ... ")
        cur_path = os.path.dirname(os.path.realpath(__file__))
        temp_path = os.path.join(cur_path, 'temp')
        image_directory = glob.glob(temp_path + '/*.png')
        images = []
        for im_dir in image_directory:
            images.append(Image.open(im_dir))
        images[0].save('movie.gif', save_all=True, append_images=images, duration=30)

    def set_network_architecture(self, network_architecture):
        self.network_architecture = network_architecture

    def set_mnist_data(self):
        # Extract data
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()
        test_images = mnist.test_images()
        test_labels = mnist.test_labels()

        # Assign data
        self.set_train_data(train_images)
        self.set_train_labels(train_labels)
        self.set_test_data(test_images)
        self.set_test_labels(test_labels)

    def normalize_data(self, data):
        data = np.array(data)
        data = data.astype('float64')
        # Assign mean and var vector if not available
        if 'self.feature_mean_vector' not in locals():
            feature_mean_vector = []
            for iFeat in range(0, data.shape[1]):
                feature_mean_vector.append(np.mean(data[:, iFeat]))
            self.feature_mean_vector = feature_mean_vector
        if 'self.feature_var_vector' not in locals():
            feature_var_vector = []
            for iFeat in range(0, data.shape[1]):
                feature_var_vector.append(np.var(data[:, iFeat]))
            self.feature_var_vector = feature_var_vector
        if 'self.feature_min_vector' not in locals():
            self.data_min = np.min(np.min(data))
            feature_min_vector = []
            for iFeat in range(0, data.shape[1]):
                feature_min_vector.append(np.min(data[:, iFeat]))
            self.feature_min_vector = feature_min_vector
        if 'self.feature_max_vector' not in locals():
            self.data_max = np.max(np.max(data))
            feature_max_vector = []
            for iFeat in range(0, data.shape[1]):
                feature_max_vector.append(np.max(data[:, iFeat]))
            self.feature_max_vector = feature_max_vector
        # Normalize data
        for iData in range(0, data.shape[0]):
            d = np.array(data[iData, :])
            d_norm = (2*(d - self.data_min)) / (self.data_max - self.data_min) - 1
            data[iData, :] = d_norm
        return data

    def set_train_data(self, data):
        train_data = []

        if len(data.shape) == 3:
            self.input_layer_size = data.shape[1] * data.shape[2]
            self.num_of_train_samples = data.shape[0]
            self.input_shape = [data.shape[1], data.shape[2]]
            # Reshape data
            for i in range(0, len(data)):
                train_data.append(np.array(np.matrix.flatten(data[i])))
        elif len(data.shape) == 2:
            self.input_layer_size = data.shape[1]
            self.num_of_train_samples = data.shape[0]
            self.input_shape = [1, data.shape[1]]
            train_data = data
        train_data = self.normalize_data(np.array(train_data))
        self.train_data = train_data

    def set_test_data(self, data):
        test_data = []
        self.num_of_test_samples = data.shape[0]
        if len(data.shape) == 3:
            self.input_shape = [data.shape[1], data.shape[2]]
            # Reshape data
            for i in range(0, len(data)):
                test_data.append(np.matrix.flatten(data[i]))
        elif len(data.shape) == 2:
            self.input_shape = [1, data.shape[1]]
        test_data = self.normalize_data(np.array(test_data))
        self.test_data = np.array(test_data)

    def set_train_labels(self, labels):
        self.train_labels = labels
        if 'self.output_layer_size' not in locals():
            self.output_layer_size = len(np.unique(labels))

    def set_test_labels(self, labels):
        self.test_labels = labels
        if 'self.output_layer_size' not in locals():
            self.output_layer_size = len(np.unique(labels))

    def forward_propagation(self, x, y=None):
        num_of_samples = len(x) if len(x.shape) > 1 else 1
        bias = np.ones((num_of_samples, 1))  # Create bias units
        a = x

        # move layer by layer
        a_list = []
        if num_of_samples > 1:
            a = np.vstack([bias.T, a.T])
            a = a.T
        else:
            a = np.append(bias, a)
        a_list.append(a)

        z_list = []  # List to hold sigmoid values
        for i, theta in enumerate(self.Theta):
            t = theta.reshape(self.network_size[i]+1, self.network_size[i+1])  # Reshape parameter matrix

            z = np.matmul(a, t)
            z_list.append(z)
            a = sigmoid(z)

            if num_of_samples > 1:
                a = np.column_stack([bias, a])  # add bias unit
            else:
                a = np.append(bias, a)

            a_list.append(a)

        h = a
        if len(h.shape) > 1:
            h = h[:, 1:]  # pop bias unit
        else:
            h = h[1:]
        a_list.pop(-1)  # Last layer in a is equivalent to h

        # Calculate cost (LSM)
        if y is None:
            # labels has not been provided
            c = None
        else:
            c = (1 / (2 * num_of_samples)) * np.sum((h - y) ** 2)

        return h, c, a_list, z_list

    def draw_and_predict(self):
        sample_idx = random.randint(0, self.test_data.shape[0])
        x = self.test_data[sample_idx, :]
        y = self.test_labels[sample_idx]
        y_hat = np.zeros((self.output_layer_size, 1))
        y_hat[y] = 1
        h, c, a, z = ANNet.forward_propagation(self, x, y_hat)
        prediction = np.argmax(h)
        print("Network predicts value: " + str(prediction))
        im = x.reshape(28, 28)
        plt.imshow(im)

    def back_propagation(self, x=None, y=None, num_of_iterations=1000):
        historic_prediction = []  # holds the predicted values for every iteration, used for visualization
        historic_theta = []       # holds the wight matrices for every iteration, used for visualization

        # Set input and label if data is not provided
        if (x is None) or (y is None):
            x = self.train_data
            y = self.train_labels

        num_of_samples = len(x)   # Samples in the training set

        # Initialize weights if they have not been initialized by user.
        if not self.Theta:
            if self.network_architecture is None:
                print("Network weights has not been initialized by user!, default hidden layer  of size " +
                      str(self.default_hidden_layer_size) + " has been applied")
                input_layer_size = self.input_layer_size
                output_layer_size = self.output_layer_size
                network_architecture = [input_layer_size, self.default_hidden_layer_size, output_layer_size]
                self.network_architecture = network_architecture
            self.init_network_params(self.network_architecture)

        # Preparing label data
        y_mat = np.zeros((y.shape[0], self.output_layer_size))
        for i, val in enumerate(y):
            y_mat[i][int(val)] = 1

        cost = np.zeros([num_of_iterations, 1])      # Holding cost value per iteration
        accuracy = np.zeros([num_of_iterations, 1])  # Holding accuracy value per iteration
        for iteration in range(0, num_of_iterations):
            # Init weight gradient matrices
            theta_grad = []
            for iLayer, theta in enumerate(self.Theta):
                theta = theta.reshape(self.network_size[iLayer]+1, self.network_size[iLayer+1])
                theta_grad.append(np.zeros(theta.shape))

            h_mat, c, a_mat, z_mat = ANNet.forward_propagation(self, x, y_mat)  # Forward propagate

            delta = -(y_mat-h_mat)
            for iLayer in range(len(a_mat)-1, -1, -1):
                z = z_mat[iLayer]
                if not iLayer == len(a_mat)-1:
                    index = iLayer + 1
                    theta = self.Theta[index]
                    t = theta.reshape(self.network_size[index] + 1, self.network_size[index + 1])
                    t = t[1:, :]

                    delta_weight = np.dot(t, delta.T)
                    sig_z = sigmoid_gradient(z)
                    delta = delta_weight * sig_z.T
                    delta = delta.T
                else:
                    delta = delta * sigmoid_gradient(z)

                a = a_mat[iLayer]
                th_grad = np.dot(a.T, delta)
                theta_grad[iLayer] += th_grad

            # Update weights
            for i, theta_val in enumerate(theta_grad):
                theta_grad[i] = (1/num_of_samples)*theta_val
                t = self.alpha * theta_grad[i]
                self.Theta[i] -= t.flatten()

            # Save theta values for animation
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
            # self.draw_and_predict()
            # ------------------------------ print --------------------------------
            cost[iteration] = c
            accuracy[iteration] = 1 - num_of_errors/num_of_samples
            print("Iteration (" + str(iteration) + "/" + str(num_of_iterations) + "), " +
                  "Cost of network: " + str(c) +
                  " , number of false predictions: " + str(num_of_errors) +
                  " , accuracy: " + str(1 - num_of_errors/num_of_samples))

        # ----- Plot Cost/Accuracy progression
        plt.close()
        plt.plot(cost)
        plt.plot(accuracy)
        plt.pause(0.001)
        plt.ion()
        plt.show()
        # -------
        self.visualize_training(x, cost, accuracy, historic_theta, historic_prediction)

    def visualize_training(self, x, cost, accuracy, historic_theta, historic_prediction):

        fig, axs = plt.subplots(2, 2)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        temp_folder = self.create_temp_folder()

        # Extract min/max weight
        absolute_weight = False
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

        image_directory = []
        num_of_digits = self.determine_number_of_digits(len(historic_prediction))
        for iteration in range(0, len(historic_prediction)):
            title_str = 'Cost: ' + str(cost[iteration]) + ' , Accuracy: ' + str(accuracy[iteration]) + \
                        ' , Itteration: ' + str(iteration)
            fig.suptitle(title_str)
            print(iteration)
            h = historic_prediction[iteration]
            col = []
            col_true = []
            for i_sample in range(0, h.shape[0]):
                col.append('#ff5959' if np.argmax(h[i_sample, :]) else '#34ebd6')
                col_true.append('#ff5959' if self.train_labels[i_sample] else '#34ebd6')

            # ----- Prediction plot -----
            axs[0, 0].scatter(x[:, 0], x[:, 1], c=col)
            axs[0, 0].axis('equal')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

            # ----- Cost / Accuracy plot -----
            axs[1, 0].plot(cost[0: iteration])
            axs[1, 0].plot(accuracy[0: iteration])

            # ----- Reference plot -----
            axs[0, 1].scatter(x[:, 0], x[:, 1], c=col_true)
            axs[0, 1].axis('equal')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            # ----- Network plot -----
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

    def create_temp_folder(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        temp_folder = os.path.join(cur_path, 'temp')
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # flush folder from png
        files = os.listdir(temp_folder)
        for file in files:
            if file.endswith(".png"):
                os.remove(os.path.join(temp_folder, file))
        return temp_folder

    def determine_number_of_digits(self, num_of_iterations):
        if num_of_iterations < 10:
            number_of_zeros = 1
        elif num_of_iterations < 100:
            number_of_zeros = 2
        elif num_of_iterations < 1000:
            number_of_zeros = 3
        elif num_of_iterations < 10000:
            number_of_zeros = 4
        elif num_of_iterations < 100000:
            number_of_zeros = 5
        elif num_of_iterations < 1000000:
            number_of_zeros = 6
        else:
            number_of_zeros = 7
        return number_of_zeros

    def append_with_zeros(self, num_of_digits, iteration):
        if iteration < 10:
            number_of_zeros = num_of_digits - 1
        elif iteration < 100:
            number_of_zeros = num_of_digits - 2
        elif iteration < 1000:
            number_of_zeros = num_of_digits - 3
        elif iteration < 10000:
            number_of_zeros = num_of_digits - 4
        elif iteration < 100000:
            number_of_zeros = num_of_digits - 5
        elif iteration < 1000000:
            number_of_zeros = num_of_digits - 6
        else:
            number_of_zeros = num_of_digits - 7

        appended_name = ''
        for i in range(0, number_of_zeros):
            appended_name += '0'
        appended_name += str(iteration)
        return appended_name

    def draw_network(self, theta, axs, max_weight=None, min_weight=None):
        neuron_radius = 5
        neuron_distance = 10
        layer_distance = 10 * neuron_distance

        # Reconstruct weight matrices
        network_architecture = []
        i_layer = len(theta)
        for i_layer, t in enumerate(theta):
            theta[i_layer] = t.reshape(self.network_size[i_layer] + 1, self.network_size[i_layer + 1])
            network_architecture.append(theta[i_layer].shape[0])
        network_architecture.append(theta[i_layer].shape[1])

        # Calculate position lists
        x = 0
        neuron_positions = []
        min_y_pos = []
        for N_neurons in network_architecture:
            p = []
            # Find vertical start position
            if N_neurons % 2:
                n = math.floor(N_neurons/2)
                y0 = (2*neuron_radius + neuron_distance)*n
            else:
                n = N_neurons/2
                y0 = (neuron_distance/2 + neuron_radius) + (neuron_distance + 2*neuron_radius)*(n-1)

            y = y0
            for i in range(0, N_neurons):
                pos_tuple = (x, y)
                p.append(pos_tuple)
                y -= (neuron_distance + 2*neuron_radius)
            neuron_positions.append(p)
            x += layer_distance
            min_y_pos.append(y)

        # ======== Get colormap ========
        blue = Color("blue")
        n_col_steps = 1000
        colors = list(blue.range_to(Color("red"), n_col_steps))

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

        # extract min/max values
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

        for iLayer, t in enumerate(theta):
            if iLayer == len(theta)-1:
                output_layer = True
            else:
                output_layer = False
            neurons_positions_prim_layer = neuron_positions[iLayer]
            neurons_positions_sec_layer = neuron_positions[iLayer+1]

            for i, p1 in enumerate(neurons_positions_prim_layer):
                for j, p2 in enumerate(neurons_positions_sec_layer):
                    if j or output_layer:
                        connection_weight = t[i, j-1]
                        col_idx = math.floor(((connection_weight - max_weight) /
                                              (max_weight - min_weight)) * (n_col_steps - 1))
                        c = str(colors[col_idx])
                        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, alpha=0.5)

        # ===== write theta sum text =====
        y_pos = min(min_y_pos) + 2*neuron_distance
        x_pos = layer_distance/4
        for t in theta:
            theta_sum = np.sum(np.sum(t))
            theta_sum = np.floor(theta_sum*1000)/1000
            s = 'weight sum: ' + str(theta_sum)
            plt.text(x_pos, y_pos, s)
            x_pos += layer_distance

        axs.set_xticks([])
        axs.set_yticks([])
