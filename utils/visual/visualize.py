import numpy as np
import matplotlib.pyplot as plt
import math
from colour import Color
import os
import random
from typing import Union
from PIL import Image
import glob


def animate_training(x: Union[list, np.ndarray],
                     cost: Union[list, np.ndarray],
                     accuracy: Union[list, np.ndarray],
                     historic_theta: Union[list, np.ndarray],
                     historic_prediction: Union[list, np.ndarray],
                     train_labels: Union[list, np.ndarray],
                     network_architecture: Union[list, np.ndarray],
                     is_mnist: bool = False,
                     input_shape: list = None,
                     original_images: list = None,
                     absolute_weight: bool = False,
                     ask_user: bool = True
                     ) -> None:
    """ The method will visualize the training session in a (2x2) subplot displaying cost/accuracy, sample
        predictions and neural network for the whole training session. The output will be stored with .png
        files stored in a temp folder.


    :param x:                   (npArray) Training input data set
    :param cost:                (list)    List holding the cost values per iteration
    :param accuracy:            (list)    List holding the accuracy values per iteration
    :param historic_theta:      (list)    List holding all weight matrices per iteration
    :param historic_prediction: (list)    List holding all predictions for every iteration
    :param train_labels (list)
    :param network_architecture (list)
    :param is_mnist             (bool)    Boolean flag to indicate if mnist is to be animated
    :param input_shape (list)
    :param original_images (list)
    :param absolute_weight (bool) If True the method will take the max and min over all weight values for all
    iterations into consideration when coloring the weights
    :param ask_user (bool) If true, the user will be asked which training iterations that should be rendered
    :return:
    """

    fig, axs = plt.subplots(2, 2)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    temp_folder = create_temp_folder()  # Create a temp-folder to store the training visualisation content

    # Extract min/max weight
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
    num_of_pics_m = 20  # Number of images vertically
    num_of_pics_n = 40  # Number of images horizontally
    tot_num_of_pics = num_of_pics_m * num_of_pics_n
    if is_mnist:
        picture_idx = random.sample(range(0, len(train_labels)), tot_num_of_pics)

    image_directory = []
    num_of_digits = determine_number_of_digits(len(historic_prediction))

    n_start = 0
    n_end = len(historic_prediction)
    if ask_user:
        try:
            n_start = int(input("Select starting index for rendering"))
            n_end = int(input("Select ending index for rendering"))
        except ValueError:
            print("Provided input is not valid, only integers.")

    # Make sure the range is valid
    if n_start > n_end:
        n_start, n_end = n_end, n_start

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
        if is_mnist:
            output_classes = list(set(train_labels))

            num_of_false_predictions = np.zeros((len(output_classes),), dtype=int)
            num_of_true_predictions = np.zeros((len(output_classes),), dtype=int)
            outcome = np.ones((len(h),), dtype=int)
            for i_sample in range(0, len(h)):
                y = train_labels[i_sample]
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
                col_true.append('#ff5959' if train_labels[i_sample] else '#34ebd6')

            axs[0, 0].scatter(x[:, 0], x[:, 1], c=col)
            axs[0, 0].axis('equal')
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

        # ----- Cost / Accuracy plot [1,0] -----
        axs[1, 0].plot(cost[0: iteration], label='Cost')
        axs[1, 0].plot(accuracy[0: iteration], label='Accuracy')
        axs[1, 0].legend()

        # ----- Reference plot  [0, 1]-----
        if is_mnist:
            classification_result = outcome[picture_idx]
            bt = 4  # Boarder thickness

            m = input_shape[0] + 2 * bt
            n = input_shape[1] + 2 * bt
            validation_image = np.zeros((num_of_pics_m * m,
                                         num_of_pics_n * n,
                                         3))
            def_im = np.zeros((m, n, 3))

            # Set frame
            im_true = def_im.copy()
            border_color_channel = 1  # Green
            im_true[0:bt - 1, 0:n, border_color_channel] = 1  # top
            im_true[m - bt + 1:m, 0:n, border_color_channel] = 1  # right
            im_true[0:m, 0:bt - 1, border_color_channel] = 1  # left
            im_true[0:m, n - bt + 1:n, border_color_channel] = 1  # bottom

            im_false = def_im.copy()
            border_color_channel = 0  # Red
            im_false[0:bt - 1, 0:n, border_color_channel] = 1  # top
            im_false[m - bt + 1:m, 0:n, border_color_channel] = 1  # right
            im_false[0:m, 0:bt - 1, border_color_channel] = 1  # left
            im_false[0:m, n - bt + 1:n, border_color_channel] = 1  # bottom

            idx = 0
            for i in range(0, num_of_pics_m):
                for j in range(0, num_of_pics_n):
                    sample_idx = picture_idx[idx]
                    mnist_im = original_images[sample_idx]
                    im = True if classification_result[idx] else False

                    # Set image
                    im[bt:m - bt, bt:n - bt, 0] = mnist_im / 255
                    im[bt:m - bt, bt:n - bt, 1] = mnist_im / 255
                    im[bt:m - bt, bt:n - bt, 2] = mnist_im / 255

                    # ----- Set overall picture -----
                    validation_image[i * m:(i + 1) * m, j * n:(j + 1) * n, :] = im
                    idx += 1
            axs[0, 1].imshow(validation_image)
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 1].set_title(str(round(tot_num_of_pics / len(train_labels) * 1000) / 10) + '% of the data')
        else:
            axs[0, 1].scatter(x[:, 0], x[:, 1], c=col_true)
            axs[0, 1].axis('equal')
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

        # ----- Network plot [1, 1]-----
        if absolute_weight:
            draw_network(historic_theta[iteration], axs[1, 1], network_architecture, max_weight, min_weight)
        else:
            draw_network(historic_theta[iteration], axs[1, 1], network_architecture)

        # ----- Save figure -----
        im_dir = append_with_zeros(num_of_digits, iteration) + '.png'
        im_dir = os.path.join(temp_folder, im_dir)
        image_directory.append(im_dir)
        plt.savefig(im_dir)

        # ----- Clear figures -----
        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[1, 1].clear()
        axs[1, 0].clear()
    plt.close(fig)

def create_gif_from_dump(duration: int = 10) -> None:
    """ Will fetch all *.png files located under the temp folder and creates an animated gif to visualize the
        training sequence.

        :param duration (int) Duration of Gif

    :return:
    """

    def extract_number(filename):
        return int(filename.split('/')[-1].split('.')[0])  # Assuming Unix-like path separator '/'

    print("Creating gif ... ")
    cur_path = os.path.dirname(os.path.realpath(__file__))
    temp_folder = os.path.abspath(os.path.join(cur_path, os.pardir, os.pardir, 'data', 'data_dump', 'images'))
    image_directory = glob.glob(temp_folder + '/*.png')

    # Sort image paths based on numeric part of filename
    image_directory_sorted = sorted(image_directory, key=extract_number)

    images = []
    i = 0
    for im_dir in image_directory_sorted:
        print(str(i / len(image_directory)))
        if i < 1000:
            images.append(Image.open(im_dir))
        i += 1
    print("Images loaded")
    images[0].save(os.path.abspath(os.path.join(cur_path, os.pardir, os.pardir, 'data', 'data_dump', 'movie.gif')),
                   save_all=True, append_images=images, duration=duration)


def create_temp_folder():
    """ The method will create a folder called temp in the current file-directory if it already does no exist. If
        the folder already exists it will flush it by deleting all .png files located in the folder.

    :return temp_folder: (str) path to temp folder
    """
    current_path = os.path.dirname(os.path.realpath(__file__))  # Get current directory path
    # Temp folder for storing image
    temp_folder = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, 'data', 'data_dump', 'images'))
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)  # Create if not already exist

    # flush folder from png
    files = os.listdir(temp_folder)  # List all files in temp folder
    for file in files:
        if file.endswith(".png"):
            os.remove(os.path.join(temp_folder, file))  # Delete if file is png
    return temp_folder


def determine_number_of_digits(num_of_iterations):
    """ Determines the number of digits needed in the image naming to hold all iterations. This is used to get the
        images naturally sorted based on name even in a windows file-explorer.

    :param num_of_iterations: (int) Total number of iterations
    :return num_of_digits:    (int) Total number of needed digits in the file-naming
    """
    return math.ceil(math.log10(num_of_iterations + 0.001))


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


def draw_network(theta, axs, network_size, max_weight=None, min_weight=None):
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
    :param network_size (list)
    :param max_weight: (float)           maximum value used for color scaling
    :param min_weight: (float)           minimum value used for color scaling

    :return:
    """
    neuron_radius = 1  # Neuron radius
    neuron_distance = 15  # Distance between neurons in each layer
    layer_distance = 10 * neuron_distance  # Distance between each layer

    # Reconstruct weight matrices
    network_architecture = []
    i_layer = len(theta)
    for i_layer, t in enumerate(theta):
        theta[i_layer] = t.reshape(network_size[i_layer] + 1,
                                   network_size[i_layer + 1])
        network_architecture.append(theta[i_layer].shape[0])
    network_architecture.append(theta[i_layer].shape[1])

    # Calculate position lists
    use_max_y = True
    max_y = (max(network_architecture) - 1) * (neuron_distance + 2 * neuron_radius)

    x = 0  # x-position of the first neural layer
    neuron_positions = []  # List holding the positions of all neurons
    min_y_pos = []  # List holding the minimum y-value for each layer
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
        x += layer_distance  # update the x-position for the new neuron layer
        min_y_pos.append(y)  # Store the minimal y-position of the current layer

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
        if iLayer == len(theta) - 1:
            output_layer = True
        else:
            output_layer = False
        neurons_positions_prim_layer = neuron_positions[iLayer]
        neurons_positions_sec_layer = neuron_positions[iLayer + 1]

        # Set alpha value
        a = alpha[iLayer]
        for i, p1 in enumerate(neurons_positions_prim_layer):
            for j, p2 in enumerate(neurons_positions_sec_layer):
                if j or output_layer:
                    connection_weight = t[i, j - 1]
                    col_idx = math.floor(((connection_weight - max_weight) /
                                          (max_weight - min_weight)) * (n_col_steps - 1))
                    c = str(colors[col_idx])
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, alpha=a)

    # ===== write theta sum text =====
    y_pos = min(min_y_pos) + 2 * neuron_distance
    x_pos = layer_distance / 4
    for t in theta:
        theta_sum = np.sum(np.sum(t))
        theta_sum = np.floor(theta_sum * 1000) / 1000
        s = 'weight sum: ' + str(theta_sum)
        plt.text(x_pos, y_pos, s)
        x_pos += layer_distance

    axs.set_xticks([])  # Delete x-ticks
    axs.set_yticks([])  # Delete y-ticks
