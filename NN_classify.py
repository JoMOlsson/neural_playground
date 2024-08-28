from ANNet import ANNet, NetworkType
from neural_playground.data.data_generation.data import create_circle, get_xor_data, get_square_data, get_cube_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# TEST WITH TRAINING CIRCLE WITH 25 25

train_xor = False
train_circle = True
train_1D_func = False

if train_1D_func:
    nnet = ANNet()
    n_ittr = 3000
    train_in, train_out, test_in, test_out = get_square_data() #get_cube_data()

    # Settings
    nnet.params.use_optimizer = True
    nnet.set_alpha(0.1)
    nnet.set_output_activation("LINEAR")
    nnet.set_activation_function("SIGMOID")
    nnet.network.network_type=NetworkType.REGRESSION
    network_architecture = [1, 75, 1]
    nnet.set_network_architecture(network_architecture)
    nnet.set_init_function(2)
    nnet.init_network_params(network_architecture)

    nnet.train(test_in, test_out, num_iterations=n_ittr, visualize=True)
    activations, zs, h = nnet.forward(test_in)
    plt.plot(train_in, train_out, 'b.')
    plt.plot(test_in, test_out, 'kx')
    plt.plot(test_in, h, 'rx')
    plt.show()

    MSE = []
    mse = 0
    for i in tqdm(range(n_ittr)):
        activations, zs, h = nnet.forward(x=train_in)
        mse = np.mean((train_out.reshape(-1, 1) - h) ** 2)
        nnet.back(activations, zs, y=train_out)
        MSE.append(mse)

    activations, zs, h = nnet.forward(test_in)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.array(MSE))
    ax[1].plot(train_in, train_out, 'b.')
    ax[1].plot(test_in, test_out, 'kx')
    ax[1].plot(test_in, h, 'rx')
    plt.show()

    # ----------------
    nnet.train_network(train_in, train_out, num_of_iterations=n_ittr)
    h_mat, c, a_mat, z_mat = nnet.forward_propagation(test_in)  # Forward propagate

    plt.plot(train_in, train_out, 'b.')
    plt.plot(test_in, test_out, 'kx')
    plt.plot(test_in, h_mat, 'rx')
    plt.show()

if train_xor:
    # ----- Init Artificial neural network -----
    n_net = ANNet()
    n_net.set_alpha(0.1)
    n_net.use_optimizer = False
    # ------- TEMP ------
    data, labels = get_xor_data()
    n_net.set_train_data(data)
    n_net.set_train_labels(labels)
    # n_net.set_output_activation("LINEAR")
    n_net.set_activation_function("RELU")
    n_net.set_output_activation("SIGMOID")
    n_net.set_activation_function("SIGMOID")

    n_ittr = 10000
    network_architecture = [2, 2, 1]
    n_net.set_network_architecture(network_architecture)
    n_net.init_network_params(network_architecture)
    #n_net.train_network(num_of_iterations=n_ittr, visualize_training=False)
    #h_mat, c, a_mat, z_mat = n_net.forward_propagation(data, labels)

    n_net.train(data, labels, num_iterations=n_ittr, visualize=True)
    _, _, h = n_net.forward(x=data)
    #MSE = []
    #mse = 0
    #for i in tqdm(range(n_ittr)):
    #    activations, zs, h = n_net.forward(x=data)
    #    mse = np.mean((labels.reshape(-1, 1) - h) ** 2)
    #    # if i % 1000 == 0:
    #    #     print(f"Iteration {i}, MSE: {mse}")
    #    n_net.back(activations, zs, y=labels)
    #    MSE.append(mse)
    #plt.plot(np.array(MSE))
    #plt.show()
    # n_net.train_network(num_of_iterations=n_ittr)
    print("")

# ------- TEMP ------
if train_circle:
    # ----- Set data ----
    n_net = ANNet()
    n_net.set_alpha(0.1)

    mnist = False
    if mnist:
        n_net.set_mnist_data()
    else:
        train_data, train_labels = create_circle(3, 8, create_internal_circles=False)
        n_net.set_train_data(train_data)
        n_net.set_train_labels(train_labels)
        test_data, test_labels = create_circle(3, 8, create_internal_circles=False,
                                               n_inner=1000, n_outer= 1500)

    if len(test_labels.shape) < 2:
        y_mat = test_labels
    elif test_labels.shape[0] == 1 or test_labels.shape[1] == 1:
        y_mat = np.zeros((test_labels.shape[0], n_net.network.output_layer_size))
        for i, val in enumerate(test_labels):
            y_mat[i][int(val)] = 1
    else:
        y_mat = test_labels

    # ----- Init network parameters -----
    input_layer_size = n_net.input_layer_size
    output_layer_size = n_net.output_layer_size
    network_architecture = [input_layer_size, 15, output_layer_size]
    network_architecture = [input_layer_size, 25, 25, output_layer_size]
    n_net.set_network_architecture(network_architecture)
    # n_net.set_output_activation("LINEAR")
    n_net.set_output_activation("SIGMOID")
    n_net.set_activation_function("SIGMOID")
    n_net.set_init_function(2)
    Theta = n_net.init_network_params(network_architecture)
    n_net.params.use_optimizer = True

    # ----- Back propagate -----
    n_ittr = 1000

    ts = time.time()
    n_net.set_alpha(0.1)
    n_net.params.use_optimizer = True
    n_net.train(x=train_data, y=train_labels, num_iterations=n_ittr, print_every=100, visualize=True)
    activations, zs, h = n_net.forward(x=test_data)
    num_of_errors_1 = 0
    PRED1 = np.array([])
    for itr, p in enumerate(h):
        prediction = np.argmax(p)

        y = np.argmax(y_mat[itr])
        if prediction != y:
            PRED1 = np.append(PRED1, np.nan)
            num_of_errors_1 += 1
        else:
            PRED1 = np.append(PRED1, p[prediction])

    t1 = time.time()
    n_net.train_network(num_of_iterations=10, visualize_training=True)
    h_mat, c, a_mat, z_mat = n_net.forward_propagation(test_data)  # Forward propagate
    num_of_errors_2 = 0
    PRED2 = np.array([])
    for itr, p in enumerate(h_mat):
        prediction = np.argmax(p)

        y = np.argmax(y_mat[itr])
        if prediction != y:
            PRED2 = np.append(PRED2, np.nan)
            num_of_errors_2 += 1
        else:
            PRED2 = np.append(PRED2, p[prediction])
    t2 = time.time()
    print(f"Time for training with old method: {t1-ts} : {num_of_errors_1}, Time for training with old method: {t2-t1} : {num_of_errors_2}")
    plt.hist(PRED1[np.isreal(PRED1)])
    plt.show()
    for i in range(0):
        activations, zs, h = n_net.forward(x=train_data)
        mse = np.mean((train_labels - h) ** 2)
        if i % 10 == 0:
            print(f"Iteration {i}, MSE: {mse}")
        n_net.back(activations, zs, y=train_labels)

    #n_net.train_network(num_of_iterations=n_ittr, visualize_training=False)
    #n_net.save("/home/joakim/Development/code/")
    #n_net = ANNet()
    #n_net.load_network("/home/joakim/Development/code/network.npy")
    #n_net.train_network(num_of_iterations=n_ittr, visualize_training=False)

    # ----- Create Visualization -----
    #n_net.create_gif()
