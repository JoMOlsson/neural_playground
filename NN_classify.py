from ANNet import ANNet
from data import create_circle

# ----- Init Artificial neural network -----
n_net = ANNet()
# n_net.create_gif()

# ----- Set data -----
# n_net.set_mnist_data()
train_data, train_labels = create_circle(3, 8)
n_net.set_train_data(train_data)
n_net.set_train_labels(train_labels)

# ----- Init network parameters -----
input_layer_size = n_net.input_layer_size
output_layer_size = n_net.output_layer_size
network_architecture = [input_layer_size, 25, 25, output_layer_size]
n_net.set_network_architecture(network_architecture)
Theta = n_net.init_network_params(network_architecture)

# ----- Back propagate -----
n_net.back_propagation(num_of_iterations=6000)

# ----- Create Visualization -----
n_net.create_gif()
print("")
