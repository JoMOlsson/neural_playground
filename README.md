# Neural_playground


A platform for deep-learning written from scratch used for constructing, training, deploying and visualizing neural networks.

![rendered_network](/assets/rendered_network.gif)
![rendered_network](/assets/training.gif)

The user specifies the desired network architecture and feeds it with arbitrary data-sets for training and validation. The project also supports visualization of the training session and rendering of the resulting networks with support of graphical rendering in blender. 

## Example
```
n_net = ANNet()
n_net.set_mnist_data()

# ----- Init network parameters -----
input_layer_size = n_net.input_layer_size
output_layer_size = n_net.output_layer_size
network_architecture = [input_layer_size, 40, 40, output_layer_size]
n_net.set_network_architecture(network_architecture)
Theta = n_net.init_network_params(network_architecture)

# ----- Back propagate -----
n_net.train_network(num_of_iterations=2000)

```

![rendered_network](/assets/training_xy.gif)
