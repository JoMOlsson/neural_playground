import numpy as np


class AdamOptimizer:
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters  # List of parameters (numpy arrays)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step

        # Initialize first and second moment vectors
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]

    def set_parameters(self, parameters):
        self.parameters = parameters

        # Initialize first and second moment vectors
        if len(self.m) != len(self.parameters):
            self.m = [np.zeros_like(param) for param in parameters]
            self.v = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        self.t += 1
        updated_parameters = []

        for i in range(len(self.parameters)):
            # g = gradients[i].flatten()
            g = gradients[i]

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.parameters[i] -= param_update
            updated_parameters.append(self.parameters[i])

        return updated_parameters
