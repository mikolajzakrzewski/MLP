import numpy as np
import random


class Neuron:
    def __init__(self, input_size, input_values=None, weights=None):
        self.input_values = input_values
        if weights is None:
            self.weights = 0.5 * np.random.randn(input_size)
        else:
            self.weights = weights

        self.bias = random.uniform(-0.5, 0.5)
        if input_values is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()
        else:
            self.net_sum = 0
            self.output = 0

    def calculate_net_sum(self):
        return np.dot(self.input_values, self.weights) + self.bias

    def calculate_output(self):
        return np.tanh(self.net_sum)

    def calculate_output_derivative(self):
        return 1 - np.tanh(self.net_sum) ** 2

    def update(self, input_values=None, weights=None, bias=None):
        if input_values is not None:
            self.input_values = np.array(input_values)

        if weights is not None:
            self.weights = np.array(weights)

        if bias is not None:
            self.bias = bias

        if self.input_values is not None and self.weights is not None and self.bias is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_num, hidden_layers_sizes, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers = [np.empty(hidden_layer_size, dtype=Neuron) for hidden_layer_size in hidden_layers_sizes]
        neuron_input_size = input_layer_size
        for hidden_layer in self.hidden_layers:
            for i in range(len(hidden_layer)):
                hidden_layer[i] = Neuron(neuron_input_size)

            neuron_input_size = len(hidden_layer)

        self.output_layer_size = output_layer_size
        self.output_layer = np.empty(self.output_layer_size, dtype=Neuron)
        for i in range(len(self.output_layer)):
            self.output_layer[i] = Neuron(neuron_input_size)

    def feedforward(self, input_values):
        input_values = np.array(input_values)
        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.update(input_values, weights=None, bias=None)

            input_values = [neuron.output for neuron in hidden_layer]

        for neuron in self.output_layer:
            neuron.update(input_values, weights=None, bias=None)

        output_values = [neuron.output for neuron in self.output_layer]
        return output_values

    def backpropagation(self, output_values):
        total_error = 0
        for i, neuron in enumerate(self.output_layer):
            total_error += ((neuron.output - output_values[i]) ** 2) / 2

        total_error /= self.output_layer_size
        gradients = []
        output_layer_gradient = [
            (neuron.output - output_values[i]) *
            neuron.calculate_output_derivative() *
            neuron.input_values
            for i, neuron in enumerate(self.output_layer)
        ]
        gradients.append(output_layer_gradient)
        gradient_above = output_layer_gradient
        layer_above = self.output_layer
        for hidden_layer in reversed(self.hidden_layers):
            hidden_layer_gradient = []
            layer_above_weights = np.array([neuron.weights for neuron in layer_above])
            for i, neuron in enumerate(hidden_layer):
                neuron_weights = [weight[i] for weight in layer_above_weights]
                neuron_errors = [error[i] for error in gradient_above]
                layer_above_errors = np.dot(neuron_weights, neuron_errors)
                errors = layer_above_errors * neuron.calculate_output_derivative() * neuron.input_values
                hidden_layer_gradient.append(errors)

            gradients.append(hidden_layer_gradient)
            gradient_above = hidden_layer_gradient
            layer_above = hidden_layer

        return gradients[::-1]

    def adjust_weights(self, gradients, learning_rate):
        for i, hidden_layer in enumerate(self.hidden_layers):
            errors = np.array(gradients[i])
            weights = np.array([neuron.weights for neuron in hidden_layer])
            biases = np.array([neuron.bias for neuron in hidden_layer])
            weights -= learning_rate * errors
            biases -= learning_rate * np.sum(errors, axis=1)
            for j, neuron in enumerate(hidden_layer):
                neuron.update(input_values=None, weights=weights[j], bias=biases[j])

        errors = np.array(gradients[-1])
        weights = np.array([neuron.weights for neuron in self.output_layer])
        biases = np.array([neuron.bias for neuron in self.output_layer])
        weights -= learning_rate * errors
        biases -= learning_rate * np.sum(errors, axis=1)
        for i, neuron in enumerate(self.output_layer):
            neuron.update(input_values=None, weights=weights[i], bias=biases[i])

    def train(self, train_data, learning_rate, epochs):
        for _ in range(epochs):
            random.shuffle(train_data)
            for sample in train_data:
                self.feedforward(sample[0])
                gradients = self.backpropagation(sample[1])
                self.adjust_weights(gradients, learning_rate)
