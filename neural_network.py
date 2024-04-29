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
        return 1 / (1 + np.exp(-self.net_sum))

    def calculate_output_derivative(self):
        return self.output * (1 - self.output)

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

    def backpropagation(self, output):
        cost = 0
        for i, neuron in enumerate(self.output_layer):
            cost += ((neuron.output - output[i]) ** 2) / 2

        cost /= self.output_layer_size
        weight_grads = []
        bias_grads = []
        weight_grad = []
        bias_grad = []
        for i, neuron in enumerate(self.output_layer):
            error_signal = (neuron.output - output[i]) * neuron.calculate_output_derivative()
            bias_grad.append(error_signal)
            errors = error_signal * neuron.input_values
            weight_grad.append(errors)

        weight_grads.append(weight_grad)
        bias_grads.append(bias_grad)
        grad_above = weight_grad
        layer_above = self.output_layer
        for hidden_layer in reversed(self.hidden_layers):
            weight_gradient = []
            bias_gradient = []
            layer_above_weights = np.array([neuron.weights for neuron in layer_above])
            for i, neuron in enumerate(hidden_layer):
                neuron_weights = [weight[i] for weight in layer_above_weights]
                neuron_errors = [error[i] for error in grad_above]
                error_signal = np.dot(neuron_weights, neuron_errors) * neuron.calculate_output_derivative()
                bias_gradient.append(error_signal)
                errors = error_signal * neuron.input_values
                weight_gradient.append(errors)

            weight_grads.append(weight_gradient)
            bias_grads.append(bias_gradient)
            grad_above = weight_gradient
            layer_above = hidden_layer

        return weight_grads[::-1], bias_grads[::-1]

    def adjust(self, weight_grads, bias_grads, learning_rate):
        for i, hidden_layer in enumerate(self.hidden_layers):
            weight_errs = np.array(weight_grads[i])
            bias_errs = np.array(bias_grads[i])
            weights = np.array([neuron.weights for neuron in hidden_layer])
            biases = np.array([neuron.bias for neuron in hidden_layer])
            weights -= learning_rate * weight_errs
            biases -= learning_rate * bias_errs
            for j, neuron in enumerate(hidden_layer):
                neuron.update(input_values=None, weights=weights[j], bias=biases[j])

        weight_errs = np.array(weight_grads[-1])
        bias_errs = np.array(bias_grads[-1])
        weights = np.array([neuron.weights for neuron in self.output_layer])
        biases = np.array([neuron.bias for neuron in self.output_layer])
        weights -= learning_rate * weight_errs
        biases -= learning_rate * bias_errs

        for i, neuron in enumerate(self.output_layer):
            neuron.update(input_values=None, weights=weights[i], bias=biases[i])

    def train(self, train_data, learning_rate, epochs):
        for _ in range(epochs):
            random.shuffle(train_data)
            for sample in train_data:
                self.feedforward(sample[0])
                weight_grads, bias_grads = self.backpropagation(sample[1])
                self.adjust(weight_grads, bias_grads, learning_rate)

