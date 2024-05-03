import numpy as np
import random


class Neuron:
    def __init__(self, input_size, input_values=None, weights=None, include_bias=True):
        self.input_values = input_values
        if weights is None:
            self.weights = 0.5 * np.random.randn(input_size)
        else:
            self.weights = weights

        self.include_bias = include_bias
        if include_bias:
            self.bias = random.uniform(-0.5, 0.5)
        else:
            self.bias = 0.0

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

        if bias is not None and self.include_bias:
            self.bias = bias

        if self.input_values is not None and self.weights is not None and self.bias is not None:
            self.net_sum = self.calculate_net_sum()
            self.output = self.calculate_output()


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layers_num, hidden_layers_sizes, output_layer_size, include_bias=True):
        self.input_layer_size = input_layer_size
        self.hidden_layers_num = hidden_layers_num
        neuron_input_size = input_layer_size
        layers_sizes = hidden_layers_sizes
        layers_sizes.append(output_layer_size)
        self.layers_sizes = layers_sizes
        self.layers = [np.empty(layer_size, dtype=Neuron) for layer_size in layers_sizes]
        for layer in self.layers:
            layer_size = len(layer)
            for i in range(layer_size):
                layer[i] = Neuron(neuron_input_size, include_bias=include_bias)

            neuron_input_size = layer_size

        self.include_bias = include_bias

    def feedforward(self, values):
        values = np.array(values)
        for layer in self.layers:
            for neuron in layer:
                neuron.update(values)

            values = [neuron.output for neuron in layer]

        return values

    def backpropagation(self, output):
        total_err = 0
        for i, neuron in enumerate(self.layers[-1]):
            total_err += ((neuron.output - output[i]) ** 2) / 2

        total_err /= self.layers_sizes[-1]
        weight_grads = []
        bias_grads = []
        weight_grad = []
        bias_grad = []
        for i, neuron in enumerate(self.layers[-1]):
            error_signal = (neuron.output - output[i]) * neuron.calculate_output_derivative()
            bias_grad.append(error_signal)
            errors = error_signal * neuron.input_values
            weight_grad.append(errors)

        weight_grads.append(weight_grad)
        bias_grads.append(bias_grad)
        grad_above = weight_grad
        layer_above = self.layers[-1]
        for hidden_layer in reversed(self.layers[:-1]):
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

        return total_err, weight_grads[::-1], bias_grads[::-1]

    def adjust(self, weight_grads, prev_weight_grads, bias_grads, prev_bias_grads, learning_rate, momentum):
        for i, layer in enumerate(self.layers):
            weight_errs = np.array(weight_grads[i])
            weights = np.array([neuron.weights for neuron in layer])
            weights -= learning_rate * weight_errs
            bias_errs = np.array(bias_grads[i])
            biases = np.array([neuron.bias for neuron in layer])
            biases -= learning_rate * bias_errs
            if prev_weight_grads is not None and prev_bias_grads is not None:
                prev_weight_errs = np.array(prev_weight_grads[i])
                weights -= momentum * learning_rate * prev_weight_errs
                prev_bias_errs = np.array(prev_bias_grads[i])
                biases -= momentum * learning_rate * prev_bias_errs

            for j, neuron in enumerate(layer):
                neuron.update(input_values=None, weights=weights[j], bias=biases[j])

    def epoch(self, train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples):
        if shuffle_samples:
            random.shuffle(train_data)

        for sample in train_data:
            self.feedforward(sample[0])
            total_err, weight_grads, bias_grads = self.backpropagation(sample[1])
            if stop_err is not None and total_err < stop_err:
                return prev_weight_grads, prev_bias_grads, True

            self.adjust(weight_grads, prev_weight_grads, bias_grads, prev_bias_grads, learning_rate, momentum)
            prev_weight_grads, prev_bias_grads = weight_grads, bias_grads

        if stop_err is not None:
            return prev_weight_grads, prev_bias_grads, False
        else:
            return prev_weight_grads, prev_bias_grads

    def train(self, train_data, learning_rate, epochs=None, stop_err=None, momentum=0.0, shuffle_samples=True):
        prev_weight_grads = None
        prev_bias_grads = None

        if epochs is None and stop_err is None:
            raise ValueError("Either epochs or stop_err must be specified")

        if epochs is not None and stop_err is None:
            for _ in range(epochs):
                weight_grads, bias_grads = self.epoch(
                    train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples
                )
                prev_weight_grads, prev_bias_grads = weight_grads, bias_grads
        else:
            while True:
                weight_grads, bias_grads, stop_err_reached = self.epoch(
                    train_data, prev_weight_grads, prev_bias_grads, learning_rate, stop_err, momentum, shuffle_samples
                )
                if stop_err_reached:
                    return

                prev_weight_grads, prev_bias_grads = weight_grads, bias_grads
