import os

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import neural_network as nn
import file_utils as fu
import plotting as pl
import input_handling as ih


def create_network(input_size, output_size):
    hidden_layers_num = ih.get_verified_int_input('Number of hidden layers: ', True)
    hidden_layers_sizes = []
    for i in range(hidden_layers_num):
        hidden_layers_sizes.append(ih.get_verified_int_input('  * ' + str(i + 1) + ' hidden layer size: ', True))

    include_bias = ih.get_confirmation('- Include bias?')
    return nn.NeuralNetwork(
        input_size, hidden_layers_num, hidden_layers_sizes, output_size, include_bias
    )


def train(neural_network, train_data, valid_data=None, plot_accuracies=True):
    stop_condition = ih.get_verified_input('Stop condition:\n'
                                           '(1) – number of epochs\n'
                                           '(2) – total error\n'
                                           '(3) – both\n', ['1', '2', '3'])

    epochs = ih.get_verified_int_input(
        '  * Number of epochs: ', True) if stop_condition == '1' or stop_condition == '3' else None
    stop_err = ih.get_verified_float_input(
        '  * Total error: ', True) if stop_condition == '2' or stop_condition == '3' else None
    learning_rate = ih.get_verified_float_input('  * Learning rate: ', True)
    momentum = ih.get_verified_float_input('  * Momentum: ', True) if ih.get_confirmation(
        '- Include momentum?') else 0.0
    shuffle = ih.get_confirmation('- Shuffle training data?')
    print(' <Training in progress>')
    neural_network.train(
        learning_rate, train_data, valid_data, epochs, stop_err, momentum, shuffle
    )
    print(' <Training complete>')
    validation = True if valid_data is not None else False
    pl.plot_errors(filename='errors.png', validation=validation)
    if plot_accuracies:
        pl.plot_accuracies(filename='accuracies.png', validation=validation)


def test(neural_network, test_inputs, test_outputs, output_values=None):
    if output_values is not None:
        predicted_classes = []
        for i in range(len(test_inputs)):
            predicted_classes.append(
                np.array([neural_network.feedforward(test_input) for test_input in test_inputs[i]])
            )
            max_values = predicted_classes[i].max(axis=1).reshape(-1, 1)
            predicted_classes[i] = np.where(predicted_classes[i] == max_values, 1, 0)
            predicted_classes[i] = [tuple(predicted_class) for predicted_class in predicted_classes[i]]

        confusion_matrix = pd.DataFrame(data=np.zeros([len(output_values), len(output_values)]),
                                        index=list(output_values.keys()),
                                        columns=list(output_values.keys()))

        value_outputs = dict(zip(output_values.values(), output_values.keys()))
        for i in range(len(predicted_classes)):
            for j, predicted_class in enumerate(predicted_classes[i]):
                confusion_matrix.loc[value_outputs[test_outputs[i][j]], value_outputs[predicted_class]] += 1

        proper_classifications = np.array(
            [confusion_matrix.loc[output_class, output_class] for output_class in value_outputs.values()]
        ).astype(int)

        print('Total number of properly classified objects: ' + str(np.sum(proper_classifications)))
        for i, output_class in enumerate(value_outputs.values()):
            print('Properly classified objects of class ' + str(output_class) + ': ' + str(proper_classifications[i]))

        print('Confusion matrix:')
        print(confusion_matrix)

        precisions = []
        recalls = []
        for output_class in value_outputs.values():
            if confusion_matrix[output_class].sum() != 0:
                precisions.append([confusion_matrix.loc[output_class, output_class] /
                                   confusion_matrix[output_class].sum()])
            else:
                precisions.append([0])

            if confusion_matrix.loc[output_class].sum() != 0:
                recalls.append([confusion_matrix.loc[output_class, output_class] /
                                confusion_matrix.loc[output_class].sum()])
            else:
                recalls.append([0])

        precision = np.sum(np.array(precisions)) / confusion_matrix.shape[0]
        recall = np.sum(np.array(recalls)) / confusion_matrix.shape[0]
        f_measure = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('F-Measure: ' + str(f_measure))

        pl.plot_confusion_matrix(confusion_matrix, precision, recall, f_measure, filename='confusion_matrix.png')

    if not os.path.exists('stats/'):
        os.makedirs('stats/')

    samples_num = np.sum([len(i) for i in test_inputs])
    df_layers = [f'Layer {j + 1}' for j in range(len(neural_network.layers))]
    df_neurons = [f'Neuron {j + 1}' for j in range(max([len(k) for k in neural_network.layers]))]
    df_samples = [f'Sample {i + 1}' for i in range(samples_num)]
    total_in_out_exp_out = []
    total_outputs = []
    total_error_signals = []
    total_loss = []
    for i, test_input_group in enumerate(test_inputs):
        in_out_exp_out = []
        outputs = []
        error_signals = []
        loss = []
        for j, test_input in enumerate(test_input_group):
            output = mlp.feedforward(test_input)
            output_formatted = [round(i, 2) for i in output]
            expected_output = test_outputs[i][j]
            in_out_exp_out.append([test_input, output_formatted, expected_output])
            loss.append(neural_network.calculate_total_error(test_outputs[i][j]))
            _, grad = neural_network.backpropagation(test_outputs[i][j])
            error_signals.append(pd.DataFrame(grad, index=df_layers, columns=df_neurons))
            outputs.append(pd.DataFrame([[neuron.output for neuron in layer] for layer in neural_network.layers],
                                        index=df_layers,
                                        columns=df_neurons))

        total_in_out_exp_out.extend(in_out_exp_out)
        total_outputs.extend(outputs)
        total_error_signals.extend(error_signals)
        total_loss.extend(loss)

    weights = [[np.around(neuron.weights, decimals=2) for neuron in layer] for layer in neural_network.layers]
    weights_df = pd.DataFrame(weights, index=df_layers, columns=df_neurons)
    weights_df.index.name = 'Layer'
    weights_df.to_csv('stats/weights.csv')

    biases = [[neuron.bias for neuron in layer] for layer in neural_network.layers]
    biases_df = pd.DataFrame(biases, index=df_layers, columns=df_neurons)
    biases_df.index.name = 'Layer'
    biases_df.to_csv('stats/biases.csv')

    total_outputs_df = pd.concat(total_outputs, keys=df_samples, names=['Sample', 'Layer'])
    total_outputs_df.to_csv('stats/outputs.csv')

    total_error_signals_df = pd.concat(total_error_signals, keys=df_samples, names=['Sample', 'Layer'])
    total_error_signals_df.to_csv('stats/error_signals.csv', float_format='%.6f')

    total_loss_df = pd.DataFrame(total_loss, index=df_samples, columns=['Loss'])
    total_loss_df.index.name = 'Sample'
    total_loss_df.to_csv('stats/loss.csv')

    in_out_exp_out_df = pd.DataFrame(total_in_out_exp_out, index=df_samples,
                                     columns=['Input', 'Output', 'Expected output'])
    in_out_exp_out_df.index.name = 'Sample'
    in_out_exp_out_df.to_csv('stats/inputs_outputs_expected_outputs.csv')

    print(' <Testing statistics saved to files>')


if __name__ == '__main__':
    training_data = None
    validation_data = None
    test_data_inputs = None
    test_data_outputs = None
    classes_outputs = None
    plot_accuracies = True
    task_choice = ih.get_verified_input('Choose a task:\n(1) – iris dataset classification\n(2) – autoencoder\n',
                                        ['1', '2'])
    if task_choice == '1':
        # fetch dataset
        iris = fetch_ucirepo(id=53)

        # data (as pandas dataframes)
        X = iris.data.features
        y = iris.data.targets

        input_values = X.to_numpy()
        iris_classes = list(y['class'])
        classes_outputs = {'Iris-setosa': (1, 0, 0), 'Iris-versicolor': (0, 1, 0), 'Iris-virginica': (0, 0, 1)}
        iris_output_values = np.array([classes_outputs[i] for i in iris_classes])

        iris_train_inputs = []
        iris_train_outputs = []
        iris_valid_inputs = []
        iris_valid_outputs = []
        for i in range(iris_output_values.shape[1]):
            range_start = 50 * i
            range_end = range_start + 20
            iris_train_inputs.extend(input_values[range_start:range_end - 5])
            iris_train_outputs.extend(iris_output_values[range_start:range_end - 5])
            iris_valid_inputs.extend(input_values[range_end - 5:range_end])
            iris_valid_outputs.extend(iris_output_values[range_end - 5:range_end])

        training_data = [(iris_train_inputs[i], iris_train_outputs[i]) for i in range(len(iris_train_inputs))]
        validation_data = [(iris_valid_inputs[i], iris_valid_outputs[i]) for i in range(len(iris_valid_inputs))]

        test_data_inputs = []
        test_data_outputs = []
        for i in range(iris_output_values.shape[1]):
            range_start = 50 * i + 20
            range_end = range_start + 30
            test_data_inputs.append(input_values[range_start:range_end])
            test_data_outputs.append([tuple(row) for row in iris_output_values[range_start:range_end]])

    else:
        plot_accuracies = False
        test_data_inputs = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        test_data_outputs = test_data_inputs
        training_data = [(values, values) for values in test_data_inputs]
        test_data_inputs = [test_data_inputs]
        test_data_outputs = [test_data_outputs]

    choice = ih.get_confirmation('- Load network from file?')
    if choice:
        mlp = fu.load_obj(ih.get_verified_string_input('   * Enter file name: '))
        print(' <Loaded network from file>')
    else:
        mlp = create_network(len(training_data[0][0]), len(training_data[0][1]))

    while True:
        choice = ih.get_verified_input('What to do with the network:\n(1) – train\n(2) – test\n(3) – exit\n',
                                       ['1', '2', '3'])
        if choice == '1':
            train(mlp, training_data, validation_data, plot_accuracies)
        elif choice == '2':
            test(mlp, test_data_inputs, test_data_outputs, classes_outputs)
        else:
            break

    confirmation = ih.get_confirmation('- Save network to file?')
    if confirmation:
        fu.save_obj(mlp, ih.get_verified_string_input('   * Enter file name: '))
        print(' <Saved network to file>')
