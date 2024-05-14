import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import neural_network as nn
import file_utils as fu
import plotting as pl


def create_network(input_size, output_size):
    hidden_layers_num = int(input('Number of hidden layers: '))
    hidden_layers_sizes = []
    for i in range(hidden_layers_num):
        hidden_layers_sizes.append(int(input('  * ' + str(i + 1) + ' hidden layer size: ')))

    include_bias = True if str(input('- Include bias? (Y/N): ')) == 'Y' else False
    return nn.NeuralNetwork(
        input_size, hidden_layers_num, hidden_layers_sizes, output_size, include_bias
    )


def train(neural_network, train_data, valid_data=None):
    stop_condition = input('Stop condition:\n'
                           '(1) – number of epochs\n'
                           '(2) – total error\n'
                           '(3) – both\n')

    epochs = int(input('  * Number of epochs: ')) if stop_condition == '1' or stop_condition == '3' else None
    stop_err = float(input('  * Total error: ')) if stop_condition == '2' or stop_condition == '3' else None
    learning_rate = float(input('  * Learning rate: '))
    momentum = float(input('  * Momentum: ')) if str(input('- Include momentum? (Y/N): ')) == 'Y' else 0.0
    shuffle = True if str(input('- Shuffle training data? (Y/N): ')) == 'Y' else False
    neural_network.train(
        learning_rate, train_data, valid_data, epochs, stop_err, momentum, shuffle
    )
    validation = True if valid_data is not None else False
    pl.plot_errors(filename='errors.png', validation=validation)
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

    fu.clear_outputs()
    for i, test_input_group in enumerate(test_inputs):
        for j, test_input in enumerate(test_input_group):
            output = mlp.feedforward(test_input)
            output_formatted = [round(i, 2) for i in output]
            expected_output = test_outputs[i][j]
            fu.save_outputs(output_formatted, expected_output)


if __name__ == '__main__':
    training_data = None
    validation_data = None
    test_data_inputs = None
    test_data_outputs = None
    classes_outputs = None
    task_choice = str(input('Choose a task:\n(1) – iris dataset classification\n(2) – autoencoder\n'))
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

    elif task_choice == '2':
        test_data_inputs = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        test_data_outputs = test_data_inputs
        training_data = [(values, values) for values in test_data_inputs]
        test_data_inputs = [test_data_inputs]
        test_data_outputs = [test_data_outputs]

    if str(input('- Load network from file? (Y/N): ')) == 'N':
        mlp = create_network(len(training_data[0][0]), len(training_data[0][1]))
    else:
        mlp = fu.load_obj(str(input('   * Enter file name: ')))

    while True:
        choice = input('What to do with the network:\n(1) – train\n(2) – test\n(3) – exit\n')
        if choice == '1':
            train(mlp, training_data, validation_data)
        elif choice == '2':
            test(mlp, test_data_inputs, test_data_outputs, classes_outputs)
        elif choice == '3':
            break
        else:
            print(' ! Invalid choice !')

    if str(input('- Save network to file? (Y/N): ')) == 'Y':
        fu.save_obj(mlp, str(input('   * Enter file name: ')))
