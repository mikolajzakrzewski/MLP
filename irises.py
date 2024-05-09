import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os.path
import neural_network as nn
import file_utils as fu


def train(neural_network, train_data):
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
        train_data, learning_rate=learning_rate, epochs=epochs,
        stop_err=stop_err, momentum=momentum, shuffle_samples=shuffle
    )
    plot_errors()
    plot_accuracies()


def plot_errors():
    errors = np.loadtxt(os.path.join('stats/', 'training_errors.txt'))
    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    error_plot = sns.lineplot(
        errors, legend=False, color='#5D3FD3'
    )
    error_plot.set(xlabel='Epoch', ylabel='Error')
    error_plot.set_title('Training errors')
    plt.show()


def plot_accuracies():
    accuracies = np.loadtxt(os.path.join('stats/', 'training_accuracies.txt'))
    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    error_plot = sns.lineplot(
        accuracies, legend=False, color='#5D3FD3'
    )
    error_plot.set(xlabel='Epoch', ylabel='Accuracy')
    error_plot.set_title('Training accuracy')
    plt.show()


def test(neural_network, test_inputs, test_outputs, classes_outputs):
    predicted_classes = []
    for i in range(len(test_inputs)):
        predicted_classes.append(
            np.array([neural_network.feedforward(test_input) for test_input in test_inputs[i]])
        )
        max_values = predicted_classes[i].max(axis=1).reshape(-1, 1)
        predicted_classes[i] = np.where(predicted_classes[i] == max_values, 1, 0)
        predicted_classes[i] = [tuple(predicted_class) for predicted_class in predicted_classes[i]]

    confusion_matrix = pd.DataFrame(data=np.zeros([len(classes_outputs), len(classes_outputs)]),
                                    index=list(classes_outputs.keys()),
                                    columns=list(classes_outputs.keys()))

    output_classes = dict(zip(classes_outputs.values(), classes_outputs.keys()))
    for i in range(len(predicted_classes)):
        for j, predicted_class in enumerate(predicted_classes[i]):
            confusion_matrix.loc[output_classes[test_outputs[i][j]], output_classes[predicted_class]] += 1

    proper_classifications = np.array(
        [confusion_matrix.loc[output_class, output_class] for output_class in output_classes.values()]
    ).astype(int)

    print('Total number of properly classified objects: ' + str(np.sum(proper_classifications)))
    for i, output_class in enumerate(output_classes.values()):
        print('Properly classified objects of class ' + str(output_class) + ': ' + str(proper_classifications[i]))

    print('Confusion matrix:')
    print(confusion_matrix)

    precisions = []
    recalls = []
    for output_class in output_classes.values():
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

    plt.figure(figsize=(6, 8))
    sns.set_context('paper', font_scale=1.5)
    confusion_matrix_plot = sns.heatmap(
        confusion_matrix, annot=True, square=True, cbar=False,
        cmap='Purples', linecolor='black', linewidths=0.5, clip_on=False
    )
    confusion_matrix_plot.set_title('Confusion matrix')
    confusion_matrix_plot.set_xlabel('Predicted classes\n' +
                                     '\nPrecision: ' + str(precision) +
                                     '\nRecall: ' + str(recall) +
                                     '\nF-Measure: ' + str(f_measure), labelpad=10)
    confusion_matrix_plot.set_ylabel('True classes', labelpad=10)
    plt.show()


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

input_values = X.to_numpy()
iris_classes = list(y['class'])
iris_classes_outputs = {'Iris-setosa': (1, 0, 0), 'Iris-versicolor': (0, 1, 0), 'Iris-virginica': (0, 0, 1)}
iris_output_values = np.array([iris_classes_outputs[i] for i in iris_classes])

iris_train_inputs = []
iris_train_outputs = []
for i in range(iris_output_values.shape[1]):
    range_start = 50 * i
    range_end = range_start + 15
    iris_train_inputs.extend(input_values[range_start:range_end])
    iris_train_outputs.extend(iris_output_values[range_start:range_end])

iris_train_data = [(iris_train_inputs[i], iris_train_outputs[i]) for i in range(len(iris_train_inputs))]

iris_test_inputs = []
iris_test_outputs = []
for i in range(iris_output_values.shape[1]):
    range_start = 50 * i + 15
    range_end = range_start + 35
    iris_test_inputs.append(input_values[range_start:range_end])
    iris_test_outputs.append([tuple(row) for row in iris_output_values[range_start:range_end]])

if str(input('- Load network from file? (Y/N): ')) == 'N':
    hidden_layers_num = int(input('Number of hidden layers: '))
    hidden_layers_sizes = []
    for i in range(hidden_layers_num):
        hidden_layers_sizes.append(int(input('  * ' + str(i + 1) + ' hidden layer size: ')))

    include_bias = True if str(input('- Include bias? (Y/N): ')) == 'Y' else False
    mlp = nn.NeuralNetwork(
        input_values.shape[1], hidden_layers_num, hidden_layers_sizes, iris_output_values.shape[1], include_bias
    )
else:
    mlp = fu.load_obj(str(input('   * Enter file name: ')))

while True:
    choice = input('What to do with the network:\n(1) – train\n(2) – test\n(3) – exit\n')
    if choice == '1':
        train(mlp, iris_train_data)
    elif choice == '2':
        test(mlp, iris_test_inputs, iris_test_outputs, iris_classes_outputs)
    elif choice == '3':
        break
    else:
        print(' ! Invalid choice !')

if str(input('- Save network to file? (Y/N): ')) == 'Y':
    fu.save_obj(mlp, str(input('   * Enter file name: ')))
