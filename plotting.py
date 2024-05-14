import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_errors(validation=True, filename=None):
    if os.stat(os.path.join('stats/', 'training_errors.txt')).st_size == 0:
        return

    train_errors = np.loadtxt(os.path.join('stats/', 'training_errors.txt'), ndmin=1)
    if train_errors.shape[0] == 1:
        return

    if validation:
        valid_errors = np.loadtxt(os.path.join('stats/', 'validation_errors.txt'))
        errors = pd.DataFrame(np.column_stack((train_errors, valid_errors)),
                              columns=['Training errors', 'Validation errors'],
                              index=[(i + 1) * 10 for i in range(len(train_errors))])
    else:
        errors = pd.DataFrame(train_errors, columns=['Training error'],
                              index=[(i + 1) * 10 for i in range(len(train_errors))])

    plt.figure(figsize=(8, 8))
    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    error_plot = sns.lineplot(
        errors, palette='viridis'
    )
    error_plot.set(ylim=(-0.1, None))
    error_plot.set(xlabel='Epoch', ylabel='Error')
    if validation:
        error_plot.set_title('Training/validation data errors during training')
    else:
        error_plot.set_title('Training data errors during training')

    if filename is not None:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

        plt.savefig(os.path.join('plots/', filename))

    plt.show()


def plot_accuracies(validation=True, filename=None):
    if os.stat(os.path.join('stats/', 'training_accuracies.txt')).st_size == 0:
        return

    train_accuracies = np.loadtxt(os.path.join('stats/', 'training_accuracies.txt'), ndmin=1)
    if train_accuracies.shape[0] == 1:
        return

    if validation:
        valid_accuracies = np.loadtxt(os.path.join('stats/', 'validation_accuracies.txt'))
        accuracies = pd.DataFrame(np.column_stack((train_accuracies, valid_accuracies)),
                                  columns=['Training accuracy', 'Validation accuracy'],
                                  index=[(i + 1) * 10 for i in range(len(train_accuracies))])
    else:
        accuracies = pd.DataFrame(train_accuracies, columns=['Training accuracy'],
                                  index=[(i + 1) * 10 for i in range(len(train_accuracies))])

    plt.figure(figsize=(8, 8))
    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    accuracy_plot = sns.lineplot(
        accuracies, palette='viridis'
    )
    accuracy_plot.set(ylim=(-0.1, 1.1))
    accuracy_plot.set(xlabel='Epoch', ylabel='Accuracy')
    if validation:
        accuracy_plot.set_title('Training/validation data accuracy during training')
    else:
        accuracy_plot.set_title('Training data accuracy during training')

    if filename is not None:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

        plt.savefig(os.path.join('plots/', filename))

    plt.show()


def plot_confusion_matrix(confusion_matrix, precision, recall, f_measure, filename=None):
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

    if filename is not None:
        if not os.path.exists('plots/'):
            os.makedirs('plots/')

        plt.savefig(os.path.join('plots/', filename))

    plt.show()
