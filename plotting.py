import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_errors(validation=True):
    train_errors = np.loadtxt(os.path.join('stats/', 'training_errors.txt'))
    if validation:
        valid_errors = np.loadtxt(os.path.join('stats/', 'validation_errors.txt'))
        errors = pd.DataFrame(np.column_stack((train_errors, valid_errors)),
                              columns=['Training errors', 'Validation errors'])
    else:
        errors = pd.DataFrame(train_errors, columns=['Training error'])

    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    error_plot = sns.lineplot(
        errors, palette='viridis'
    )
    error_plot.set(xlabel='Epoch', ylabel='Error')
    if validation:
        error_plot.set_title('Training/validation data errors during training')
    else:
        error_plot.set_title('Training data errors during training')

    plt.show()


def plot_accuracies(validation=True):
    train_accuracies = np.loadtxt(os.path.join('stats/', 'training_accuracies.txt'))
    if validation:
        valid_accuracies = np.loadtxt(os.path.join('stats/', 'validation_accuracies.txt'))
        accuracies = pd.DataFrame(np.column_stack((train_accuracies, valid_accuracies)),
                                  columns=['Training accuracy', 'Validation accuracy'])
    else:
        accuracies = pd.DataFrame(train_accuracies, columns=['Training accuracy'])

    sns.set_context('paper', font_scale=1.5)
    sns.set_style('darkgrid')
    error_plot = sns.lineplot(
        accuracies, palette='viridis'
    )
    error_plot.set(xlabel='Epoch', ylabel='Accuracy')
    if validation:
        error_plot.set_title('Training/validation data accuracy during training')
    else:
        error_plot.set_title('Training data accuracy during training')

    plt.show()
