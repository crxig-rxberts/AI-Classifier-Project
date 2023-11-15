import numpy as np
import seaborn as sns
from keras.src.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os
import pandas as pd

from constants import METRIC_DESCRIPTIONS, CLASS_NAMES, CLASS_COLOURS, ACTIVATION_FUNCTIONS, ACTIVATION_COLOURS


def get_metrics_from_models(models, test_generator, y_test, model_dir):
    """
    Calculate and store metrics for each class for a list of models.

    :param models: List of model file names.
    :param test_generator: The test data generator.
    :param y_test: True labels for the test data.
    :param model_dir: Directory where models are stored.
    :return: Dictionary of metrics for each class.
    """
    class_metrics = {
        label: {'log_loss': [], 'accuracy': [], 'recall': [], 'precision': [], 'f1_score': [], 'specificity': []} for
        label in CLASS_NAMES}

    for model_name in models:
        model = load_model(model_dir + model_name)
        steps = np.ceil(test_generator.n / test_generator.batch_size)
        y_pred = model.predict(test_generator, steps=steps)
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_test_trimmed = y_test[:len(y_pred)]

        for i, label in enumerate(CLASS_NAMES):
            class_y_test = y_test_trimmed[:, i]
            class_y_pred = y_pred[:, i]
            class_y_pred_binary = y_pred_binary[:, i]

            class_metrics[label]['log_loss'].append(log_loss(class_y_test, class_y_pred))
            class_metrics[label]['accuracy'].append(accuracy_score(class_y_test, class_y_pred_binary))
            class_metrics[label]['recall'].append(recall_score(class_y_test, class_y_pred_binary))
            class_metrics[label]['precision'].append(precision_score(class_y_test, class_y_pred_binary))
            class_metrics[label]['f1_score'].append(f1_score(class_y_test, class_y_pred_binary))

            tn, fp, fn, tp = confusion_matrix(class_y_test, class_y_pred_binary).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            class_metrics[label]['specificity'].append(specificity)

    return class_metrics


def init_test_data_generator():
    """
    Load labels from attribute.csv, initialize an ImageDataGenerator, and create a DataFrameIterator.

    :return: Initialized ImageDataGenerator, DataFrameIterator, and the DataFrame with existing images.
    """
    batch_size = 32
    target_size = (200, 200)
    csv_file_path = '../attribute.csv'
    image_dir = '../partitioned-data/test/'
    class_names = ['Attractive', 'Male', 'Young', 'Receding_Hairline']

    # Load the CSV with image names and labels
    df = pd.read_csv(csv_file_path)
    df['exists'] = df['image_id'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))
    df_exists = df[df['exists']].copy()
    df_exists[class_names] = (df_exists[class_names].values == 1).astype(int)

    # Create an instance of ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1. / 255)

    # Create a DataFrameIterator
    generator = datagen.flow_from_dataframe(
        dataframe=df_exists,
        directory=image_dir,
        x_col='image_id',
        y_col=class_names,
        class_mode='raw',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    return datagen, generator, df_exists


def plot_metrics(class_metrics, x_values, x_label, title):
    """
    Plot graphs for combined metrics across all classes.

    :param class_metrics: Dictionary of metrics for each class.
    :param x_values: List of x-axis values (e.g., batch sizes or epochs).
    :param x_label: Label for the x-axis (e.g., 'Batch Size' or 'Number of Epochs').
    :param title: Title for the plot.
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    axes = axes.ravel()

    for i, (metric, description) in enumerate(METRIC_DESCRIPTIONS.items()):
        for label in CLASS_NAMES:
            axes[i].plot(x_values, class_metrics[label][metric], marker='o', label=label, color=CLASS_COLOURS[label])
        axes[i].set_title(f'{metric.capitalize()}: {description}')
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend(title='Classes')
        axes[i].grid(True)

    # Delete unused subplots
    for i in range(len(METRIC_DESCRIPTIONS), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_activation_function_metrics(activation_metrics, title):
    """
    Plot bar graphs for all metrics across classes for different models.

    :param activation_metrics: Dictionary of metrics for each class.
    :param title: Title for the plot.
    """
    n_metrics = len(METRIC_DESCRIPTIONS)
    n_models = 3
    bar_width = 0.15
    index = np.arange(n_metrics) * (n_models + 1) * bar_width

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    for i, (class_name, metrics) in enumerate(activation_metrics.items()):
        for j, metric in enumerate(METRIC_DESCRIPTIONS):
            values = metrics[metric]
            for k, value in enumerate(values):
                # Plot each value as a separate bar
                ax.bar(index[j] + (i * n_models + k) * bar_width, value, bar_width,
                       label=f'{class_name} Model {k + 1}' if j == 0 else "",
                       color=CLASS_COLOURS[class_name])

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Metrics by Class and Model')
    ax.set_xticks(index + bar_width * n_models / 2)
    ax.set_xticklabels([metric.capitalize() for metric in METRIC_DESCRIPTIONS])
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_activation_metrics(metrics_data):
    activation_functions = ['ReLU', 'Sigmoid', 'Tanh']
    metrics = ['log_loss', 'accuracy', 'recall', 'precision', 'f1_score', 'specificity']
    classes = list(metrics_data.keys())

    # Create a DataFrame for easier plotting
    data = []
    for metric in metrics:
        for cls in classes:
            for i, activation in enumerate(activation_functions):
                data.append({
                    'Class': cls,
                    'Activation Function': activation,
                    'Metric': metric,
                    'Value': metrics_data[cls][metric][i]
                })
    df = pd.DataFrame(data)

    # Plotting
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='Value', hue='Activation Function', data=df[df['Metric'] == metric])
        plt.title(f'{metric.capitalize()} by Class and Activation Function')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.legend(title='Activation Function')
        plt.show()


def plot_final_vs_b32_metrics(metrics_data):
    activation_functions = ['Final Model', 'BS 10, EPOCH 4, Model']
    metrics = ['log_loss', 'accuracy', 'recall', 'precision', 'f1_score', 'specificity']
    classes = list(metrics_data.keys())

    # Create a DataFrame for easier plotting
    data = []
    for metric in metrics:
        for cls in classes:
            for i, activation in enumerate(activation_functions):
                data.append({
                    'Class': cls,
                    'Activation Function': activation,
                    'Metric': metric,
                    'Value': metrics_data[cls][metric][i]
                })
    df = pd.DataFrame(data)

    # Plotting
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='Value', hue='Activation Function', data=df[df['Metric'] == metric])
        plt.title(f'{metric.capitalize()} Final V BS_32_E_4 Model')
        plt.ylabel('Score')
        plt.xlabel('Class')
        plt.legend(title='Model')
        plt.show()
