import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, multilabel_confusion_matrix)
import seaborn as sns
from tabulate import tabulate
from tensorflow.keras.models import load_model
from constants import CLASS_NAMES
from functions import init_test_data_generator

OPPOSITE_ATTRIBUTES = {
    'Male': 'Female',
    'Attractive': 'Unattractive',
    'Receding_Hairline': 'Full_Hair',
    'Young': 'Old',
}


def load_and_predict(model_path, test_data_generator):
    model = load_model(model_path)
    predictions = model.predict(test_data_generator)
    return predictions


def calculate_metrics(true_labels, predictions, threshold=0.5):
    binary_predictions = (predictions >= threshold).astype(int)
    precision = precision_score(true_labels, binary_predictions, average=None)
    recall = recall_score(true_labels, binary_predictions, average=None)
    f1 = f1_score(true_labels, binary_predictions, average=None)

    accuracies = []
    for i in range(true_labels.shape[1]):
        accuracies.append(accuracy_score(true_labels[:, i], binary_predictions[:, i]))

    return precision, recall, f1, accuracies, binary_predictions


def plot_confusion_matrix(conf_matrix, attribute):
    opposite_attribute = OPPOSITE_ATTRIBUTES[attribute]

    labels = np.array([[f"TP: {conf_matrix[0][0]}", f"FN: {conf_matrix[0][1]}"],
                       [f"FP: {conf_matrix[1][0]}", f"TN: {conf_matrix[1][1]}"]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix, annot=labels, fmt="", cmap="Blues",
                xticklabels=[attribute, opposite_attribute],
                yticklabels=[attribute, opposite_attribute],
                cbar=False, ax=ax)

    rect = Rectangle((0, 0), 2, 2, facecolor="none", edgecolor="none", linewidth=0, clip_on=False)
    ax.add_patch(rect)
    ax.autoscale(tight=True)

    ax.set_title(f'Confusion Matrix: {attribute}')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    plt.tight_layout()
    plt.show()


def plot_metrics_table(metrics, attribute):
    # Create a figure and axis to plot the table
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the figure size as needed

    # Create the table and customize its appearance
    table = plt.table(cellText=metrics,
                      colLabels=["Metric", "Score (%)"],
                      loc='center',
                      cellLoc='center',  # Center align the text in cells
                      colColours=["blue"] * 2,  # Color for column headers
                      colWidths=[0.3, 0.2],  # Adjust column widths
                      )

    # Modify the table properties for better visual appeal
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust the font size
    table.scale(1.2, 1.5)  # Scale the table size

    # Set title and turn off the axis
    plt.title(f'Metrics: {attribute}')
    plt.axis('off')

    # Adjust layout for better fit
    plt.tight_layout()

    plt.show()


def main():
    model_path = '../../models/final/final_model.h5'
    datagen, generator, df_exists = init_test_data_generator()
    predictions = load_and_predict(model_path, generator)
    true_labels = df_exists[CLASS_NAMES].values
    precision_scores, recall_scores, f1_scores, accuracy_scores, binary_predictions = calculate_metrics(true_labels, predictions)

    for i, attr in enumerate(CLASS_NAMES):
        conf_matrix = multilabel_confusion_matrix(true_labels[:, i], binary_predictions[:, i])[0]
        metrics = [
            ["Precision", round(precision_scores[i] * 100, 2)],
            ["Recall", round(recall_scores[i] * 100, 2)],
            ["F1 Score", round(f1_scores[i] * 100, 2)],
            ["Accuracy", round(accuracy_scores[i] * 100, 2)]  # Use per-class accuracy
        ]

        # Plot and display/save confusion matrix
        plt.figure()
        plot_confusion_matrix(conf_matrix, attr)
        plt.show()  # or plt.savefig(f'{attr}_conf_matrix.png')

        # Plot and display/save metrics table
        plt.figure(figsize=(6, 2))
        plot_metrics_table(metrics, attr)
        plt.show()  # or plt.savefig(f'{attr}_metrics_table.png')

main()
