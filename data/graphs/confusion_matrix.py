import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

CLASS_NAMES = [
    "Attractive",
    "Male",
    "Young",
    "Receding_Hairline",
]

confusion_matrix = [[[6179, 2716], [2003, 7335]], [[9263, 1289], [1545, 6136]], [[1453, 2583], [886, 13311]], [[16789, 27], [1389, 28]]]


def plot_confusion_matrices(confusion_matrices, class_names, titles):
    num_matrices = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_matrices, figsize=(20, 5))
    if num_matrices == 1:
        axes = [axes]

    sns.set(font_scale=1.4)

    for i, matrix in enumerate(confusion_matrices):
        labels = np.array([[f"TP: {matrix[0][0]}", f"FN: {matrix[0][1]}"],
                           [f"FP: {matrix[1][0]}", f"TN: {matrix[1][1]}"]])

        ax = sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues",
                         xticklabels=["Not " + class_names[i], class_names[i]],
                         yticklabels=["Not " + class_names[i], class_names[i]],
                         cbar=False, ax=axes[i])

        rect = Rectangle((0, 0), 2, 2, facecolor="none", edgecolor="none", linewidth=0, clip_on=False)
        ax.add_patch(rect)
        ax.autoscale(tight=True)

        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Predicted Class')
        axes[i].set_ylabel('True Class')

    plt.tight_layout()
    plt.show()


# Generate titles for each confusion matrix
titles = [f"Confusion Matrix - {name}" for name in CLASS_NAMES]

# Call the plotting function with the confusion matrices, class names, and titles
plot_confusion_matrices(confusion_matrix, CLASS_NAMES, titles)


def plot_confusion_matrix(matrix_array, class_names, title='Confusion Matrix'):
    labels = np.array([[f"TN: {matrix_array[0][0]}", f"FP: {matrix_array[0][1]}"],
                       [f"FN: {matrix_array[1][0]}", f"TP: {matrix_array[1][1]}"]])

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # for label size
    ax = sns.heatmap(matrix_array, annot=labels, fmt="", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                     cbar=False)

    # Add a slightly rounded rectangle to the heatmap to make it have rounded corners
    rect = Rectangle((0, 0), 2, 2, facecolor="none", edgecolor="none", linewidth=0, clip_on=False)
    ax.add_patch(rect)
    ax.autoscale(tight=True)

    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()


# Sample input
confusion_matrix = np.array([[10715, 1116], [415, 8013]])
classes = ['Male', 'Female']


def plot_metrics(history):
    epochs = range(1, len(history['loss']) + 1)

    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# # Example usage
# hist = {
#     'loss': [1.3959, 1.1143, 0.9115, 0.7537, 0.6231, 0.5233, 0.4440, 0.3891, 0.3429, 0.3073],
#     'accuracy': [0.6577, 0.7682, 0.8126, 0.8427, 0.8691, 0.8854, 0.8991, 0.9055, 0.9119, 0.9172],
#     'val_loss': [1.3425, 1.1388, 0.9270, 0.7919, 0.5848, 0.4705, 0.3868, 0.3317, 0.3132, 0.2756],
#     'val_accuracy': [0.6029, 0.6871, 0.7404, 0.7429, 0.8659, 0.8926, 0.9117, 0.9212, 0.9157, 0.9236]
# }
#
# plot_metrics(hist)
