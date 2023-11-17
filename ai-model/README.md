# README.md for AI Image Classifier

## Project Overview

This project is an AI Image Classifier developed in Python using TensorFlow and Keras. It is designed to classify images based on various attributes such as attractiveness, gender, age, and hairline status but can be expanded easily to classify more classes. The classifier uses convolutional neural networks (CNNs) and can be trained on a dataset with labeled images.

## Features

- **Customizable CNN Model:** The classifier uses a sequential CNN model that can be customized in terms of the number of layers, activation functions, and learning rate.
- **Data Augmentation:** Implements image data augmentation for better training performance.
- **Performance Metrics:** Calculates various metrics like precision, recall, F1-score, ROC-AUC, specificity, and confusion matrix for each class.
- **Flexibility:** Can be trained with any number of classes based on the labeled dataset.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Scikit-Learn
- Numpy
- Pandas

## Model Training and Validation

1. **Data Preparation:**
   The model requires a dataset with labeled images. The labels should be in a CSV file, and images should be organized in directories.

2. **Training:**
   The model can be trained by calling the `train_and_save_model` function with appropriate parameters like the number of epochs, batch size, learning rate, and activation function.

3. **Validation Metrics:**
   After training, the model computes various validation metrics to evaluate performance.

4. **Model Saving:**
   The trained model can be saved to a specified directory for later use or deployment.

## Usage

- **Initialize the Classifier:** Create an instance of `ImageClassifier` with the desired number of classes and other parameters.
- **Train the Model:** Use the `train` method to train the model on your dataset.
- **Predictions:** After training, use the `predict` method to classify new images.
- **Get Validation Metrics:** Retrieve detailed validation metrics using the `get_validation_metrics` function.

## Customization

- **Number of Classes:** Define the number of classes based on your dataset.
- **Model Parameters:** Customize the model's architecture, learning rate, and activation functions as per your requirements.
- **Data Augmentation:** Adjust the data augmentation settings in the `train_and_save_model` function.

---

This README provides an overview of the AI Image Classifier's capabilities, requirements, and usage instructions. For detailed implementation, refer to the provided Python scripts.
