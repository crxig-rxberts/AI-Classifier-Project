# AI-Classifier-Project

## Project Overview

This project encompasses a complete AI Image Classification system, consisting of three main components: 

1. **model-ai**: A Python-based machine learning model training application using TensorFlow and Keras for image classification.
2. **model-api**: A Flask-based API server that interfaces with a pre-trained model to give confidence based classification predictions on image data.
3. **classifier-ui**: A React front-end application that allows users to upload images and view classification results.

The system is designed to classify images based on the attributes attractiveness, gender, age, and hairline status (receding or full hair). Model-AI uses a convolutional neural network to analyze images, Model-API serves as a bridge between the model and the UI, and Classifier UI provides an interactive interface for users.

## Achievements

- **Image Classification**: Leveraging deep learning for image classification.
- **Modular Design**: Clear separation of concerns between model training, API handling, and user interface.
- **Customizability**: Flexibility to adapt and extend the model for different datasets and classification tasks.
- **User-Friendly Interface**: An intuitive UI for easy interaction with the AI model.

## Local Setup Guide

To set up and run the system locally with a pre-trained model refer to the README's of [model-api](https://github.com/crxig-rxberts/AI-Classifier-Project/blob/main/model-api/READ_ME.md) and then [classifier-ui](https://github.com/crxig-rxberts/AI-Classifier-Project/blob/main/classifier-ui/README.md)

## Usage Flow when training a new model

1. **Train the Model**: First, use the Model-AI project to train an image classification model.
2. **Start the API Server**: Next, set up Model-API to load your newly trained model at run time and then run the Flask server.
3. **Use the Classifier UI**: Finally, use the Classifier UI to upload images and view classification results.

---

This guide provides an overview of the entire system's capabilities. For detailed instructions on each component, refer to their respective READMEs.
