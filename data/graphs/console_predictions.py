import os
import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


def predict_on_test():
    def load_image(img_path, show=False):
        img = image.load_img(img_path, target_size=(178, 218))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor

    def print_predictions(predictions, labels):
        print("Predictions:")
        for i, label in enumerate(labels):
            print(f"{label}: {predictions[0][i] * 100:.2f}%")

    # Load the model
    model_path = '../../models/epoch/e_28_model.h5'
    model = load_model(model_path)

    # Labels (you should have the same order as your training data)
    label_columns = ['Attractive', 'Male', 'Young', 'Receding_Hairline']

    # Let the user select an image
    test_images_dir = '../partitioned-data/test/'
    all_test_images = [img for img in os.listdir(test_images_dir) if img.endswith('.jpg')]
    selected_images = random.sample(all_test_images, 5)  # Randomly select 5 images to choose from

    print("Please select a number corresponding to the test images:")
    for i, img_name in enumerate(selected_images):
        print(f"{i}: {img_name}")

    selected_index = int(input("Enter the number of the image you want to predict: "))
    selected_image_path = os.path.join(test_images_dir, selected_images[selected_index])

    # Load and preprocess the image
    new_image = load_image(selected_image_path)

    # Predict the probabilities for each label
    preds = model.predict(new_image)

    # Print predictions
    print_predictions(preds, label_columns)


predict_on_test()
