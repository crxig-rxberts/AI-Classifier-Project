import base64
import io
import keras.models
import numpy as np
from tensorflow import keras
from PIL import Image


class ModelService:

    def __init__(self):
        self.model = keras.models.load_model('../models/final/bs_32_model.h5')

    @staticmethod
    def convert_to_img_array(base64_img):
        if "base64," in base64_img:
            base64_img = base64_img.split("base64,")[1]

        img_data = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_data))
        img = img.resize((218, 178))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array /= 255.0

        return np.expand_dims(img_array, axis=0)

    def predict(self, base64_img):
        predictions = self.model.predict(self.convert_to_img_array(base64_img))
        labels = ["Attractive", "Male", "Young", "Receding Hairline"]

        result = {}

        for i, label in enumerate(labels):
            confidence_percentage = predictions.tolist()[0][i] * 100
            result[label] = confidence_percentage

        return result

