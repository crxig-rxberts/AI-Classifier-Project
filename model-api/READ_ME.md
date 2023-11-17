# README.md for model-api

## Project Overview

This project is a Flask-based API for image prediction. It uses a pretrained Keras model to predict attributes from images, such as attractiveness, gender, age, and hairline (receding or full). 

The API accepts base64 encoded images, processes them, and returns predictions with confidence percentages.

## Requirements

- Python 3.x
- Flask
- TensorFlow
- Keras
- Pillow

## Run Local Server

1. `pip install flask tensorflow keras pillow flask-cors`
2. `python app.py`
3. App should now be listening on port 5000

## API Usage

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Data Format:** JSON with a base64 encoded image.
- **Example Request:**
  ```
  curl -X POST http://0.0.0.0:3000/predict -H "Content-Type: application/json" -d '{"image": "[base64_encoded_image]"}'
  ```
- **Example Response:**
  ```
  {
    "Attractive": 69.56726908683777,
    "Male": 99.83156323432922,
    "Receding Hairline": 0.13182308757677674,
    "Young": 96.62508964538574
  }
  ```
