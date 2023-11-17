# README.md for classifier-ui

## Project Overview

This React application is designed for image classification using AI. Users can upload an image, and the app will classify it based on attributes gender, attractiveness, age, and hairline status. This frontend interacts with a separate Flask-based API server that processes the image and returns predictions.

## Requirements

Node.js
npm
A running instance of the Flask-based model-api server, refer to the [model-api's README](https://github.com/crxig-rxberts/AI-Classifier-Project/blob/main/model-api/READ_ME.md)


## Running the Application Locally

### `npm start`

Runs the app in the development mode.
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.
You may also see any lint errors in the console.


## Using the Application

**Upload an Image:**
Use the ImageUploader component to upload an image. The app will display a preview of the image.

**Get Predictions:**
Click the "Get Prediction" button. The app will send the image to the Flask API server for processing, and predictions will be displayed in a table format.

## Troubleshooting

Ensure Node.js and npm are correctly installed.
Check if the Flask API server is running and accessible.
Confirm the API endpoint URL is correct in the App.js file.
Check browser console for any errors if the app doesn't behave as expected.


