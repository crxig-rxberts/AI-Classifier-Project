import React, { useState } from 'react';
import './App.css';
import Title from './components/Title';
import ImageUploader from './components/ImageUploader';
import PredictionsTable from './components/PredictionsTable';
import PredictButton from './components/PredictButton.js'

function App() {
    const [base64Image, setBase64Image] = useState(null);
    const [predictions, setPredictions] = useState(null);

    const handleImageUpload = (base64) => {
        setBase64Image(base64);
    }

    const handlePredictClick = async () => {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });
        const data = await response.json();
        setPredictions(data);
    }

    return (
        <div className="app">
            <Title />
            <div className="content">
                <ImageUploader onImageUpload={handleImageUpload} />
                <PredictionsTable predictions={predictions} />
            </div>
            <PredictButton onUpload={handlePredictClick} />
        </div>
    );
}

export default App;
