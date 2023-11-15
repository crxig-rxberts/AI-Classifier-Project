import React from 'react';

function PredictButton(props) {
    return (
        <button className="predict-button" onClick={props.onUpload}>
            Get Prediction
        </button>
    );
}

export default PredictButton;
