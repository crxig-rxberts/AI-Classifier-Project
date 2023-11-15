import React from 'react';
import '../App.css';

function Title() {
    return (
        <div className="title">
            <h1>AI Image Classifier</h1>
            <p>Upload an image to have it classified into</p>
            <div className="class-container">
                <span className="class-box">Male or Female</span>
                <span className="class-box">Attractive or Unattractive</span>
                <span className="class-box">Young or Old</span>
                <span className="class-box">Receding Hairline or Full Hair</span>
            </div>
        </div>
    );
}

export default Title;
