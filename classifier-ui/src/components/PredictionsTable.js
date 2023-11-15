import React from 'react';
import '../App.css';

const PredictionsTable = ({ predictions }) => {
    if (!predictions) return null;
    const getOppositeClass = (className, percentage) => {
        const oppositeClasses = {
            'Male': 'Female',
            'Attractive': 'Unattractive',
            'Receding Hairline': 'Full Hair',
            'Young': 'Old'
        };

        return percentage >= 50 ? className : oppositeClasses[className];
    };
    const getAdjustedPercentage = (percentage) => {
        return percentage >= 50 ? percentage : 100 - percentage;
    };
    const getThresholdName = (percentage) => {
        if (percentage >= 90) {
            return 'Highly Likely';
        } else if (percentage >= 70 && percentage < 90) {
            return 'Potentially';
        } else {
            return 'Maybe';
        }
    };
    return (
        <div className="predictions-table">
            <table>
                <tbody>
                {Object.keys(predictions).map((key) => {
                    const percentage = predictions[key];
                    return (
                        <tr key={key}>
                            <td>{getOppositeClass(key, percentage)}</td>
                            <td>{getAdjustedPercentage(percentage).toFixed(2)}%</td>
                            <td>{getThresholdName(getAdjustedPercentage(percentage))}</td>
                        </tr>
                    );
                })}
                </tbody>
            </table>
        </div>
    );
};

export default PredictionsTable;
