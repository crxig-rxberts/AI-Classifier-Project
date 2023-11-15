from flask import Blueprint, request, jsonify
from service.model_service import ModelService

api = Blueprint('api', __name__)
model_service = ModelService()


@api.route('/predict', methods=['POST'])
def predict():
    try:
        prediction = model_service.predict(request.json.get('image'))
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
