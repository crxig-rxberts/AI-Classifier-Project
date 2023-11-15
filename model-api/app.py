from flask_cors import CORS
from flask import Flask
from controller import controller

app = Flask(__name__)
CORS(app)
app.register_blueprint(controller.api)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
