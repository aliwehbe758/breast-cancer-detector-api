from flask import Flask
from flask_cors import CORS
from extensions import db

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:sa@localhost/breast-cancer-detector'

    db.init_app(app)

    # Import and register blueprints
    from pretrained_models import pretrained_models_bp
    from prediction import prediction_bp

    app.register_blueprint(pretrained_models_bp)
    app.register_blueprint(prediction_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(port=5001)