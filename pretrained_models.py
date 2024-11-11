from flask import Blueprint, request, jsonify
import os, json, datetime
from datetime import datetime
from app import db
from db_models import PretrainedModel
from lightly.transforms import SimCLRTransform, utils
from lightly.data import LightlyDataset

# dataset = LightlyDataset()

pretrained_models_bp = Blueprint('pretrained_models_bp', __name__)

@pretrained_models_bp.route('/pretrained-models', methods=['POST'])
def upload_pretrained_model():

    global directory_path
    pth_file = request.files.get('pth_file')
    model_py_file = request.files.get('model_py_file')
    params_py_file = request.files.get('params_py_file')
    json_data = request.form.get('json_data')

    if (
            not json_data or 'id' not in json_data or 'name' not in json_data
            or 'pth_file_name' not in json_data or 'model_py_file_name' not in json_data
            or 'params_py_file_name' not in json_data or 'folder_name' not in json_data
            or not pth_file or not model_py_file or not params_py_file
    ):
        return jsonify({"error": "Missing required fields"}), 400

    json_data = json.loads(json_data)
    folder_name = json_data['folder_name']
    directory_path = f"models/{folder_name}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    pth_file_path = os.path.join(directory_path, json_data['pth_file_name'])
    pth_file.save(pth_file_path)

    model_py_file_path = os.path.join(directory_path, json_data['model_py_file_name'])
    model_py_file.save(model_py_file_path)

    params_py_file_path = os.path.join(directory_path, json_data['params_py_file_name'])
    params_py_file.save(params_py_file_path)

    pretrained_model = PretrainedModel(
        id=None,
        name=json_data['name'],
        description=json_data['description'],
        pth_file_name=json_data['pth_file_name'],
        model_py_file_name=json_data['model_py_file_name'],
        params_py_file_name=json_data['params_py_file_name'],
        folder_name=json_data['folder_name'],
        upload_date=datetime.now(),
    )
    db.session.add(pretrained_model)
    db.session.commit()

    # Process JSON data if needed
    print("Received JSON data:", json_data)

    return jsonify(pretrained_model.to_dict()), 200

@pretrained_models_bp.route('/pretrained-models', methods=['GET'])
def get_all_models():
    pretrained_models = PretrainedModel.query.all()
    pretrained_models_list = [model.to_dict() for model in pretrained_models]
    return jsonify(pretrained_models_list), 200



