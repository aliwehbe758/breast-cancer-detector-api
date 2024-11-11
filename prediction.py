from flask import Blueprint, request, jsonify
import torch
from torchvision import transforms
import io
import base64
import logging
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from app import db
import datetime
from datetime import datetime
from db_models import PredictionH, PredictionD, PretrainedModel
import custom_transforms
import importlib.util

# Setup logging
logging.basicConfig(level=logging.DEBUG)

transform = transforms.Compose([
    custom_transforms.ResizeWithPad(new_shape=(224, 224)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    custom_transforms.CustomNormalize()
])

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/history', methods=['GET'])
def get_history():
    prediction_hs = PredictionH.query.all()
    history = [prediction_h.to_dict2() for prediction_h in prediction_hs]
    return jsonify(history), 200

@prediction_bp.route('/history/<int:id>', methods=['GET'])
def get_header_details(id):
    history = PredictionH.query.get(id)
    if history is None:
        return jsonify({"error": "Header not found"}), 404

    return jsonify(history.to_dict1()), 200

# Load the Python file dynamically and call the get_model function
def load_model_from_file(model_py_file_name, pth_file_name):
    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location("model_module", model_py_file_name)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Assuming the file has a function named 'get_model'
    model = model_module.get_model(pth_file_name)
    return model

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    json_data = request.json

    pretrained_model_db = PretrainedModel.query.get(json_data['pretrained_model_id'])
    model_py_file_name = './models/' + pretrained_model_db.folder_name + '/' + pretrained_model_db.model_py_file_name
    pth_file_name = pretrained_model_db.pth_file_name
    model, cam_extractor = load_model_from_file(model_py_file_name, pth_file_name)

    benign_count = 0
    malignant_count = 0
    normal_count = 0
    prediction_h = PredictionH(
        id=None,
        uploaded_folder_name=json_data['uploaded_folder_name'],
        total_images=len(json_data['images']),
        date=datetime.now(),
        pretrained_model_id=json_data['pretrained_model_id']
    )
    for item in json_data['images']:
        image = item['image']
        image_data = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_name = item['file_name']
        # Predict using the model
        probability, prediction, heatmap_image  = predict_image(image, model, cam_extractor)
        if prediction == 0:
            benign_count += 1
        elif prediction == 1:
            malignant_count += 1
        else:
            normal_count += 1
        print(f"name: {image_name}, prediction: {str(prediction)}, probability: {str(probability)}")
        prediction_d = PredictionD(
            id=None,
            file_name=image_name,
            prediction='Benign' if prediction == 0 else 'Malignant' if prediction == 1 else 'Normal',
            probability=probability,
            image=image_data,
            grad_cam=base64.b64decode(heatmap_image)
        )
        prediction_h.details.append(prediction_d)

    prediction_h.benign_count=benign_count
    prediction_h.malignant_count = malignant_count
    prediction_h.normal_count = normal_count

    # Add to the session and commit
    db.session.add(prediction_h)
    db.session.commit()

    return jsonify(prediction_h.to_dict1())

def predict_image(image, model, cam_extractor):
    """
    Predict the class of an image using the loaded PyTorch model and generate Grad-CAM heatmap.
    """
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # Ensure the model is in evaluation mode
    output = model(image_tensor)

    # Get the predicted class and probability
    probability, predicted = torch.max(F.softmax(output, dim=1), dim=1)

    # Generate GradCAM for the predicted class
    activation_map = cam_extractor(predicted.item(), output)

    # Since activation_map is a list, extract the first element (assuming only one map)
    activation_map = activation_map[0]

    # Convert the activation map to a NumPy array
    activation_map = activation_map.cpu().numpy()

    # Check if the activation map is 3D (C, H, W) and reduce it to 2D (H, W)
    if activation_map.ndim == 3:
        activation_map = np.mean(activation_map, axis=0)  # Average across the channel dimension
    elif activation_map.ndim != 2:
        raise ValueError("Unexpected shape for the activation map.")

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(activation_map, (image.width, image.height))

    # Normalize the heatmap between 0 and 1
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Convert the heatmap to an 8-bit image (0-255)
    heatmap = np.uint8(255 * heatmap)

    # Apply the JET color map for a standard heatmap look (red to blue)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Threshold to keep only significant regions
    threshold = 125  # Adjust this value to focus on the most important regions
    mask = heatmap[:, :, 2] > threshold

    # Convert the heatmap to RGBA
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)

    # Convert the heatmap and original image to PIL Image for further processing
    heatmap = Image.fromarray(heatmap)
    original_img = image.convert("RGBA")

    # Invert mask to highlight important regions instead of the less important ones
    inverted_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')

    # Blend the heatmap with the original image with specified opacity
    opacity = 0.5  # Set the desired opacity level (between 0 and 1)
    heatmap_with_opacity = Image.blend(original_img, heatmap, opacity)

    # Apply the blended heatmap only to important regions
    overlay = Image.composite(heatmap_with_opacity, original_img, inverted_mask)

    # Convert the result image to base64 for JSON response
    buffered = io.BytesIO()
    overlay.save(buffered, format="PNG")
    result_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return probability.item(), predicted.item(), result_img_base64


# def predict_image(image):
#     """
#     Predict the class of an image using the loaded PyTorch model and generate Grad-CAM heatmap.
#     """
#     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#     image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model.eval()  # Ensure the model is in evaluation mode
#     output = model(image_tensor)
#
#     # Get the predicted class and probability
#     probability, predicted = torch.max(F.softmax(output, dim=1), dim=1)
#
#     # Generate GradCAM for the predicted class
#     activation_map = cam_extractor(predicted.item(), output)
#
#     # Convert the activation map to a heatmap
#     heatmap = to_pil_image(activation_map[0], mode='F')
#     heatmap = heatmap.resize((image.width, image.height), resample=Image.BILINEAR)
#
#     # Overlay the heatmap on the original image
#     result_img = overlay_mask(image, heatmap, alpha=0.5)
#
#     # Convert the result image to base64 for JSON response
#     buffered = io.BytesIO()
#     result_img.save(buffered, format="PNG")
#     result_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
#
#     return probability.item(), predicted.item(), result_img_base64
#############################################################################################################################################

# def predict_image(image):
#     """
#     Predict the class of an image using the loaded PyTorch model and generate Grad-CAM heatmap.
#     """
#     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#     image_tensor = image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model.eval()  # Ensure the model is in evaluation mode
#     output = model(image_tensor)
#
#     # Get the predicted class and probability
#     probability, predicted = torch.max(F.softmax(output, dim=1), dim=1)
#
#     # Generate GradCAM for the predicted class
#     activation_map = cam_extractor(predicted.item(), output)
#
#     # Since activation_map is a list, extract the first element (assuming only one map)
#     activation_map = activation_map[0]
#
#     # Convert the activation map to a NumPy array
#     activation_map = activation_map.cpu().numpy()
#
#     # Check if the activation map is 3D (C, H, W) and reduce it to 2D (H, W)
#     if activation_map.ndim == 3:
#         activation_map = np.mean(activation_map, axis=0)  # Average across the channel dimension
#     elif activation_map.ndim != 2:
#         raise ValueError("Unexpected shape for the activation map.")
#
#     # Resize the heatmap to match the original image size
#     heatmap = cv2.resize(activation_map, (image.width, image.height))
#
#     # Normalize the heatmap between 0 and 1
#     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
#
#     # Convert the heatmap to an 8-bit image (0-255)
#     heatmap = np.uint8(255 * heatmap)
#
#     # Apply the color map (convert single-channel to 3-channel using the colormap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#     # Threshold to keep only significant regions
#     threshold = 125  # Higher threshold to focus on the most important regions
#     mask = heatmap[:, :, 2] > threshold
#
#     # Convert to PIL image
#     heatmap = Image.fromarray(heatmap)
#     heatmap = heatmap.convert("RGBA")
#
#     # Convert original image to RGBA for alpha compositing
#     original_img = image.convert("RGBA")
#
#     # Invert mask to highlight important regions instead of the less important ones
#     inverted_mask = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
#
#     # Apply heatmap only to important regions, leaving the rest of the original image intact
#     overlay = Image.composite(heatmap, original_img, inverted_mask)
#
#     # Convert the result image to base64 for JSON response
#     buffered = io.BytesIO()
#     overlay.save(buffered, format="PNG")
#     result_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
#
#     return probability.item(), predicted.item(), result_img_base64
