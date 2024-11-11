from flask import Blueprint, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import logging
import timm
prediction_bp = Blueprint('prediction_bp', __name__)
import torch.nn.functional as F
import cv2
import numpy as np
from grad_cam import GradCAM

MODEL_PATH = 'models/timm-convnext-v2/timm-convnext-v2.pth'
model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False, num_classes=3)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
print(model)
model.eval()
logging.basicConfig(level=logging.DEBUG)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image.
    """
    image_data = base64.b64decode(base64_string.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

def predict_image(image):
    """
    Predict the class of an image using the loaded PyTorch model and return Grad-CAM visualization.
    """
    image_tensor = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Grad-CAM
    target_layer = model.stages[-1].blocks[-2].conv_dw
    grad_cam = GradCAM(model, target_layer)

    # Get model prediction and Grad-CAM
    with torch.no_grad():
        output = model(image_tensor)
        probability, predicted = torch.max(F.softmax(output, dim=1), dim=1)

    cam = grad_cam(image_tensor, predicted.item())

    # Convert cam to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap + np.float32(image.resize((image_tensor.shape[2], image_tensor.shape[3]))) / 255
    heatmap = heatmap / np.max(heatmap)

    # Convert heatmap to PIL image
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)

    # Save Grad-CAM overlay as base64 string
    buffer = io.BytesIO()
    heatmap.save(buffer, format="JPEG")
    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return probability.item(), predicted.item(), gradcam_base64

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    results = []

    # Predictions and saving csv_data for CSV
    for item in json_data:
        image_base64 = item['content']
        image_name = item['name']
        # Convert base64 to PIL image
        image = base64_to_image(image_base64)
        # Predict using the model
        probability, prediction, heatmap_image  = predict_image(image)

        print(f"name: {image_name}, prediction: {str(prediction)}, probability: {str(probability)}")
        results.append({
            'name': image_name,
            'prediction': prediction,
            'probability': probability,
            'content': image_base64,
            'grad_cam': heatmap_image
        })
    return jsonify(results)