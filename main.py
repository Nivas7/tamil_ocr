from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
import os

# Use your actual class list here!
classes = [
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
    'க', 'ங', 'ச', 'ஜ', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய',
    'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ஷ', 'ஸ', 'ஹ', 'க்ஷ', 'ஃ'
    # Add/remove to match your specific model
]

app = Flask(__name__)
CORS(app)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, len(classes))  # Output matches your class count

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool1(F.relu(self.bn6(self.conv6(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x

MODEL_PATH = os.environ.get("MODEL_PATH", "tamil_net.pt")
net = Net()
net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
net.eval()

def preprocess_image_from_file(file_stream):
    img = Image.open(file_stream)
    converted = img.convert("LA")
    la = np.array(converted)
    la[la[..., -1] == 0] = [255, 255]
    whiteBG = Image.fromarray(la)
    converted = whiteBG.convert("L")
    inverted = ImageOps.invert(converted)
    bounding_box = inverted.getbbox()
    if bounding_box is None:
        bounding_box = (0, 0, inverted.size[0], inverted.size[1])
    padded_box = (
        max(bounding_box[0] - 5, 0),
        max(bounding_box[1] - 5, 0),
        min(bounding_box[2] + 5, inverted.size[0]),
        min(bounding_box[3] + 5, inverted.size[1])
    )
    cropped = inverted.crop(padded_box)
    thick = cropped.filter(ImageFilter.MaxFilter(5))
    ratio = 48.0 / max(thick.size)
    new_size = tuple([int(round(x * ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)
    arr = np.asarray(res)
    com = ndimage.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
    result.paste(res, box)
    return result

def transform_img(img):
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return my_transforms(img).unsqueeze(0)

def get_prediction_from_image(file_stream, net):
    img = preprocess_image_from_file(file_stream)
    transformed = transform_img(img)
    with torch.no_grad():
        output = net(transformed)
        prob, predicted = torch.max(output.data, 1)
        confidence = float(prob.item()) * 100
        # Defensive: if prediction is outside label range, return "Unknown"
        if predicted < len(classes):
            return classes[predicted], int(confidence)
        else:
            return "Unknown", int(confidence)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "TamilNet backend is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        character, confidence = get_prediction_from_image(file.stream, net)
        return jsonify({"character": character, "confidence": confidence}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/suggest', methods=['GET'])
def suggest():
    suggestion = random.choice(classes)
    return jsonify({"suggestion": suggestion}), 200

#if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 10000))
#    app.run(host='0.0.0.0', port=port)
