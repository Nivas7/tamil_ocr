from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import random

# Define Flask app
app = Flask(__name__)

# Define character classes (replace with your full list)
classes = ['அ', 'ஆ', 'ஓ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ', 'ஃ', 'ரூ', 'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ']

# Define the CNN model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 8 * 8, len(classes))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.softmax(self.fc1(x), dim=1)
        return x

# Load the model
net = Net()
net.load_state_dict(torch.load('tamil_net.pt', map_location=torch.device('cpu')))
net.eval()

# Preprocessing function
def preprocess_image(image_data):
    # Convert base64 image to PIL Image
    image = Image.open(image_data).convert('L')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    return image

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']  # Get uploaded image file
        image = preprocess_image(file)
        output = net(image)
        _, predicted = torch.max(output, 1)
        prediction = classes[predicted.item()]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Suggest endpoint
@app.route('/suggest', methods=['GET'])
def suggest():
    suggestion = random.choice(classes)
    return jsonify({'suggestion': suggestion})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
