
from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import random
 
# Use all 156 Tamil character labels, in the same order as the model was trained.
classes = ['அ', 'ஆ', 'ஓ', 'ஙூ', 'சூ', 'ஞூ', 'டூ', 'ணூ', 'தூ', 'நூ', 'பூ', 'மூ', 'யூ', 'ஃ', 'ரூ', 'லூ', 'வூ', 'ழூ', 'ளூ', 'றூ', 'னூ', 'ா', 'ெ', 'ே', 'க', 'ை', 'ஸ்ரீ', 'ஸு', 'ஷு', 'ஜு', 'ஹு', 'க்ஷு', 'ஸூ', 'ஷூ', 'ஜூ', 'ங', 'ஹூ', 'க்ஷூ', 'க்', 'ங்', 'ச்', 'ஞ்', 'ட்', 'ண்', 'த்', 'ந்', 'ச', 'ப்', 'ம்', 'ய்', 'ர்', 'ல்', 'வ்', 'ழ்', 'ள்', 'ற்', 'ன்', 'ஞ', 'ஸ்', 'ஷ்', 'ஜ்', 'ஹ்', 'க்ஷ்', 'ஔ', 'ட', 'ண', 'த', 'ந', 'இ', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ஈ', 'ஸ', 'ஷ', 'ஜ', 'ஹ', 'க்ஷ', 'கி', 'ஙி', 'சி', 'ஞி', 'டி', 'உ', 'ணி', 'தி', 'நி', 'பி', 'மி', 'யி', 'ரி', 'லி', 'வி', 'ழி', 'ஊ', 'ளி', 'றி', 'னி', 'ஸி', 'ஷி', 'ஜி', 'ஹி', 'க்ஷி', 'கீ', 'ஙீ', 'எ', 'சீ', 'ஞீ', 'டீ', 'ணீ', 'தீ', 'நீ', 'பீ', 'மீ', 'யீ', 'ரீ', 'ஏ', 'லீ', 'வீ', 'ழீ', 'ளீ', 'றீ', 'னீ', 'ஸீ', 'ஷீ', 'ஜீ', 'ஹீ', 'ஐ', 'க்ஷீ', 'கு', 'ஙு', 'சு', 'ஞு', 'டு', 'ணு', 'து', 'நு', 'பு', 'ஒ', 'மு', 'யு', 'ரு', 'லு', 'வு', 'ழு', 'ளு', 'று', 'னு', 'கூ']

app = Flask(__name__)
 
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
        self.fc3 = nn.Linear(512, 156)  # 156 outputs for all Tamil characters
 
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
 
net = Net()
net.load_state_dict(torch.load("D:/TamilNet-master (1)/TamilNet-master/app/tamil_net.pt", map_location=torch.device('cpu')))
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
    padded_box = tuple(map(lambda i, j: max(i + j, 0), bounding_box, (-5, -5, 5, 5)))
    cropped = inverted.crop(padded_box)
 
    thick = cropped.filter(ImageFilter.MaxFilter(5))
 
    ratio = 48.0 / max(thick.size)
    new_size = tuple([int(round(x * ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)
 
    arr = np.asarray(res)
    com = ndimage.measurements.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
    result.paste(res, box)
    return result
 
def transform_img(img):
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return my_transforms(img).unsqueeze(0)
 
def get_prediction_from_image(file_stream, net):
    img = preprocess_image_from_file(file_stream)
    transformed = transform_img(img)
    output = net(transformed)
    prob, predicted = torch.max(output.data, 1)
    confidence = int(round(prob.item() * 100))
    return classes[predicted], confidence
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    character, confidence = get_prediction_from_image(file.stream, net)
    return f"{character} {confidence}"
 
@app.route('/suggest', methods=['GET'])
def suggest():
    suggestion = random.choice(classes)
    return suggestion
 
# Optional: Uncomment for local debugging
if __name__ == '__main__':

        app.run(debug=True)


