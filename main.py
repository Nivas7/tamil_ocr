from flask import Flask, request, jsonify
from ocr_tamil.ocr import OCR
from PIL import Image
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Initialize OCR model
ocr = OCR(detect=True, lang=["tamil"])  # Modify parameters as needed

@app.route('/process-image', methods=['POST'])
def process_image():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    
    try:
        # Convert the uploaded file to an image
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        # Perform OCR using the model
        text_list = ocr.predict(image_np)  # Directly pass the image array

        # Format the response
        response = {
            "text": text_list
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Hello, tamil OCR!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
