from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained YOLO model (path can be updated based on deployment environment)
model = YOLO('content/runs1/detect/train/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    # Check if image is provided in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image from the request
    file = request.files['image']
    
    # Convert image to OpenCV-readable format
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = np.array(img)
    
    # Perform object detection
    results = model(img)
    
    # Initialize variable to store the best detection
    best_detection = None
    max_confidence = -1
    
    # Iterate over detection results
    for result in results:
        boxes = result.boxes.xyxy.tolist()    # Bounding box (xyxy format)
        classes = result.boxes.cls.tolist()     # Object classes
        confidences = result.boxes.conf.tolist()  # Confidence scores
        
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf > max_confidence:
                max_confidence = conf
                best_detection = {
                    'box': box,                # Bounding box [x1, y1, x2, y2]
                    'class': int(cls),         # Object class (ID)
                    'confidence': conf         # Confidence score
                }
    
    # If no detections found, return an empty list
    if best_detection is None:
        return jsonify({'detections': []})
    
    return jsonify({'detections': [best_detection]})

# Remove or comment out the app.run() in production environment
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
