from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model YOLOv8 yang sudah dilatih
model = YOLO('content/runs1/detect/train/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    # Cek jika file gambar ada di request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Ambil file gambar dari request
    file = request.files['image']
    
    # Convert gambar ke format yang bisa dibaca OpenCV
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = np.array(img)
    
    # Lakukan deteksi objek
    results = model(img)
    
    # Inisialisasi untuk menyimpan deteksi dengan confidence tertinggi
    best_detection = None
    max_confidence = -1
    
    # Iterasi untuk setiap hasil deteksi
    for result in results:
        boxes = result.boxes.xyxy.tolist()    # Bounding box (xyxy format)
        classes = result.boxes.cls.tolist()     # Kelas objek
        confidences = result.boxes.conf.tolist()  # Confidence score
        
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf > max_confidence:
                max_confidence = conf
                best_detection = {
                    'box': box,                # Bounding box [x1, y1, x2, y2]
                    'class': int(cls),         # Kelas objek (ID)
                    'confidence': conf         # Confidence score
                }
    
    # Jika tidak ada deteksi, kembalikan list kosong
    if best_detection is None:
        return jsonify({'detections': []})
    
    return jsonify({'detections': [best_detection]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
