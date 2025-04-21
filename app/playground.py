from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from recognition import get_embedding
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO('models\yolov11s-face.pt') 
model.to(device)  # Move the model to the specified device 
  # Should print: True

# Path to the image file you want to detect faces in
image_path = 'test-image\IMG_20201105_012427_557.jpg'

# Open the image with PIL and convert to RGB
image = Image.open(image_path).convert("RGB")

# Convert PIL image to OpenCV format (numpy array with BGR color)
frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

results = model(frame, conf=0.7)[0]
faces = []
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    faces.append((x1, y1, x2, y2))


# Show image with boxes
print(f"Detected {len(faces)} face(s):")

embeddings = []
for x1, y1, x2, y2 in faces:
    face_crop = frame[y1:y2, x1:x2] 
    face_crop = cv2.resize(face_crop, (112, 112))  # Resize to 112x112 for ArcFace
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)  # Convert to RGB for ArcFace
    embedding = get_embedding(face_crop)  # Get the embedding using ArcFace
    if embedding is not None:
        embeddings.append(embedding)
        print(f"Face bounding box: ({x1}, {y1}, {x2}, {y2})")
        print(f"Embedding shape: {embedding.shape}")
    plt.figure()
    plt.imshow(face_crop)
    plt.title("Cropped Face")
    plt.axis('off')
    plt.show()
   

