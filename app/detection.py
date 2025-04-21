from ultralytics import YOLO

model = YOLO('models/yolov11n-face.pt' )  

def detect_faces(frame):
    results = model(frame, conf=0.7)[0]
    faces = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        faces.append((x1, y1, x2, y2))
    return faces