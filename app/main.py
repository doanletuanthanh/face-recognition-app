from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io

from app.detection import detect_faces
from app.recognition import get_embedding
from app.face_db import FaceDB

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. For production, specify your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = FaceDB()


try:
    db.reset()
    db.load()
except:
    pass

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    detections = detect_faces(frame)
    results = []

    for (x1, y1, x2, y2) in detections:
        face_crop = frame[y1:y2, x1:x2]
        embedding = get_embedding(face_crop)
        
        if embedding is not None:
            # Normalize the embedding to match the database shape
            embedding = embedding.flatten()  # Ensure it has shape (512,)
            matches = db.search(embedding, threshold=0.35)
            for m in matches:
                print(f"[PROCESS_FRAME] Match: {m[0]} with similarity {m[1]}")
            name = matches[0][0] if matches else "Unknown"
            results.append({"name": name, "bbox": [x1, y1, x2, y2]})

            # Draw the bounding box and name on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Encode the frame and send it back
    _, encoded_img = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")


@app.post("/add_face")
async def add_face(name: str = Form(...), file: UploadFile = File(...)):
    print(f"[DEBUG] Received name: {name}")
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    detections = detect_faces(frame)
    if not detections:
        return JSONResponse(content={"error": "No face found"}, status_code=400)

    x1, y1, x2, y2 = detections[0]
    face_crop = frame[y1:y2, x1:x2]
    embedding = get_embedding(face_crop)
    if embedding is None:
        return JSONResponse(content={"error": "Embedding failed"}, status_code=400)

    # Flatten and normalize the embedding before adding to the database
    embedding = embedding.flatten()  # Ensure it's a 1D array (512,)
    db.add_face(embedding, name)
    db.save()
    
    print(f"[ADD_FACE] Successfully added: {name}")
    print(f"[DB] Current total faces: {db.index.ntotal}")
    print(f"[DB] All faces in DB:")
    for i, label in enumerate(db.labels):
        print(f"  {i+1}. {label}")

    return {"status": "face added", "name": name}


@app.post("/process_stream")
async def process_stream(request: Request):
    """
    Accepts a stream of images (multipart/x-mixed-replace or repeated POSTs) from the client,
    processes each frame, and returns the processed frame as a stream.
    """
    async def frame_generator():
        async for chunk in request.stream():
            # Assume each chunk is a complete JPEG image
            nparr = np.frombuffer(chunk, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Detect faces
            detections = detect_faces(frame)
            for (x1, y1, x2, y2) in detections:
                face_crop = frame[y1:y2, x1:x2]
                embedding = get_embedding(face_crop)
                if embedding is not None:
                    embedding = embedding.flatten()
                    matches = db.search(embedding, threshold=0.35)
                    name = matches[0][0] if matches else "Unknown"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            _, encoded_img = cv2.imencode('.jpg', frame)
            frame_bytes = encoded_img.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')
