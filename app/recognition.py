import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

handler = insightface.model_zoo.get_model('models/arcface_r18.onnx')
print(handler)
handler.prepare(ctx_id=0)

def get_embedding(face_img):
    # Use ArcFace directly on the cropped face (from YOLO)
    try:
        embedding = handler.get_feat(face_img)
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None