import faiss
import numpy as np
import pickle

class FaceDB:
    def __init__(self, dim=512):
        self.index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
        self.labels = []

    def _normalize(self, emb):
        emb = emb.astype('float32')
        if emb.ndim == 2:
            emb = emb[0]  # (1, 512) -> (512,)
        emb = emb / np.linalg.norm(emb)
        return emb.reshape(1, -1)  # FAISS expects 2D array: (1, 512)

    def add_face(self, embedding, label):
        embedding = self._normalize(np.array(embedding))
        self.index.add(embedding)
        self.labels.append(label)

    def search(self, embedding, top_k=1, threshold=0.4):
        # Handle empty database gracefully
        if self.index.ntotal == 0 or not self.labels:
            return []
        embedding = self._normalize(np.array(embedding))
        D, I = self.index.search(embedding, top_k)

        results = []
        for j, i in enumerate(I[0]):
            if i < len(self.labels):
                sim = D[0][j]
                print(f"[SEARCH] Match: {self.labels[i]} (similarity: {sim})")
                if sim > threshold:
                    results.append((self.labels[i], sim))
        return results

    def save(self, index_path='face.index', label_path='face_index.pkl'):
        faiss.write_index(self.index, index_path)
        with open(label_path, 'wb') as f:
            pickle.dump(self.labels, f)

    def load(self, index_path='face.index', label_path='face_index.pkl'):
        self.index = faiss.read_index(index_path)
        with open(label_path, 'rb') as f:
            self.labels = pickle.load(f)
    def reset(self, index_path='face.index', label_path='face_index.pkl'):
        # Reset the index and labels
        self.index = faiss.IndexFlatIP(512)  # create a new empty index
        self.labels = []  # clear the labels list

        # Optionally, save the reset state if you want to persist this reset
        self.save(index_path, label_path)
        print("Database has been reset.")
