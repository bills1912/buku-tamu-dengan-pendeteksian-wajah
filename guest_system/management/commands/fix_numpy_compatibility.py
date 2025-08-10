import numpy as np
from deepface import DeepFace

# Test dengan model terbaru
test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

# Test FaceNet
embedding_facenet = DeepFace.represent(
    img_path=test_image,
    model_name="Facenet",
    detector_backend="opencv",
    enforce_detection=False
)

# Test GhostFaceNet (new model!)
embedding_ghost = DeepFace.represent(
    img_path=test_image,
    model_name="GhostFaceNet",  # New in 0.0.95!
    detector_backend="opencv",
    enforce_detection=False
)

print(f"FaceNet embedding: {len(embedding_facenet[0]['embedding'])} dimensions")
print(f"GhostFaceNet embedding: {len(embedding_ghost[0]['embedding'])} dimensions")