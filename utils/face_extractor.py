import cv2
import numpy as np

class FaceExtractor:
    def __init__(self, *args, **kwargs):
        try:
            # We bypass the 'solutions' wrapper and go straight to the implementation
            from mediapipe.python.solutions import face_mesh
            self.face_mesh = face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True
            )
            print("✓ FaceExtractor (Surgical Implementation) Ready")
        except Exception as e:
            try:
                import mediapipe as mp
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
                print("✓ FaceExtractor (Standard Fallback) Ready")
            except:
                print(f"✗ FaceExtractor Failed: {e}")
                self.face_mesh = None

    def extract(self, image):
        if not self.face_mesh: return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        return res.multi_face_landmarks if res.multi_face_landmarks else None
