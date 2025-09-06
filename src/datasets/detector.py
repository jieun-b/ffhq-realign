import mediapipe as mp
import numpy as np

class MediaPipe(object):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,  
            min_detection_confidence=0.5
        )

    def run(self, image):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: bbox (x1,y1,x2,y2), type
        """
        h, w, _ = image.shape
        results = self.face_mesh.process(image)

        if not results.multi_face_landmarks:
            return [0], 'bbox'

        # 첫 얼굴 랜드마크만 사용
        landmarks = results.multi_face_landmarks[0].landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

        bbox = [int(left), int(top), int(right), int(bottom)]
        return bbox, 'bbox'
