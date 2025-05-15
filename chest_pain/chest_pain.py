import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class ChestPainDetector:
    def __init__(self, history_length=15):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.history_length = history_length
        self.nose_x_history = deque(maxlen=history_length)

        # Thresholds
        self.chest_proximity_threshold = 0.15
        self.head_movement_threshold = 0.02
        self.eye_squint_threshold = 0.04  # normalized by interocular distance
        self.brow_tension_threshold = 0.08
        self.mouth_width_threshold = 0.35
        self.chest_pain_confidence_threshold = 0.35

    def _calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def _calculate_midpoint(self, point1, point2):
        class Point:
            pass
        midpoint = Point()
        midpoint.x = (point1.x + point2.x) / 2
        midpoint.y = (point1.y + point2.y) / 2
        midpoint.z = (point1.z + point2.z) / 2 if hasattr(point1, 'z') and hasattr(point2, 'z') else 0
        return midpoint

    def _is_wrist_on_chest(self, landmarks):
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_mid = self._calculate_midpoint(left_shoulder, right_shoulder)
        hip_mid = self._calculate_midpoint(left_hip, right_hip)
        chest_center = self._calculate_midpoint(shoulder_mid, hip_mid)

        left_dist = self._calculate_distance(left_wrist, chest_center)
        right_dist = self._calculate_distance(right_wrist, chest_center)
        min_dist = min(left_dist, right_dist)

        score = 1.0 - min(1.0, min_dist / self.chest_proximity_threshold)
        return score, min_dist

    def _calculate_head_movement(self):
        if len(self.nose_x_history) < 5:
            return 0.0
        x_coords = list(self.nose_x_history)[-5:]
        x_std = np.std(x_coords)
        return min(1.0, x_std / self.head_movement_threshold)

    def _calculate_facial_discomfort(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(frame_rgb)
        if not face_results.multi_face_landmarks:
            return 0.0

        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Normalize using interocular distance
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            interocular_distance = self._calculate_distance(left_eye_outer, right_eye_outer)

            if interocular_distance == 0:
                return 0.0  # Avoid division by zero

            # Eye squint score
            left_eye_dist = self._calculate_distance(landmarks[159], landmarks[145])
            right_eye_dist = self._calculate_distance(landmarks[386], landmarks[374])
            avg_eye_dist = (left_eye_dist + right_eye_dist) / 2
            normalized_eye = avg_eye_dist / interocular_distance
            eye_squint_score = 1.0 - min(1.0, normalized_eye / self.eye_squint_threshold)

            # Brow tension score
            left_brow_dist = abs(landmarks[70].y - landmarks[33].y)
            right_brow_dist = abs(landmarks[300].y - landmarks[263].y)
            avg_brow_dist = (left_brow_dist + right_brow_dist) / 2
            normalized_brow = avg_brow_dist / interocular_distance
            brow_tension_score = 1.0 - min(1.0, normalized_brow / self.brow_tension_threshold)

            # Mouth tension score
            mouth_width = abs(landmarks[61].x - landmarks[291].x)
            normalized_mouth = mouth_width / interocular_distance
            mouth_tension_score = min(1.0, normalized_mouth / self.mouth_width_threshold)

            # Combine
            discomfort_score = (eye_squint_score * 0.4 +
                                brow_tension_score * 0.3 +
                                mouth_tension_score * 0.3)
            return discomfort_score

        return 0.0

    def _calculate_chest_pain_confidence(self, wrist_score, head_score, facial_score):
        return (wrist_score * 0.4 +
                head_score * 0.3 +
                facial_score * 0.3)

    def _draw_debug_visualization(self, frame, pose_results, wrist_distance, confidence, discomfort_score):
        mp_drawing = mp.solutions.drawing_utils
        debug_frame = frame.copy()

        mp_drawing.draw_landmarks(
            debug_frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS)

        indicator_color = (0, 255, 0) if confidence > self.chest_pain_confidence_threshold else (0, 0, 255)

        cv2.putText(debug_frame, f"Wrist-Chest Distance: {wrist_distance:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Chest Pain Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
        cv2.putText(debug_frame, f"Facial Discomfort Score: {discomfort_score:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if confidence > self.chest_pain_confidence_threshold:
            cv2.putText(debug_frame, "CHEST PAIN DETECTED", (int(frame.shape[1] / 2) - 150, int(frame.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return debug_frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            if not pose_results.pose_landmarks:
                cv2.imshow('Chest Pain Detector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            landmarks = pose_results.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            self.nose_x_history.append(nose.x)

            wrist_score, wrist_distance = self._is_wrist_on_chest(landmarks)
            head_score = self._calculate_head_movement()
            facial_score = self._calculate_facial_discomfort(frame)
            confidence = self._calculate_chest_pain_confidence(wrist_score, head_score, facial_score)

            debug_frame = self._draw_debug_visualization(frame, pose_results, wrist_distance, confidence, facial_score)

            cv2.imshow('Chest Pain Detector', debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\projects\coughing_mediapipe\chestpain.mp4"  # Change to your video path
    detector = ChestPainDetector()
    detector.process_video(video_path)
