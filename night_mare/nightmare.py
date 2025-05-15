import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class NightmareDetector:
    def __init__(self, history_length=10):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.nose_y_history = deque(maxlen=history_length)
        self.shoulder_y_history = deque(maxlen=history_length)
        self.wrist_y_history = deque(maxlen=history_length)
        self.nightmare_threshold = 0.5  # Final decision threshold
        self.detected = False

    def _calc_std(self, coords):
        return np.std(coords) if len(coords) >= 3 else 0.0

    def _calculate_nightmare_score(self):
        head_movement = self._calc_std(self.nose_y_history)
        shoulder_movement = self._calc_std(self.shoulder_y_history)
        wrist_movement = self._calc_std(self.wrist_y_history)

        # Normalize and weight
        head_score = min(1.0, head_movement / 0.02)
        shoulder_score = min(1.0, shoulder_movement / 0.03)
        wrist_score = min(1.0, wrist_movement / 0.05)

        # Weighted average
        nightmare_score = (0.3 * head_score) + (0.4 * shoulder_score) + (0.3 * wrist_score)
        return nightmare_score

    def _draw_feedback(self, frame, score):
        color = (0, 0, 255) if score > self.nightmare_threshold else (0, 255, 0)
        text = f"Nightmare Score: {score:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if score > self.nightmare_threshold:
            cv2.putText(frame, "NIGHTMARE DETECTED", (int(frame.shape[1] / 2) - 200, int(frame.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
                left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
                right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
                right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y

                self.nose_y_history.append(nose_y)
                avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                self.shoulder_y_history.append(avg_shoulder_y)
                avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
                self.wrist_y_history.append(avg_wrist_y)

                nightmare_score = self._calculate_nightmare_score()

                self._draw_feedback(frame, nightmare_score)
                cv2.imshow("Nightmare Detector", frame)

                if nightmare_score > self.nightmare_threshold:
                    self.detected = True
                    print("Nightmare Detected!")
                    break

            else:
                cv2.imshow("Nightmare Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.detected


if __name__ == "__main__":
    video_path = r"C:\projects\coughing_mediapipe\night_mare.mp4"
    detector = NightmareDetector()
    result = detector.process_video(video_path)
    print("Nightmare detected:", result)
