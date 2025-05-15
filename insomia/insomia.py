import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class InsomniaDetector:
    def __init__(self, history_length=20):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.nose_y_history = deque(maxlen=history_length)
        self.shoulder_y_history = deque(maxlen=history_length)
        self.wrist_head_proximity = deque(maxlen=history_length)
        self.insomnia_threshold = 0.5
        self.detected = False

    def _calc_std(self, coords):
        return np.std(coords) if len(coords) >= 3 else 0.0

    def _calculate_insomnia_score(self):
        head_std = self._calc_std(self.nose_y_history)
        shoulder_std = self._calc_std(self.shoulder_y_history)
        hand_to_head_freq = np.mean(self.wrist_head_proximity) if self.wrist_head_proximity else 0.0

        head_score = min(1.0, head_std / 0.01)
        shoulder_score = min(1.0, shoulder_std / 0.015)
        proximity_score = min(1.0, hand_to_head_freq)

        insomnia_score = (0.4 * head_score) + (0.3 * shoulder_score) + (0.3 * proximity_score)
        return insomnia_score

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _draw_feedback(self, frame, score):
        color = (0, 0, 255) if score > self.insomnia_threshold else (0, 255, 0)
        cv2.putText(frame, f"Insomnia Score: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if score > self.insomnia_threshold:
            cv2.putText(frame, "INSOMNIA SIGNS DETECTED", (int(frame.shape[1] / 2) - 200, int(frame.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

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
                lm = results.pose_landmarks.landmark
                nose = lm[self.mp_pose.PoseLandmark.NOSE]
                left_shoulder = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                self.nose_y_history.append(nose.y)
                avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                self.shoulder_y_history.append(avg_shoulder_y)

                # Check how close wrists are to head
                left_dist = self._calculate_distance(left_wrist, nose)
                right_dist = self._calculate_distance(right_wrist, nose)
                is_hand_near_head = 1.0 if min(left_dist, right_dist) < 0.1 else 0.0
                self.wrist_head_proximity.append(is_hand_near_head)

                insomnia_score = self._calculate_insomnia_score()
                self._draw_feedback(frame, insomnia_score)
                cv2.imshow("Insomnia Detector", frame)

                if insomnia_score > self.insomnia_threshold:
                    self.detected = True
                    print("Insomnia Symptoms Detected.")
                    break

            else:
                cv2.imshow("Insomnia Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return self.detected


if __name__ == "__main__":
    video_path = r"C:\projects\coughing_mediapipe\insomia.mp4"  # Replace with your actual path
    detector = InsomniaDetector()
    result = detector.process_video(video_path)
    print("Insomnia symptoms detected:", result)
