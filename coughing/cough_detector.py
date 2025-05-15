import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class CoughDetector:
    def __init__(self, history_length=15):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.hand_face_distance_history = deque(maxlen=history_length)
        self.head_movement_history = deque(maxlen=history_length)
        self.right_wrist_history = deque(maxlen=history_length)
        self.left_wrist_history = deque(maxlen=history_length)
        self.nose_y_history = deque(maxlen=history_length)
        self.nose_x_history = deque(maxlen=history_length)
        self.hand_face_threshold = 0.15
        self.head_movement_threshold = 0.02
        self.cough_confidence_threshold = 0.23  # Lowered from 0.4 to 0.25
        self.head_tilt_threshold = 0.015
        self.palm_folded_threshold = 0.1
        self.palm_to_head_threshold = 0.15
        self.possible_cough_in_progress = False
        self.frames_since_hand_raised = 0
        self.max_frames_for_cough = 20

    def detect_cough(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return False, frame, 0.0
        landmarks = results.pose_landmarks.landmark
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]

        right_hand_distance = self._calculate_distance(right_wrist, nose)
        left_hand_distance = self._calculate_distance(left_wrist, nose)
        min_hand_distance = min(right_hand_distance, left_hand_distance)

        head_position = self._calculate_midpoint(right_ear, left_ear)

        self.hand_face_distance_history.append(min_hand_distance)
        self.head_movement_history.append((head_position.x, head_position.y))
        self.right_wrist_history.append((right_wrist.x, right_wrist.y))
        self.left_wrist_history.append((left_wrist.x, left_wrist.y))
        self.nose_y_history.append(nose.y)
        self.nose_x_history.append(nose.x)

        if len(self.hand_face_distance_history) < 5 or len(self.head_movement_history) < 5:
            return False, self._draw_debug_visualization(frame, results, min_hand_distance), 0.0

        cough_confidence = self._calculate_cough_confidence()
        debug_frame = self._draw_debug_visualization(frame, results, min_hand_distance, cough_confidence)
        is_coughing = cough_confidence > self.cough_confidence_threshold
        return is_coughing, debug_frame, cough_confidence

    def _calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def _calculate_midpoint(self, point1, point2):
        class Point:
            pass
        midpoint = Point()
        midpoint.x = (point1.x + point2.x) / 2
        midpoint.y = (point1.y + point2.y) / 2
        midpoint.z = (point1.z + point2.z) / 2
        return midpoint

    def _calculate_head_movement(self):
        if len(self.head_movement_history) < 5:
            return 0.0
        x_coords = [pos[0] for pos in self.head_movement_history]
        y_coords = [pos[1] for pos in self.head_movement_history]
        x_std = np.std(x_coords[-5:])
        y_std = np.std(y_coords[-5:])
        return max(x_std, y_std)

    def _calculate_hand_approach(self):
        if len(self.hand_face_distance_history) < 5:
            return 0.0
        recent_min = min(list(self.hand_face_distance_history)[-5:])
        proximity_score = 1.0 - min(1.0, recent_min / self.hand_face_threshold)
        return proximity_score

    def _calculate_head_tilt(self):
        if len(self.nose_y_history) < 5:
            return 0.0
        nose_y_positions = list(self.nose_y_history)[-5:]
        tilt = nose_y_positions[-1] - nose_y_positions[0]
        tilt_score = max(0.0, min(1.0, tilt / self.head_tilt_threshold))
        return tilt_score

    def _calculate_palm_folded(self):
        if len(self.right_wrist_history) < 1 or len(self.left_wrist_history) < 1:
            return 0.0, False
        right_wrist = list(self.right_wrist_history)[-1]
        left_wrist = list(self.left_wrist_history)[-1]
        wrist_distance = math.sqrt(
            (right_wrist[0] - left_wrist[0])**2 +
            (right_wrist[1] - left_wrist[1])**2
        )
        palm_folded_score = 1.0 - min(1.0, wrist_distance / self.palm_folded_threshold)
        is_folded = palm_folded_score > 0.5
        return palm_folded_score, is_folded

    def _calculate_palm_to_head(self):
        if len(self.right_wrist_history) < 1 or len(self.left_wrist_history) < 1 or len(self.nose_y_history) < 1 or len(self.nose_x_history) < 1:
            return 0.0
        right_wrist = list(self.right_wrist_history)[-1]
        left_wrist = list(self.left_wrist_history)[-1]
        nose_x = list(self.nose_x_history)[-1]
        nose_y = list(self.nose_y_history)[-1]
        palm_midpoint_x = (right_wrist[0] + left_wrist[0]) / 2
        palm_midpoint_y = (right_wrist[1] + left_wrist[1]) / 2
        palm_to_head_distance = math.sqrt(
            (palm_midpoint_x - nose_x)**2 +
            (palm_midpoint_y - nose_y)**2
        )
        palm_to_head_score = 1.0 - min(1.0, palm_to_head_distance / self.palm_to_head_threshold)
        return palm_to_head_score

    def _calculate_cough_confidence(self):
        hand_approach = self._calculate_hand_approach()
        head_movement = self._calculate_head_movement()
        head_tilt = self._calculate_head_tilt()
        palm_folded_score, is_folded = self._calculate_palm_folded()
        palm_to_head_score = self._calculate_palm_to_head() if is_folded else 0.0

        # Adjusted weights (removed wrist_velocity, redistributed weights)
        hand_approach_weight = 0.3
        head_movement_weight = 0.15
        head_tilt_weight = 0.2
        palm_folded_weight = 0.2
        palm_to_head_weight = 0.15

        # Boost head tilt contribution when palms are folded and near head
        if is_folded and palm_to_head_score > 0.5:
            head_tilt_weight = 0.3
            hand_approach_weight = 0.25
            head_movement_weight = 0.1
            palm_folded_weight = 0.2
            palm_to_head_weight = 0.15

        confidence = (
            (hand_approach * hand_approach_weight) +
            (min(1.0, head_movement / self.head_movement_threshold) * head_movement_weight) +
            (head_tilt * head_tilt_weight) +
            (palm_folded_score * palm_folded_weight) +
            (palm_to_head_score * palm_to_head_weight)
        )
        return confidence

    def _draw_debug_visualization(self, frame, pose_results, min_hand_distance, cough_confidence=0.0):
        mp_drawing = mp.solutions.drawing_utils
        debug_frame = frame.copy()
        mp_drawing.draw_landmarks(
            debug_frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS)
        indicator_color = (0, 255, 0) if cough_confidence > self.cough_confidence_threshold else (0, 0, 255)
        cv2.putText(
            debug_frame,
            f"Hand-Face Distance: {min_hand_distance:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            debug_frame,
            f"Cough Confidence: {cough_confidence:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            indicator_color,
            2
        )
        if cough_confidence > self.cough_confidence_threshold:
            cv2.putText(
                debug_frame,
                "COUGH DETECTED",
                (int(frame.shape[1] / 2) - 150, int(frame.shape[0] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
        return debug_frame