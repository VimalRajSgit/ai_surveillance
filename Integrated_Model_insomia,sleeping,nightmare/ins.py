import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import time
import os

class BaseDetector:
    """Base class for all symptom detectors"""
    def __init__(self):
        self.detected = False
        self.detection_score = 0.0
        self.detection_threshold = 0.5

    def process_frame(self, frame, display=True):
        """Process a single frame and return the detection result"""
        pass

    def reset(self):
        """Reset the detector state"""
        self.detected = False
        self.detection_score = 0.0

class NightmareDetector(BaseDetector):
    def __init__(self, history_length=10):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.nose_y_history = deque(maxlen=history_length)
        self.shoulder_y_history = deque(maxlen=history_length)
        self.wrist_y_history = deque(maxlen=history_length)
        self.detection_threshold = 0.5

    def _calc_std(self, coords):
        return np.std(coords) if len(coords) >= 3 else 0.0

    def _calculate_nightmare_score(self):
        head_movement = self._calc_std(self.nose_y_history)
        shoulder_movement = self._calc_std(self.shoulder_y_history)
        wrist_movement = self._calc_std(self.wrist_y_history)

        head_score = min(1.0, head_movement / 0.02)
        shoulder_score = min(1.0, shoulder_movement / 0.03)
        wrist_score = min(1.0, wrist_movement / 0.05)

        nightmare_score = (0.3 * head_score) + (0.4 * shoulder_score) + (0.3 * wrist_score)
        self.detection_score = nightmare_score
        return nightmare_score

    def process_frame(self, frame, display=True):
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
            self.detected = nightmare_score > self.detection_threshold

            if display:
                color = (0, 0, 255) if self.detected else (0, 255, 0)
                cv2.putText(frame, f"Nightmare Score: {nightmare_score:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return self.detected

class SleepDetector(BaseDetector):
    def __init__(self, eye_threshold=0.25, movement_threshold=2.0, history_length=30):
        super().__init__()
        self.eye_threshold = eye_threshold
        self.movement_threshold = movement_threshold
        self.previous_position = None
        self.detection_threshold = 0.5

        self.sleep_start_time = None
        self.total_sleep_time = 0
        self.sleeping_frames = 0
        self.total_frames = 0
        self.sleep_history = deque(maxlen=history_length)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        coords = [(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices]
        hor_dist = distance(coords[0], coords[3])
        ver_dist1 = distance(coords[1], coords[5])
        ver_dist2 = distance(coords[2], coords[4])
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
        return ear

    def is_sleeping(self, landmarks):
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        left_ear = self.get_eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = self.get_eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        return avg_ear < self.eye_threshold

    def detect_movement(self, landmarks):
        nose_tip = landmarks[1]
        current_pos = (nose_tip.x, nose_tip.y)

        if self.previous_position is None:
            self.previous_position = current_pos
            return False

        movement = math.hypot(current_pos[0] - self.previous_position[0],
                              current_pos[1] - self.previous_position[1])

        self.previous_position = current_pos
        return movement > self.movement_threshold

    def process_frame(self, frame, display=True):
        self.total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        sleep_score = 0.0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            sleeping = self.is_sleeping(landmarks)
            moving = self.detect_movement(landmarks)

            if sleeping and not moving:
                sleep_score = 0.9
                self.sleep_history.append(1)
                self.sleeping_frames += 1

                if self.sleep_start_time is None:
                    self.sleep_start_time = time.time()
            else:
                sleep_score = 0.1
                self.sleep_history.append(0)

                if self.sleep_start_time is not None:
                    self.total_sleep_time += time.time() - self.sleep_start_time
                    self.sleep_start_time = None

        if len(self.sleep_history) > 5:
            recent_sleep_ratio = sum(self.sleep_history) / len(self.sleep_history)
            self.detection_score = recent_sleep_ratio
            self.detected = recent_sleep_ratio > self.detection_threshold
        else:
            self.detection_score = sleep_score
            self.detected = sleep_score > self.detection_threshold

        if display:
            color = (0, 0, 255) if self.detected else (0, 255, 0)
            status_text = "Sleeping" if self.detected else "Awake"
            cv2.putText(frame, f"Status: {status_text}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Sleep Score: {self.detection_score:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            sleep_seconds = self.total_sleep_time
            if self.sleep_start_time is not None:
                sleep_seconds += time.time() - self.sleep_start_time

            minutes = int(sleep_seconds // 60)
            seconds = int(sleep_seconds % 60)
            cv2.putText(frame, f"Sleep Period: {minutes}m {seconds}s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        return self.detected

    def get_sleep_stats(self):
        sleep_seconds = self.total_sleep_time
        if self.sleep_start_time is not None:
            sleep_seconds += time.time() - self.sleep_start_time

        sleep_percentage = 0
        if self.total_frames > 0:
            sleep_percentage = (self.sleeping_frames / self.total_frames) * 100

        return {
            "total_sleep_time": sleep_seconds,
            "sleep_percentage": sleep_percentage
        }

class InsomniaDetector(BaseDetector):
    def __init__(self, history_length=30):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.head_rotation_history = deque(maxlen=history_length)
        self.head_x_positions = deque(maxlen=history_length)
        self.restlessness_scores = deque(maxlen=history_length)
        self.frustration_scores = deque(maxlen=history_length)
        self.detection_threshold = 0.52
        self.turn_count = 0
        self.last_turn_direction = None
        self.movement_threshold = 0.02

    def _calculate_head_rotation(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE_OUTER]
        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER]

        eye_distance = abs(right_eye.x - left_eye.x)
        return eye_distance

    def _detect_head_turning(self):
        if len(self.head_x_positions) < 5:
            return 0.0

        derivatives = np.diff([p for p in self.head_x_positions])
        direction_changes = 0
        for i in range(1, len(derivatives)):
            if (derivatives[i] > self.movement_threshold and derivatives[i - 1] < -self.movement_threshold) or \
                    (derivatives[i] < -self.movement_threshold and derivatives[i - 1] > self.movement_threshold):
                direction_changes += 1

        max_possible_changes = len(derivatives) - 1
        turn_score = min(1.0, direction_changes / (max_possible_changes * 0.5))
        return turn_score

    def _detect_facial_frustration(self, face_landmarks):
        if not face_landmarks:
            return 0.0

        landmarks = face_landmarks.landmark
        left_eyebrow = landmarks[65]
        right_eyebrow = landmarks[295]
        forehead = landmarks[10]

        left_brow_height = forehead.y - left_eyebrow.y
        right_brow_height = forehead.y - right_eyebrow.y
        brow_height = (left_brow_height + right_brow_height) / 2
        brow_score = 1.0 - min(1.0, brow_height * 10)

        upper_lip = landmarks[0]
        lower_lip = landmarks[17]
        mouth_openness = abs(upper_lip.y - lower_lip.y)
        mouth_score = 1.0 - min(1.0, mouth_openness * 15)

        frustration_score = (0.7 * brow_score) + (0.3 * mouth_score)
        return frustration_score

    def _calculate_insomnia_score(self):
        turn_score = self._detect_head_turning()
        avg_frustration = np.mean(self.frustration_scores) if self.frustration_scores else 0.0
        avg_restlessness = np.mean(self.restlessness_scores) if self.restlessness_scores else 0.0

        if len(self.head_rotation_history) > 3:
            rotation_std = np.std(self.head_rotation_history)
            rotation_score = min(1.0, rotation_std / 0.05)
        else:
            rotation_score = 0.0

        insomnia_score = (0.4 * turn_score) + (0.3 * avg_frustration) + (0.2 * avg_restlessness) + (0.1 * rotation_score)
        self.detection_score = insomnia_score
        return insomnia_score

    def process_frame(self, frame, display=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        current_frustration = 0.0
        current_turn_score = 0.0

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            nose = lm[self.mp_pose.PoseLandmark.NOSE]
            self.head_x_positions.append(nose.x)

            head_rotation = self._calculate_head_rotation(lm)
            self.head_rotation_history.append(head_rotation)

            if len(self.head_x_positions) > 3:
                recent_movement = np.std(list(self.head_x_positions)[-5:])
                restlessness = min(1.0, recent_movement / 0.02)
                self.restlessness_scores.append(restlessness)

            current_turn_score = self._detect_head_turning()

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            current_frustration = self._detect_facial_frustration(face_landmarks)
            self.frustration_scores.append(current_frustration)

        insomnia_score = self._calculate_insomnia_score()
        self.detected = insomnia_score > self.detection_threshold

        if display:
            color = (0, 0, 255) if self.detected else (0, 255, 0)
            cv2.putText(frame, f"Insomnia Score: {insomnia_score:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Head Turning: {current_turn_score:.2f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
            cv2.putText(frame, f"Frustration: {current_frustration:.2f}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)

        return self.detected

class SleepSymptomsDetector:
    """Main class that combines all detectors"""
    def __init__(self):
        self.nightmare_detector = NightmareDetector()
        self.sleep_detector = SleepDetector()
        self.insomnia_detector = InsomniaDetector()
        self.current_condition = "None"
        self.condition_confidence = 0.0
        self.condition_history = []

    def process_video(self, video_path, output_path=None):
        """Process a video file and detect sleep symptoms, optionally saving the output"""
        # Verify the video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return False

        # Check read permissions
        if not os.access(video_path, os.R_OK):
            print(f"Error: No read permission for video file {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}. Check if the file is a valid video format or not corrupted.")
            return False

        # Get video properties
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Warning: Could not retrieve FPS from video. Using default FPS of 30.")
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize VideoWriter if output_path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"Error: Cannot open video writer for output path {output_path}")
                cap.release()
                return False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every other frame to improve performance
            if frame_count % 2 != 0:
                continue

            # Create a copy of the frame for display
            display_frame = frame.copy()

            # Process frame with all detectors
            nightmare_detected = self.nightmare_detector.process_frame(frame, display=False)
            sleep_detected = self.sleep_detector.process_frame(frame, display=False)
            insomnia_detected = self.insomnia_detector.process_frame(frame, display=False)

            # Determine the current condition based on detector scores
            conditions = {
                "Nightmare": (nightmare_detected, self.nightmare_detector.detection_score),
                "Sleeping": (sleep_detected, self.sleep_detector.detection_score),
                "Insomnia": (insomnia_detected, self.insomnia_detector.detection_score),
            }

            # Find the condition with the highest score that is above its threshold
            highest_score = 0
            detected_condition = "None"

            for condition, (detected, score) in conditions.items():
                if detected and score > highest_score:
                    highest_score = score
                    detected_condition = condition

            self.current_condition = detected_condition
            self.condition_confidence = highest_score

            # Add to history
            if frame_count % 10 == 0:
                timestamp = frame_count / fps
                self.condition_history.append({
                    "time": timestamp,
                    "condition": self.current_condition,
                    "confidence": self.condition_confidence
                })

            # Display the results on the frame
            self._draw_results(display_frame)

            # Write the processed frame to the output video
            if out:
                out.write(display_frame)

            # Show the frame
            cv2.imshow("Sleep Symptoms Detector", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Report sleep period statistics
        sleep_stats = self.sleep_detector.get_sleep_stats()
        print(f"\nSleep Statistics:")
        print(f"Total sleep time: {sleep_stats['total_sleep_time']:.2f} seconds")
        print(f"Sleep percentage: {sleep_stats['sleep_percentage']:.2f}%")

        # Print symptom summary
        self._print_symptom_summary()

        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        return True

    def _draw_results(self, frame):
        """Draw detection results on the frame"""
        if self.current_condition in ["Insomnia", "Nightmare"]:
            banner_color = (255, 165, 0) if self.current_condition == "Insomnia" else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), banner_color, -1)
            cv2.putText(frame, f"Detected Condition: {self.current_condition}",
                        (frame.shape[1] // 2 - 150, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        y_offset = 330
        cv2.putText(frame, "Detection Scores:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Nightmare: {self.nightmare_detector.detection_score:.2f}",
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if self.current_condition == "Nightmare" else (200, 200, 200), 2)

        cv2.putText(frame, f"Sleeping: {self.sleep_detector.detection_score:.2f}",
                    (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if self.current_condition == "Sleeping" else (200, 200, 200), 2)

        cv2.putText(frame, f"Insomnia: {self.insomnia_detector.detection_score:.2f}",
                    (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 165, 0) if self.current_condition == "Insomnia" else (200, 200, 200), 2)

        sleep_stats = self.sleep_detector.get_sleep_stats()
        minutes = int(sleep_stats['total_sleep_time'] // 60)
        seconds = int(sleep_stats['total_sleep_time'] % 60)
        cv2.putText(frame, f"Sleep Duration: {minutes}m {seconds}s",
                    (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"Sleep Percentage: {sleep_stats['sleep_percentage']:.1f}%",
                    (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    def _print_symptom_summary(self):
        """Print a summary of detected symptoms"""
        print("\nSymptom Detection Summary:")
        print("-" * 30)

        condition_counts = {
            "Nightmare": 0,
            "Sleeping": 0,
            "Insomnia": 0,
            "None": 0
        }
        total_entries = len(self.condition_history)

        for entry in self.condition_history:
            condition_counts[entry["condition"]] += 1

        for condition, count in condition_counts.items():
            percentage = (count / total_entries * 100) if total_entries > 0 else 0
            print(f"{condition}: {count} detections ({percentage:.1f}%)")

        sleep_stats = self.sleep_detector.get_sleep_stats()
        minutes = int(sleep_stats['total_sleep_time'] // 60)
        seconds = int(sleep_stats['total_sleep_time'] % 60)
        print(f"\nTotal Sleep Duration: {minutes} minutes {seconds} seconds")
        print(f"Sleep Percentage: {sleep_stats['sleep_percentage']:.1f}%")

        if condition_counts["Nightmare"] > total_entries * 0.1:
            print("Warning: Frequent nightmares detected. Consider consulting a sleep specialist.")
        if condition_counts["Insomnia"] > total_entries * 0.2:
            print("Warning: Significant insomnia symptoms detected. Consider consulting a healthcare provider.")

    def reset(self):
        """Reset all detectors and clear history"""
        self.nightmare_detector.reset()
        self.sleep_detector.reset()
        self.insomnia_detector.reset()
        self.current_condition = "None"
        self.condition_confidence = 0.0
        self.condition_history.clear()

if __name__ == "__main__":
    video_path = r"C:/projects/coughing_mediapipe/all.mp4"
    output_path = r"C:/projects/coughing_mediapipe/analyzed_output.mp4"
    detector = SleepSymptomsDetector()
    detector.process_video(video_path, output_path)