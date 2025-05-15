import cv2
import mediapipe as mp
import numpy as np
import math

class PhoneUsageDetector:
    def __init__(self, angle_threshold=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.angle_threshold = angle_threshold
        self.start_frame = None
        self.total_look_frames = 0
        self.look_detected = False

    def get_head_angle(self, nose, neck):
        # Calculate vertical angle between neck and nose
        dx = nose.x - neck.x
        dy = neck.y - nose.y  # neck is below nose
        angle = math.degrees(math.atan2(dy, dx))
        return abs(angle)

    def is_looking_at_phone(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2
        neck = type(nose)(x=neck_x, y=neck_y, z=0)

        head_angle = self.get_head_angle(nose, neck)
        return head_angle > self.angle_threshold

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps  # total video duration in seconds
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if self.is_looking_at_phone(landmarks):
                    if not self.look_detected:
                        self.start_frame = current_frame
                        self.look_detected = True
                    elapsed_frames = current_frame - self.start_frame
                    elapsed_seconds = elapsed_frames / fps
                    cv2.putText(frame, f"Looking at phone: {int(elapsed_seconds)} sec", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if self.look_detected:
                        # Accumulate the elapsed frames when the "looking at phone" period ends
                        elapsed_frames = current_frame - self.start_frame
                        self.total_look_frames += elapsed_frames
                    self.look_detected = False
                    self.start_frame = None
                    cv2.putText(frame, f"Not looking at phone", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Phone Usage Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Add the final period if the video ends while looking at the phone
        if self.look_detected:
            elapsed_frames = current_frame - self.start_frame
            self.total_look_frames += elapsed_frames

        # Convert total frames to seconds
        total_look_time = self.total_look_frames / fps
        # Ensure total_look_time does not exceed total_duration
        total_look_time = min(total_look_time, total_duration)

        cap.release()
        cv2.destroyAllWindows()
        return round(total_look_time, 2), round(total_duration, 2)


if __name__ == "__main__":
    video_path = r"C:\projects\coughing_mediapipe\phone videos\phonee.mp4"  # <-- Replace with your video path
    detector = PhoneUsageDetector(angle_threshold=30)
    total_look_time, total_duration = detector.process_video(video_path)
    print(f"\nTotal time looking at phone: {total_look_time} seconds")
    print(f"Total video duration: {total_duration} seconds")