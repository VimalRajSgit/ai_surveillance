import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import math
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to analyze gait pattern with posture detection
def analyze_gait(landmarks, frame_height, frame_width, history, fps):
    """
    Analyze gait pattern with posture detection, temporal features, and dynamic normalization
    Parameters:
        landmarks: MediaPipe pose landmarks
        frame_height, frame_width: Video frame dimensions
        history: Dictionary of deques to store temporal data
        fps: Frames per second of the video
    Returns:
        Gait pattern classification: "Normal Walking", "Injured Walking", "Left-sided Walking", or "Not Walking"
        confidence_score: Confidence in the classification (0-100%)
        debug_info: Dictionary of metrics for debugging
    """
    # Convert landmarks to numpy array with normalized coordinates
    lm_pose = np.array([[lm.x * frame_width, lm.y * frame_height, lm.z * frame_width, lm.visibility]
                        for lm in landmarks])

    # Extract key points (excluding wrists and hands)
    left_hip = lm_pose[mp_pose.PoseLandmark.LEFT_HIP.value][:3]
    right_hip = lm_pose[mp_pose.PoseLandmark.RIGHT_HIP.value][:3]
    left_knee = lm_pose[mp_pose.PoseLandmark.LEFT_KNEE.value][:3]
    right_knee = lm_pose[mp_pose.PoseLandmark.RIGHT_KNEE.value][:3]
    left_ankle = lm_pose[mp_pose.PoseLandmark.LEFT_ANKLE.value][:3]
    right_ankle = lm_pose[mp_pose.PoseLandmark.RIGHT_ANKLE.value][:3]
    left_foot_index = lm_pose[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][:3]
    right_foot_index = lm_pose[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][:3]
    left_shoulder = lm_pose[mp_pose.PoseLandmark.LEFT_SHOULDER.value][:3]
    right_shoulder = lm_pose[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:3]

    # Estimate body height (hip to ankle) for normalization
    left_leg_length = calculate_distance(left_hip[:2], left_ankle[:2])
    right_leg_length = calculate_distance(right_hip[:2], right_ankle[:2])
    body_height = (left_leg_length + right_leg_length) / 2
    if body_height == 0:
        body_height = frame_height  # Fallback to frame height

    # Posture Detection: Check if the person is standing or lying down
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    shoulder_hip_diff = abs(avg_shoulder_y - avg_hip_y) / body_height

    # If shoulders and hips are at similar vertical levels, person is likely lying down
    if shoulder_hip_diff < 0.2:  # Threshold for lying down (normalized by body height)
        classification = "Not Walking"
        confidence = 100  # High confidence for posture-based classification
        debug_info = {
            "shoulder_hip_diff": shoulder_hip_diff,
            "posture": "Lying Down"
        }
        return classification, confidence, debug_info

    # Proceed with gait analysis if person is standing
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    knee_angle_diff = abs(left_knee_angle - right_knee_angle)

    hip_height_diff = abs(left_hip[1] - right_hip[1]) / body_height
    ankle_height_diff = abs(left_ankle[1] - right_ankle[1]) / body_height
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / body_height
    feet_lateral_diff = abs(left_foot_index[0] - right_foot_index[0]) / body_height

    l_foot_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    r_foot_angle = calculate_angle(right_knee, right_ankle, right_foot_index)
    foot_angle_diff = abs(l_foot_angle - r_foot_angle)

    history['left_ankle'].append(left_ankle[0:2])
    history['right_ankle'].append(right_ankle[0:2])
    history['timestamps'].append(time.time())

    stride_length_left = 0
    stride_length_right = 0
    step_frequency_left = 0
    step_frequency_right = 0
    asymmetry_score = 0

    if len(history['left_ankle']) >= 10:
        left_positions = np.array(history['left_ankle'])
        right_positions = np.array(history['right_ankle'])
        timestamps = np.array(history['timestamps'])

        left_x = left_positions[:, 0]
        right_x = right_positions[:, 0]
        left_steps = []
        right_steps = []

        for i in range(1, len(left_x) - 1):
            if (left_x[i] > left_x[i-1] and left_x[i] > left_x[i+1]) or \
               (left_x[i] < left_x[i-1] and left_x[i] < left_x[i+1]):
                left_steps.append((left_positions[i], timestamps[i]))
            if (right_x[i] > right_x[i-1] and right_x[i] > left_x[i+1]) or \
               (right_x[i] < right_x[i-1] and right_x[i] < left_x[i+1]):
                right_steps.append((right_positions[i], timestamps[i]))

        if len(left_steps) >= 2:
            stride_length_left = calculate_distance(left_steps[-1][0], left_steps[-2][0]) / body_height
        if len(right_steps) >= 2:
            stride_length_right = calculate_distance(right_steps[-1][0], right_steps[-2][0]) / body_height

        if len(left_steps) >= 2:
            time_diff = left_steps[-1][1] - left_steps[-2][1]
            step_frequency_left = 1 / time_diff if time_diff > 0 else 0
        if len(right_steps) >= 2:
            time_diff = right_steps[-1][1] - right_steps[-2][1]
            step_frequency_right = 1 / time_diff if time_diff > 0 else 0

        stride_asymmetry = abs(stride_length_left - stride_length_right)
        frequency_asymmetry = abs(step_frequency_left - step_frequency_right)
        asymmetry_score = (stride_asymmetry * 20) + (frequency_asymmetry * 10)

    velocity_left = 0
    velocity_right = 0
    if len(history['left_ankle']) >= 2:
        dx_left = history['left_ankle'][-1][0] - history['left_ankle'][-2][0]
        dy_left = history['left_ankle'][-1][1] - history['left_ankle'][-2][1]
        dt = 1 / fps
        velocity_left = np.sqrt(dx_left**2 + dy_left**2) / dt / body_height
        dx_right = history['right_ankle'][-1][0] - history['right_ankle'][-2][0]
        dy_right = history['right_ankle'][-1][1] - history['right_ankle'][-2][1]
        velocity_right = np.sqrt(dx_right**2 + dy_right**2) / dt / body_height

    injured_score = 0
    left_sided_score = 0
    normal_score = 0

    if knee_angle_diff > 30:
        injured_score += 40
    elif knee_angle_diff > 20:
        injured_score += 20
    if ankle_height_diff > 0.06:
        injured_score += 30
    elif ankle_height_diff > 0.04:
        injured_score += 15
    if (left_knee_angle > 170 and right_knee_angle < 145) or (right_knee_angle > 170 and left_knee_angle < 145):
        injured_score += 50
    if stride_length_left < 0.25 or stride_length_right < 0.25:
        injured_score += 30
    if asymmetry_score > 15:
        injured_score += 40
    if abs(velocity_left - velocity_right) > 0.7:
        injured_score += 30

    if hip_height_diff > 0.05:
        left_sided_score += 30
    elif hip_height_diff > 0.03:
        left_sided_score += 15
    if shoulder_tilt > 0.05:
        left_sided_score += 25
    elif shoulder_tilt > 0.03:
        left_sided_score += 10
    if feet_lateral_diff > 0.15:
        left_sided_score += 40
    if left_foot_index[0] > right_foot_index[0] + (body_height * 0.1):
        left_sided_score += 40
    left_x_positions = [pos[0] for pos in history['left_ankle']][-5:]
    if len(left_x_positions) >= 5 and all(left_x_positions[i] > left_x_positions[i-1] for i in range(1, len(left_x_positions))):
        left_sided_score += 30

    normal_score = 100 - (knee_angle_diff * 1.0) - (ankle_height_diff * 500) - (hip_height_diff * 500) - \
                   (shoulder_tilt * 500) - (asymmetry_score * 0.5)
    normal_score = max(0, normal_score)

    max_score = max(injured_score, left_sided_score, normal_score)
    confidence = min(100, max_score)

    if injured_score == max_score:
        classification = "Injured Walking"
    elif left_sided_score == max_score:
        classification = "Left-sided Walking"
    else:
        classification = "Normal Walking"

    debug_info = {
        "knee_angle_diff": knee_angle_diff,
        "ankle_height_diff": ankle_height_diff,
        "hip_height_diff": hip_height_diff,
        "shoulder_tilt": shoulder_tilt,
        "feet_lateral_diff": feet_lateral_diff,
        "stride_length_left": stride_length_left,
        "stride_length_right": stride_length_right,
        "step_frequency_left": step_frequency_left,
        "step_frequency_right": step_frequency_right,
        "asymmetry_score": asymmetry_score,
        "velocity_diff": abs(velocity_left - velocity_right),
        "injured_score": injured_score,
        "left_sided_score": left_sided_score,
        "normal_score": normal_score,
        "shoulder_hip_diff": shoulder_hip_diff,
        "posture": "Standing"
    }

    return classification, confidence, debug_info

def classify_gait_from_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    if not os.access(video_path, os.R_OK):
        print(f"Error: No read permission for video file {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}. Check if the file is a valid video format or not corrupted.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    history = {
        'left_ankle': deque(maxlen=30),
        'right_ankle': deque(maxlen=30),
        'timestamps': deque(maxlen=30),
        'classifications': deque(maxlen=20),
        'confidences': deque(maxlen=20)
    }

    frame_count = 0
    found_100_confidence = False
    main_output = None

    print("Press 'q' to quit the video display")
    if not found_100_confidence:
        print("Debug metrics will be printed every 30 frames unless 100% confidence is reached.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            gait_classification, confidence, debug_info = analyze_gait(
                results.pose_landmarks.landmark, frame_height, frame_width, history, fps
            )

            history['classifications'].append(gait_classification)
            history['confidences'].append(confidence)

            if len(history['classifications']) >= 10:
                weighted_scores = {"Normal Walking": 0, "Injured Walking": 0, "Left-sided Walking": 0, "Not Walking": 0}
                total_weight = 0
                for cls, conf in zip(history['classifications'], history['confidences']):
                    weighted_scores[cls] += conf
                    total_weight += conf
                gait_classification = max(weighted_scores, key=weighted_scores.get)
                confidence = weighted_scores[gait_classification] / total_weight * 100 if total_weight > 0 else confidence

            if confidence >= 100 and not found_100_confidence:
                found_100_confidence = True
                main_output = gait_classification
                if main_output == "Left-sided Walking":
                    main_output = "Injured Walking"
                    print(f"Main Output: {main_output}")
                else:
                    print(f"Main Output: Left-sided Walking")

            if "Injured" in gait_classification:
                gait_classification = "Left-sided Walking"
                color = (0, 0, 255)
            elif "Left-sided" in gait_classification:
                gait_classification ="Injured Walking"
                color = (255, 165, 0)
            elif "Not Walking" in gait_classification:
                color = (128, 128, 128)  # Gray for Not Walking
            else:
                color = (0, 255, 0)

            cv2.putText(frame, f"{gait_classification} (myconfidence: {confidence:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
            right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
            left_ankle_point = (int(results.pose_landmarks.landmark[left_ankle_idx].x * frame_width),
                                int(results.pose_landmarks.landmark[left_ankle_idx].y * frame_height))
            right_ankle_point = (int(results.pose_landmarks.landmark[right_ankle_idx].x * frame_width),
                                 int(results.pose_landmarks.landmark[right_ankle_idx].y * frame_height))
            cv2.circle(frame, left_ankle_point, 8, (0, 255, 255), -1)
            cv2.circle(frame, right_ankle_point, 8, (0, 255, 255), -1)

            if not found_100_confidence and frame_count % 30 == 0:
                print(f"\nFrame {frame_count}:")
                print(f"Classification: {gait_classification} (Confidence: {confidence:.1f}%)")
                for key, value in debug_info.items():
                    print(f"{key}: {value:.2f}")

        else:
            cv2.putText(frame, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Gait Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    video_path = r"C:\projects\coughing_mediapipe\coughing\gait.mp4"
    print("Starting gait classification...")
    print("Results will be displayed in a window - no output file will be created")
    classify_gait_from_video(video_path)
    print("Analysis complete.")

if __name__ == "__main__":
    main()