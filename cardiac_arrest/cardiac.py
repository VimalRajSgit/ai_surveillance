import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose (single instance)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False,
                    min_detection_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

# Global variables for state persistence
pose_history = deque(maxlen=150)  # Temporal window of 150 frames (~5 seconds at 30 FPS)
detected_poses = set()


# Function to compute neck position (midpoint of shoulders)
def compute_neck(landmarks):
    shoulder_left = landmarks[11] if landmarks[11].visibility > 0.2 else None
    shoulder_right = landmarks[12] if landmarks[12].visibility > 0.2 else None

    if shoulder_left and shoulder_right:
        neck_x = (shoulder_left.x + shoulder_right.x) / 2
        neck_y = (shoulder_left.y + shoulder_right.y) / 2
        neck_z = (shoulder_left.z + shoulder_right.z) / 2
        neck_visibility = min(shoulder_left.visibility, shoulder_right.visibility)
        return {'x': neck_x, 'y': neck_y, 'z': neck_z, 'visibility': neck_visibility}
    return None


# Function to compute angle between three points (for leg bending)
def compute_angle(p1, p2, p3):
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# Function to check Pose 1: Lying on Side, Hands Near Chest, Legs Bent
def check_pose1(landmarks, neck):
    if not neck:
        return False

    wrist_left = landmarks[15]
    wrist_right = landmarks[16]
    if not (wrist_left.visibility > 0.2 and wrist_right.visibility > 0.2):
        return False

    wrist_chest_dist_left = np.sqrt((wrist_left.x - neck['x']) ** 2 + (wrist_left.y - neck['y']) ** 2)
    wrist_chest_dist_right = np.sqrt((wrist_right.x - neck['x']) ** 2 + (wrist_right.y - neck['y']) ** 2)
    wrists_near_chest = wrist_chest_dist_left < 0.2 and wrist_chest_dist_right < 0.2

    head = landmarks[0]
    if not head.visibility > 0.2:
        return False
    head_neck_y_diff = head.y - neck['y']
    head_position = -0.1 < head_neck_y_diff < 0.2

    left_hip, left_knee, left_ankle = landmarks[23], landmarks[25], landmarks[27]
    right_hip, right_knee, right_ankle = landmarks[24], landmarks[26], landmarks[28]
    if not all(lm.visibility > 0.2 for lm in [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
        return False

    left_leg_angle = compute_angle({'x': left_hip.x, 'y': left_hip.y},
                                   {'x': left_knee.x, 'y': left_knee.y},
                                   {'x': left_ankle.x, 'y': left_ankle.y})
    right_leg_angle = compute_angle({'x': right_hip.x, 'y': right_hip.y},
                                    {'x': right_knee.x, 'y': right_knee.y},
                                    {'x': right_ankle.x, 'y': right_ankle.y})
    legs_bent = left_leg_angle < 175 and right_leg_angle < 175

    shoulder_left, shoulder_right = landmarks[11], landmarks[12]
    shoulder_x_diff = abs(shoulder_left.x - shoulder_right.x)
    body_on_side = shoulder_x_diff < 0.15

    return wrists_near_chest and head_position and legs_bent and body_on_side


# Function to check Pose 2: Lying Flat, Hands on Chest, Legs Straight
def check_pose2(landmarks, neck):
    if not neck:
        return False

    wrist_left = landmarks[15]
    wrist_right = landmarks[16]
    if not (wrist_left.visibility > 0.2 and wrist_right.visibility > 0.2):
        return False

    wrist_chest_dist_left = np.sqrt((wrist_left.x - neck['x']) ** 2 + (wrist_left.y - neck['y']) ** 2)
    wrist_chest_dist_right = np.sqrt((wrist_right.x - neck['x']) ** 2 + (wrist_right.y - neck['y']) ** 2)
    wrists_on_chest = wrist_chest_dist_left < 0.2 and wrist_chest_dist_right < 0.2

    head = landmarks[0]
    if not head.visibility > 0.2:
        return False
    head_neck_y_diff = abs(head.y - neck['y'])
    head_aligned = head_neck_y_diff < 0.15

    left_hip, left_knee, left_ankle = landmarks[23], landmarks[25], landmarks[27]
    right_hip, right_knee, right_ankle = landmarks[24], landmarks[26], landmarks[28]
    if not all(lm.visibility > 0.2 for lm in [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
        return False

    left_leg_angle = compute_angle({'x': left_hip.x, 'y': left_hip.y},
                                   {'x': left_knee.x, 'y': left_knee.y},
                                   {'x': left_ankle.x, 'y': left_ankle.y})
    right_leg_angle = compute_angle({'x': right_hip.x, 'y': right_hip.y},
                                    {'x': right_knee.x, 'y': right_knee.y},
                                    {'x': right_ankle.x, 'y': right_ankle.y})
    legs_straight = left_leg_angle > 135 and right_leg_angle > 135

    shoulder_left, shoulder_right = landmarks[11], landmarks[12]
    shoulder_x_diff = abs(shoulder_left.x - shoulder_right.x)
    body_flat = 0.05 < shoulder_x_diff < 0.4

    return wrists_on_chest and head_aligned and legs_straight and body_flat


# Function to check Pose 3: Fetal Position, Hands Near Chest, Legs Tucked
def check_pose3(landmarks, neck):
    if not neck:
        return False

    wrist_left = landmarks[15]
    wrist_right = landmarks[16]
    if not (wrist_left.visibility > 0.2 and wrist_right.visibility > 0.2):
        return False

    wrist_chest_dist_left = np.sqrt((wrist_left.x - neck['x']) ** 2 + (wrist_left.y - neck['y']) ** 2)
    wrist_chest_dist_right = np.sqrt((wrist_right.x - neck['x']) ** 2 + (wrist_right.y - neck['y']) ** 2)
    wrists_near_chest = wrist_chest_dist_left < 0.2 and wrist_chest_dist_right < 0.2

    head = landmarks[0]
    if not head.visibility > 0.2:
        return False

    left_hip, left_knee, left_ankle = landmarks[23], landmarks[25], landmarks[27]
    right_hip, right_knee, right_ankle = landmarks[24], landmarks[26], landmarks[28]
    if not all(lm.visibility > 0.2 for lm in [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]):
        return False

    left_leg_angle = compute_angle({'x': left_hip.x, 'y': left_hip.y},
                                   {'x': left_knee.x, 'y': left_knee.y},
                                   {'x': left_ankle.x, 'y': left_ankle.y})
    right_leg_angle = compute_angle({'x': right_hip.x, 'y': right_hip.y},
                                    {'x': right_knee.x, 'y': right_knee.y},
                                    {'x': right_ankle.x, 'y': right_ankle.y})
    legs_tucked = left_leg_angle < 145 and right_leg_angle < 145

    knee_chest_dist_left = np.sqrt((left_knee.x - neck['x']) ** 2 + (left_knee.y - neck['y']) ** 2)
    knee_chest_dist_right = np.sqrt((right_knee.x - neck['x']) ** 2 + (right_knee.y - neck['y']) ** 2)
    knees_near_chest = knee_chest_dist_left < 0.5 and knee_chest_dist_right < 0.5

    shoulder_left, shoulder_right = landmarks[11], landmarks[12]
    shoulder_x_diff = abs(shoulder_left.x - shoulder_right.x)
    body_curled = shoulder_x_diff < 0.15

    return wrists_near_chest and legs_tucked and knees_near_chest and body_curled


# Function to detect cardiac arrest in a single frame
def detect_cardiac_arrest(image):
    global pose_history, detected_poses
    frame_count = len(pose_history) + 1

    # Ensure image is writeable before processing
    image.flags.writeable = False

    # Process frame with MediaPipe
    results = pose.process(image)

    # Make image writeable again before drawing
    image.flags.writeable = True

    # Check poses if landmarks are detected
    cardiac_arrest_detected = False
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Compute neck
        neck = compute_neck(landmarks)

        # Check each pose
        for pose_id, check_func in enumerate([check_pose1, check_pose2, check_pose3]):
            if check_func(landmarks, neck):
                pose_history.append((frame_count, pose_id))
                detected_poses.add(pose_id)

        # Check if at least 2 poses have been detected
        if len(detected_poses) >= 2:
            cardiac_arrest_detected = True

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # Display cardiac arrest message on the frame if detected
    if cardiac_arrest_detected:
        cv2.putText(image, "Cardiac Arrest Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, cardiac_arrest_detected