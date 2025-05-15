import cv2
from datetime import datetime
from cough_detector import CoughDetector
import queue
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_video(cough_queue):
    video_path = r"C:\projects\coughing_mediapipe\coughing11.mp4"
    logging.info(f"Attempting to open video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file at {video_path}")
        return

    logging.info(f"Successfully opened video file: {video_path}")
    detector = CoughDetector()
    patient_name = "John Doe"
    room_no = 101
    cough_count = 0
    last_cough_time = None
    min_cough_interval = 0.5  # Minimum seconds between cough detections

    while True:
        success, frame = cap.read()
        if not success:
            logging.info("End of video. Restarting video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue

        is_coughing, debug_frame, confidence = detector.detect_cough(frame)

        current_time = time.time()
        if is_coughing and (last_cough_time is None or (current_time - last_cough_time) >= min_cough_interval):
            cough_count += 1
            cough_time = datetime.now().strftime("%H:%M:%S")
            cough_queue.put((patient_name, cough_time, room_no, cough_count))
            last_cough_time = current_time
            logging.info(f"Cough detected! Total coughs: {cough_count}")

        cv2.imshow('Cough Detection', debug_frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            logging.info("ESC key pressed. Stopping video processing.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Video processing stopped.")


def database_writer(db, cough_queue):
    while True:
        try:
            patient_name, cough_time, room_no, cough_count = cough_queue.get_nowait()
            db.insert_cough_event(patient_name, cough_time, room_no, cough_count)
            logging.info(
                f"Inserted cough event: {patient_name}, {cough_time}, Room {room_no}, Cough Count {cough_count}")
        except queue.Empty:
            time.sleep(0.1)
            continue