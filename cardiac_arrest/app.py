import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from cardiac import detect_cardiac_arrest

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to track detection status
detection_status = {"cardiac_arrest_detected": False}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Reset detection status for new video
        detection_status["cardiac_arrest_detected"] = False
        return jsonify({'filepath': filepath}), 200

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/stream/<path:filepath>')
def stream_video(filepath):
    def generate():
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'Error: Could not open video.\r\n'
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results, is_detected = detect_cardiac_arrest(image)

            if is_detected and not detection_status["cardiac_arrest_detected"]:
                detection_status["cardiac_arrest_detected"] = True

            image = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    def generate():
        while True:
            if detection_status["cardiac_arrest_detected"]:
                yield "data: Cardiac Arrest Detected\n\n"
                break
            yield "data: Waiting\n\n"
            # Sleep to prevent excessive CPU usage
            import time
            time.sleep(1)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)