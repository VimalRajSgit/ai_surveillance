from flask import Flask, render_template
import threading
import queue
from db_operations import Database
from video_processor import process_video, database_writer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Database connection
db_uri = ""
db = Database(db_uri)

# Queue for communication
cough_queue = queue.Queue()


@app.route('/')
def index():
    cough_events = db.get_recent_cough_events()
    total_cough_count = db.get_total_cough_count()
    return render_template('index.html', cough_events=cough_events, total_cough_count=total_cough_count)


if __name__ == '__main__':
    logging.info("Starting cough monitoring application")
    # Start video processing and database writer in separate threads
    video_thread = threading.Thread(target=process_video, args=(cough_queue,))
    db_thread = threading.Thread(target=database_writer, args=(db, cough_queue,))

    video_thread.daemon = True
    db_thread.daemon = True

    video_thread.start()
    db_thread.start()

    # Start Flask server
    app.run(debug=True, use_reloader=False)