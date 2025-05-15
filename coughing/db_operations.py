import psycopg2

class Database:
    def __init__(self, db_uri):
        self.conn = psycopg2.connect(db_uri)
        self.cur = self.conn.cursor()
        self._initialize_db()

    def _initialize_db(self):
        # Create table if it doesn't exist
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS Patient (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            coughing_time TIME NULL,
            room_no INTEGER
        );
        """)
        # Check if cough_count column exists and add it if not
        self.cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'patient' AND column_name = 'cough_count';
        """)
        if not self.cur.fetchone():
            self.cur.execute("""
            ALTER TABLE Patient ADD COLUMN cough_count INTEGER DEFAULT 0;
            """)
        self.conn.commit()

    def insert_cough_event(self, name, cough_time, room_no, cough_count):
        self.cur.execute(
            "INSERT INTO Patient (name, coughing_time, room_no, cough_count) VALUES (%s, %s, %s, %s)",
            (name, cough_time, room_no, cough_count)
        )
        self.conn.commit()

    def get_recent_cough_events(self, limit=10):
        self.cur.execute("SELECT name, coughing_time, room_no, cough_count FROM Patient ORDER BY id DESC LIMIT %s", (limit,))
        return self.cur.fetchall()

    def get_total_cough_count(self):
        self.cur.execute("SELECT SUM(cough_count) FROM Patient")
        result = self.cur.fetchone()[0]
        return result if result is not None else 0

    def close(self):
        self.cur.close()
        self.conn.close()