from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"

# Load face encodings
def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"], data["roll_numbers"]
    return [], [], []

known_encodings, known_names, known_roll_numbers = load_encodings()

# ========== ROUTES ==========

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'vedanth' and password == 'vedanth123':
            session['user'] = 'admin'
            return redirect(url_for('index'))

        # Check in DB
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user'] = username
            return redirect(url_for('start_attendance'))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()

        flash("Signup successful. Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/index')
def index():
    if session.get('user') == 'admin':
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/start_attendance')
def start_attendance():
    if 'user' in session and session['user'] != 'admin':
        return render_template('start_attendance.html')
    return redirect(url_for('login'))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if session.get('user') != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        roll_no = request.form['roll_no']
        name = request.form['name']
        file = request.files['photo']
        if file:
            filename = f"{roll_no}_{name}.jpg"
            file.save(os.path.join(DATASET_PATH, filename))
            update_encodings()
        return redirect(url_for('admin'))
    return render_template('admin.html')

@app.route('/attendance')
def attendance():
    if session.get('user') != 'admin':
        return redirect(url_for('login'))

    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
    else:
        attendance_df = pd.DataFrame(columns=["Roll No", "Name", "Time"])

    registered_students = []
    for filename in os.listdir(DATASET_PATH):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                roll_no, name = filename.split("_", 1)
                name = os.path.splitext(name)[0]
                registered_students.append({"Roll No": roll_no, "Name": name})
            except ValueError:
                continue

    present_roll_numbers = set(attendance_df["Roll No"].astype(str))
    absent_students = [s for s in registered_students if s["Roll No"] not in present_roll_numbers]
    present_records = attendance_df.to_dict(orient="records")

    return render_template('attendance.html', present_records=present_records, absent_students=absent_students)

@app.route('/video')
def video():
    return render_template('video_feed.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/tech')
def tech():
    return render_template('tech.html')

# ========== UTILS ==========

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            name, roll_no = "Unknown", "N/A"

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                name = known_names[best_match_index]
                roll_no = known_roll_numbers[best_match_index]
                mark_attendance(roll_no, name)

            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{roll_no} - {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def mark_attendance(roll_no, name):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Roll No", "Name", "Time"])

    if not ((df["Roll No"].astype(str) == str(roll_no)) & (df["Time"].str.startswith(now.strftime("%Y-%m-%d")))).any():
        df = pd.concat([df, pd.DataFrame([[roll_no, name, timestamp]], columns=["Roll No", "Name", "Time"])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)

def update_encodings():
    global known_encodings, known_names, known_roll_numbers
    known_encodings, known_names, known_roll_numbers = [], [], []

    for filename in os.listdir(DATASET_PATH):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            try:
                roll_no, name = filename.split("_", 1)
                name = os.path.splitext(name)[0]
                image_path = os.path.join(DATASET_PATH, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                    known_roll_numbers.append(roll_no)
            except ValueError:
                continue

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names, "roll_numbers": known_roll_numbers}, f)
    print("Encodings updated.")

# ========== DB INITIALIZATION ==========
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
