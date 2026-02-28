import face_recognition
import cv2
import numpy as np
import csv
import io
import os
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "LEARNING PROJECT", "PHOTOS")
ATTENDANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "attendance_records")

known_face_encodings = []
known_face_names = []

# Attendance for the current session  {name: time_string}
attendance_record: dict[str, str] = {}
total_students: list[str] = []

camera_active = False
video_capture = None
lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dirs():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)


def load_known_faces():
    """Load every .jpg/.png in PHOTOS_DIR and encode the faces."""
    global known_face_encodings, known_face_names, total_students

    known_face_encodings = []
    known_face_names = []

    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError(f"Photos directory not found: {PHOTOS_DIR}")

    for fname in sorted(os.listdir(PHOTOS_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(PHOTOS_DIR, fname)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(fname)[0].upper()
            known_face_names.append(name)

    if not known_face_names:
        raise RuntimeError("No faces found in any photo.")

    total_students = known_face_names.copy()


def _save_csv():
    """Persist today's attendance to a CSV file."""
    _ensure_dirs()
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(ATTENDANCE_DIR, f"{date_str}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Status"])
        for name in total_students:
            if name in attendance_record:
                writer.writerow([name, attendance_record[name], "Present"])
            else:
                writer.writerow([name, "", "Absent"])


def generate_frames():
    """Yield MJPEG frames with face-recognition overlay."""
    global camera_active, video_capture

    while camera_active:
        if video_capture is None or not video_capture.isOpened():
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize for faster recognition
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_names = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small,
                                                             face_locations)
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, encoding)
                name = "Unknown"

                if known_face_encodings:
                    distances = face_recognition.face_distance(
                        known_face_encodings, encoding)
                    best_idx = np.argmin(distances)
                    if matches[best_idx]:
                        name = known_face_names[best_idx]

                face_names.append(name)

                # Record attendance
                with lock:
                    if name != "Unknown" and name not in attendance_record:
                        time_str = datetime.now().strftime("%H:%M:%S")
                        attendance_record[name] = time_str
                        _save_csv()

        # Draw boxes
        for (top, right, bottom, left), name in zip(face_locations,
                                                    face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                          color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    if not camera_active:
        return "", 204
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    global camera_active, video_capture, attendance_record

    if camera_active:
        return jsonify(status="already_running")

    try:
        load_known_faces()
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

    attendance_record = {}
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return jsonify(status="error",
                       message="Could not open camera."), 500

    camera_active = True
    return jsonify(status="started",
                   total=len(total_students),
                   students=total_students)


@app.route("/stop", methods=["POST"])
def stop():
    global camera_active, video_capture

    camera_active = False
    if video_capture is not None:
        video_capture.release()
        video_capture = None

    _save_csv()
    return jsonify(status="stopped")


@app.route("/attendance")
def attendance():
    """Return current attendance data as JSON."""
    with lock:
        present = []
        absent = []
        for name in total_students:
            if name in attendance_record:
                present.append({"name": name,
                                "time": attendance_record[name]})
            else:
                absent.append({"name": name})

    return jsonify(
        total=len(total_students),
        present_count=len(present),
        absent_count=len(absent),
        present=present,
        absent=absent,
        date=datetime.now().strftime("%Y-%m-%d"),
    )


@app.route("/download_csv")
def download_csv():
    """Generate and send attendance CSV for download."""
    with lock:
        si = io.StringIO()
        writer = csv.writer(si)
        writer.writerow(["Name", "Time", "Date", "Status"])
        date_str = datetime.now().strftime("%Y-%m-%d")
        for name in total_students:
            if name in attendance_record:
                writer.writerow([name, attendance_record[name],
                                 date_str, "Present"])
            else:
                writer.writerow([name, "", date_str, "Absent"])

    mem = io.BytesIO()
    mem.write(si.getvalue().encode("utf-8"))
    mem.seek(0)

    filename = f"attendance_{date_str}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                     download_name=filename)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _ensure_dirs()
    port = int(os.environ.get("PORT", 5001))
    print("Starting Smart Attendance System...")
    print(f"Open http://127.0.0.1:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
