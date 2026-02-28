# Smart Attendance System (Face Recognition)

## Overview

The Smart Attendance System is an automated solution for accurate and efficient attendance management. Using facial recognition technology, the system detects and identifies individuals in real-time, marking their attendance automatically and storing records securely. This eliminates manual attendance, reduces errors, and ensures authenticity.

## Features

- Real-time face detection and recognition via webcam
- Web-based dashboard with live camera feed
- Real-time attendance stats (Present / Absent / Total / Attendance %)
- Attendance log with timestamps
- Export attendance as CSV file
- Auto-detects student photos from the `LEARNING PROJECT/PHOTOS/` folder

## Technology Stack

- **Language:** Python 3.10+
- **Web Framework:** Flask
- **Libraries:** OpenCV, NumPy, face_recognition (dlib)
- **Frontend:** HTML, CSS, JavaScript
- **Storage:** CSV files

## Project Structure

```
Smart-Attendance/
├── app.py                        # Flask web application (main entry point)
├── requirements.txt              # Python dependencies
├── templates/
│   └── index.html                # Web dashboard
├── attendance_records/           # Auto-generated attendance CSVs
├── LEARNING PROJECT/
│   ├── project.py                # Original Tkinter app (legacy)
│   └── PHOTOS/                   # Student face photos
│       ├── SIDHU.jpg
│       ├── NAMO.jpg
│       ├── YOGI.jpg
│       └── RAGA.jpg
├── LICENSE
└── README.md
```

## Setup Instructions

### macOS / Linux

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajputsidhu/-Smart-Attendance.git
   cd -Smart-Attendance
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://127.0.0.1:5001
   ```

### Windows

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/).
   Make sure to check **"Add Python to PATH"** during installation.

2. **Install CMake:**
   ```bash
   pip install cmake
   ```

3. **Install Visual Studio Build Tools** (required for dlib compilation):
   - Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - During installation, select **"Desktop development with C++"** workload
   - Restart your computer after installation

4. **Install dlib:**
   ```bash
   pip install dlib
   ```
   If this fails, try the pre-built binary instead:
   ```bash
   pip install dlib-bin
   ```

5. **Clone the repository:**
   ```bash
   git clone https://github.com/rajputsidhu/-Smart-Attendance.git
   cd -Smart-Attendance
   ```

6. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

7. **Run the application:**
   ```bash
   python app.py
   ```

8. **Open in browser:**
   ```
   http://127.0.0.1:5001
   ```

## Usage

1. Add student photos (`.jpg` or `.png`) to the `LEARNING PROJECT/PHOTOS/` folder. The filename becomes the student name (e.g., `JOHN.jpg` registers as "JOHN").
2. Run `python app.py` and open `http://127.0.0.1:5001` in your browser.
3. Click **Start Recognition** to begin the live camera feed.
4. The system automatically detects faces and marks attendance in real-time.
5. Monitor the dashboard for present/absent counts and attendance percentage.
6. Click **Download CSV** to export the attendance report.
7. Click **Stop** when done.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `dlib` fails to install on Windows | Install Visual Studio Build Tools with C++ workload, or use `pip install dlib-bin` |
| Camera not opening | Make sure no other app is using the webcam |
| Port 5001 already in use | Set a different port: `PORT=5002 python app.py` |
| No face found in photo | Ensure the photo has a clear, front-facing face |

## Future Enhancements

- Integrate mobile app support for remote attendance tracking
- Add multi-camera support for large classrooms or offices
- Implement advanced analytics for attendance trends and reports
- Add database integration (MySQL/MongoDB)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
