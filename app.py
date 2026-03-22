import subprocess, sys, os

target = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "-Smart-Attendance-main", "app.py")
subprocess.run([sys.executable, target])
