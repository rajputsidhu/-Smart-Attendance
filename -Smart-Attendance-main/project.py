import subprocess, sys, os

target = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "LEARNING PROJECT", "project.py")
subprocess.run([sys.executable, target])
