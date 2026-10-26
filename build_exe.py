"""
Build a Windows executable with PyInstaller.

    pip install pyinstaller
    python build_exe.py

Output: dist/DementiaFaceAssistant/DementiaFaceAssistant.exe  (copy the whole folder to share it)
"""

import shutil
from pathlib import Path

import PyInstaller.__main__

APP_NAME = "DementiaFaceAssistant"
ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"

PyInstaller.__main__.run([
    str(ROOT / "main.py"),
    "--name", APP_NAME,
    "--onedir",                                   # fast start-up; one-file would unpack ~300 MB on every launch
    "--noconfirm",
    "--clean",
    "--distpath", str(DIST),
    "--workpath", str(BUILD),
    "--specpath", str(BUILD),
    "--collect-data", "mediapipe",                # face detection model files
    "--collect-data", "face_recognition_models",  # dlib face models
    "--exclude-module", "tkinter",
])

# Folders the app reads and writes at run time live next to the .exe.
# Only templates and instructions are copied - never real photos or people data.
app_dir = DIST / APP_NAME
for sub in ("known", "unknown"):
    (app_dir / "faces" / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "faces" / sub / "README.md", app_dir / "faces" / sub / "README.md")
(app_dir / "data").mkdir(exist_ok=True)
shutil.copy(ROOT / "data" / "known_faces.example.xml", app_dir / "data" / "known_faces.example.xml")
shutil.copy(ROOT / "README.md", app_dir / "README.md")

print(f"\nDone: {app_dir / (APP_NAME + '.exe')}")
