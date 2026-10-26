# Dementia Face Assistant

A webcam app that helps a person living with dementia remember **who is in front of
them**. Press `c` and it recognises every face in view:

- **Known people** (family, friends, carers): it shows their name, relation, affiliation
  and **when you last saw them**.
- **Strangers**: it saves them automatically as `GUEST0`, `GUEST1` and so on, and counts
  **how many times you have met them**. Someone who keeps coming back is easy to spot.

My 4th-semester computer-vision project, built with **OpenCV**, **MediaPipe** and **face_recognition**.

## Example

```
--------
Name        : Alice
Relation    : Daughter
Affiliation : Family
Last seen   : 2026-03-31 21:46:05
Match       : 62 %

--------
Stranger    : GUEST0
Times seen  : 3
Last seen   : 2026-04-01 18:18:54
```

The same details also appear under each face in the video window for a few seconds.

## Windows app (no Python needed)

1. Download `DementiaFaceAssistant-windows-x64.zip` from the
   [Releases](https://github.com/AzureSky0/dementia-face-assistant/releases/latest) page and unzip it.
2. Add photos to `faces/known/` next to the `.exe` (see "Add the people to recognise" below).
3. Double-click `DementiaFaceAssistant.exe`. On first run it creates
   `data/known_faces.xml`. Edit that file to add each person's relation and affiliation.

Windows SmartScreen may warn about an unknown publisher because the app isn't signed.
Click **More info**, then **Run anyway**. To check that the app loads without turning on
the camera, run `DementiaFaceAssistant.exe --check`.

To build the `.exe` yourself, run `pip install pyinstaller`, then `python build_exe.py`. The
output goes to `dist/DementiaFaceAssistant/`.

## Setup (from source)

You need Python 3.9–3.12 and a webcam.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then install `face_recognition` and its engine, `dlib`:

```bash
# Windows: prebuilt dlib, no compiler needed
pip install dlib-bin
pip install face_recognition --no-deps

# macOS / Linux: builds dlib from source, needs CMake
# (brew install cmake  |  sudo apt install cmake build-essential)
pip install face_recognition
```

Tested on Windows 11 with Python 3.12, mediapipe 0.10.14, OpenCV 5.0, numpy 2.x and dlib-bin 20.0.

### Add the people to recognise

1. Put one clear photo per person in [`faces/known/`](faces/known/) and name it after
   them, for example `faces/known/Alice.jpg`.
2. Run the app once. It creates `data/known_faces.xml` from
   [`data/known_faces.example.xml`](data/known_faces.example.xml).
3. Edit `data/known_faces.xml` so there is one entry per photo:

```xml
<person name="Alice">
    <relation>Daughter</relation>
    <affiliation>Family</affiliation>
    <LastEncounter>Never</LastEncounter>
</person>
```

Git ignores photos and the real XML files, so nobody's face or personal details ever get
uploaded.

## Run

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `c` | Identify everyone in view |
| `q` | Quit |

## Project structure

```
dementia-face-assistant/
├── main.py                       # The app: capture, identify, remember, display
├── build_exe.py                  # Builds the Windows .exe with PyInstaller
├── modules/
│   ├── face_recognizer.py        # FaceRecognizer: dlib face encodings and matching
│   └── face_detection.py         # FaceDetection: live MediaPipe face boxes
├── faces/
│   ├── known/                    # Photos of known people (git-ignored)
│   └── unknown/                  # Strangers saved automatically (git-ignored)
├── data/
│   ├── known_faces.example.xml   # Template for known people
│   ├── known_faces.xml           # Your people (created on first run, git-ignored)
│   └── unknown_faces.xml         # Stranger log (created on first run, git-ignored)
└── requirements.txt
```

## How it works

1. When you press `c`, `face_recognition` finds every face in the frame and turns each
   one into a 128-number encoding.
2. Each encoding is compared with the photos in `faces/known/`. A distance of
   **0.45 or less** counts as a match.
3. If nobody matches, it is compared with the saved strangers in `faces/unknown/`.
4. If it is still new, the face is cropped, saved as `GuestN.jpg` and logged.
5. Every match updates `LastEncounter` (and the encounter count for strangers) in the XML
   files.

## Ideas for the future

- Read the name and relation aloud with text-to-speech.
- Add a simple screen for turning a guest into a known person.
- Recognise faces continuously instead of only when you press a key.

## License

[MIT](LICENSE) © Akshat Sahu
