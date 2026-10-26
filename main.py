"""
Dementia Face Assistant

Helps a person living with dementia remember who is in front of them.
Press  c  to capture: every face in view is recognised and the app shows
  - for known people: name, relation, affiliation and when they were last seen
  - for strangers: a guest number and how many times they have been seen before
Press  q  to quit.

Known people  : one photo each in faces/known/ + an entry in data/known_faces.xml
Strangers     : saved automatically to faces/unknown/ + data/unknown_faces.xml

Author : Akshat Sahu  (first written 2022)
"""

import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import cv2

from modules.face_detection import FaceDetection
from modules.face_recognizer import FaceRecognizer

# Folder of the .exe when frozen with PyInstaller, otherwise the folder of this script.
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "faces" / "known"
UNKNOWN_DIR = BASE_DIR / "faces" / "unknown"
DATA_DIR = BASE_DIR / "data"
KNOWN_XML = DATA_DIR / "known_faces.xml"
KNOWN_XML_EXAMPLE = DATA_DIR / "known_faces.example.xml"
UNKNOWN_XML = DATA_DIR / "unknown_faces.xml"

MATCH_TOLERANCE = 0.45   # lower = stricter matching
LABEL_SECONDS = 6        # how long results stay on screen after a capture


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_xml(path, root_tag):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_xml(ET.ElementTree(ET.Element(root_tag)), path)
    return ET.parse(path)


def save_xml(tree, path):
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def find_person(root, name):
    """Find a <person> by name, ignoring upper/lower case."""
    for person in root.findall("person"):
        if (person.get("name") or "").upper() == name.upper():
            return person
    return None


def text_of(person, tag, default="Not recorded"):
    element = person.find(tag) if person is not None else None
    return element.text if element is not None and element.text else default


def set_text(person, tag, value):
    element = person.find(tag)
    if element is None:
        element = ET.SubElement(person, tag)
    element.text = str(value)


class FaceMemoryAssistant:
    def __init__(self):
        KNOWN_DIR.mkdir(parents=True, exist_ok=True)
        UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
        if not KNOWN_XML.exists() and KNOWN_XML_EXAMPLE.exists():
            shutil.copy(KNOWN_XML_EXAMPLE, KNOWN_XML)
            print(f"Created {KNOWN_XML.relative_to(BASE_DIR)} from the example. Edit it to add your people.")

        self.known_tree = load_xml(KNOWN_XML, "knownFaces")
        self.unknown_tree = load_xml(UNKNOWN_XML, "unknownFaces")

        print("Loading known faces...")
        self.known = FaceRecognizer(KNOWN_DIR, tolerance=MATCH_TOLERANCE, scale=0.5)
        self.unknown = FaceRecognizer(UNKNOWN_DIR, tolerance=MATCH_TOLERANCE, scale=0.5)
        print(f"Known people: {self.known.names or 'none yet - add photos to faces/known/'}")
        print(f"Saved guests: {len(self.unknown.names)}")

        self.detector = FaceDetection(min_detection_confidence=0.6)
        self.labels = []          # [(box, [lines], color)] from the last capture
        self.labels_until = 0

    # ------------------------------------------------------------------ capture
    def capture(self, frame):
        faces = self.known.identify(frame)
        if not faces:
            print("\nNo face detected - look at the camera and try again.")
            return

        self.labels = []
        for face in faces:
            if face["name"]:
                lines = self.handle_known(face)
                color = (0, 200, 0)
            else:
                lines = self.handle_unknown(face, frame)
                color = (0, 165, 255)
            print("\n--------\n" + "\n".join(lines))
            self.labels.append((face["box"], lines, color))
        self.labels_until = time.time() + LABEL_SECONDS

    def handle_known(self, face):
        name = face["name"]
        root = self.known_tree.getroot()
        person = find_person(root, name)
        if person is None:  # photo exists but no XML entry yet
            person = ET.SubElement(root, "person", name=name.title())

        last_seen = text_of(person, "LastEncounter", "Never")
        set_text(person, "LastEncounter", now())
        save_xml(self.known_tree, KNOWN_XML)

        return [
            f"Name        : {name.title()}",
            f"Relation    : {text_of(person, 'relation')}",
            f"Affiliation : {text_of(person, 'affiliation')}",
            f"Last seen   : {last_seen}",
            f"Match       : {int((1 - face['distance']) * 100)} %",
        ]

    def handle_unknown(self, face, frame):
        root = self.unknown_tree.getroot()
        name, distance = self.unknown.match(face["encoding"])

        if name:  # a stranger we have seen before
            person = find_person(root, name)
            if person is None:
                person = ET.SubElement(root, "person", name=name)
            last_seen = text_of(person, "LastEncounter", "Never")
            count = int(text_of(person, "noOfEncounter", "0")) + 1
            set_text(person, "LastEncounter", now())
            set_text(person, "noOfEncounter", count)
            save_xml(self.unknown_tree, UNKNOWN_XML)
            return [
                f"Stranger    : {name}",
                f"Times seen  : {count}",
                f"Last seen   : {last_seen}",
                f"Match       : {int((1 - distance) * 100)} %",
            ]

        # A brand-new stranger: save their photo and start a record.
        name = self.next_guest_name()
        cv2.imwrite(str(UNKNOWN_DIR / f"{name.title()}.jpg"), self.crop(frame, face["box"]))
        self.unknown.names.append(name)
        self.unknown.encodings.append(face["encoding"])

        person = ET.SubElement(root, "person", name=name)
        set_text(person, "LastEncounter", now())
        set_text(person, "noOfEncounter", 1)
        save_xml(self.unknown_tree, UNKNOWN_XML)
        return [f"New stranger : {name}", "Saved for next time."]

    def next_guest_name(self):
        numbers = [int(m.group(1)) for n in self.unknown.names if (m := re.fullmatch(r"GUEST(\d+)", n))]
        return f"GUEST{max(numbers, default=-1) + 1}"

    @staticmethod
    def crop(frame, box, padding=0.4):
        """Cut the face out of the frame with some margin, staying inside the image."""
        left, top, right, bottom = box
        h, w = frame.shape[:2]
        pad_x, pad_y = int((right - left) * padding), int((bottom - top) * padding)
        return frame[max(0, top - pad_y):min(h, bottom + pad_y), max(0, left - pad_x):min(w, right + pad_x)]

    # ------------------------------------------------------------------ drawing
    def draw(self, frame):
        self.detector.face_draw(frame, self.detector.process(frame))

        if time.time() < self.labels_until:
            for (left, top, right, bottom), lines, color in self.labels:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                for i, line in enumerate(lines[:3]):
                    y = bottom + 22 + i * 22
                    cv2.putText(frame, " ".join(line.split()), (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "c: who is this?   q: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    # ------------------------------------------------------------------ main loop
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("Could not open the webcam.")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    self.capture(frame)
                elif key == ord("q"):
                    break

                cv2.imshow("Dementia Face Assistant", self.draw(frame))
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    assistant = FaceMemoryAssistant()
    if "--check" in sys.argv:  # load all models, then exit without opening the webcam
        print("OK - models loaded")
    else:
        assistant.run()
