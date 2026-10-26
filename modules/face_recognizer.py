"""Face recognition helper built on the `face_recognition` library (dlib).

Put one photo per person in a folder. The file name (without extension) becomes the
person's name, e.g.  faces/Alice.jpg  ->  "ALICE".

Usage:
    recognizer = FaceRecognizer("faces")
    for face in recognizer.identify(frame):
        print(face["name"], face["distance"], face["box"])

Run this file directly for a webcam demo:  python -m modules.face_recognizer faces
"""

import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class FaceRecognizer:
    def __init__(self, known_faces_dir, tolerance=0.6, scale=0.25):
        """
        known_faces_dir: folder with one photo per person.
        tolerance: max face distance that still counts as a match (lower = stricter).
        scale: frames are shrunk by this factor before searching for faces (faster).
        """
        self.known_faces_dir = Path(known_faces_dir)
        self.tolerance = tolerance
        self.scale = scale
        self.names = []
        self.encodings = []
        self.reload()

    def reload(self):
        """(Re)load and encode every photo in the known-faces folder."""
        self.names, self.encodings = [], []
        if not self.known_faces_dir.is_dir():
            print(f"[face_recognizer] Folder not found: {self.known_faces_dir}")
            return

        for path in sorted(self.known_faces_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            found = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not found:
                print(f"[face_recognizer] No face found in {path.name}, skipping it.")
                continue
            self.names.append(path.stem.upper())
            self.encodings.append(found[0])

    def match(self, encoding):
        """Compare one face encoding to the known faces. Returns (name or None, distance or None)."""
        if not self.encodings:
            return None, None
        distances = face_recognition.face_distance(self.encodings, encoding)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        if distance <= self.tolerance:
            return self.names[best], distance
        return None, distance

    def identify(self, img):
        """Find every face in a BGR frame.

        Returns a list of dicts with:
            name      matched name, or None for an unknown face
            distance  distance to the closest known face (lower = more similar), or None
            encoding  the 128-number face encoding
            box       (left, top, right, bottom) in pixels of the original frame
        """
        small = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(small)
        encodings = face_recognition.face_encodings(small, locations)

        faces = []
        k = 1 / self.scale
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            name, distance = self.match(encoding)
            faces.append({
                "name": name,
                "distance": distance,
                "encoding": encoding,
                "box": (int(left * k), int(top * k), int(right * k), int(bottom * k)),
            })
        return faces

    def process(self, img):
        """Shortcut: names of the recognised (known) faces in the frame."""
        return [face["name"] for face in self.identify(img) if face["name"]]


def main():
    faces_dir = sys.argv[1] if len(sys.argv) > 1 else "faces"
    cap = cv2.VideoCapture(0)
    recognizer = FaceRecognizer(faces_dir)
    print("Known faces:", recognizer.names)

    while True:
        success, img = cap.read()
        if not success:
            break

        for face in recognizer.identify(img):
            left, top, right, bottom = face["box"]
            label = face["name"] or "Unknown"
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Face recognition (press q to quit)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
