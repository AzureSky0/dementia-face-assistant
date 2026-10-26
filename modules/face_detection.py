"""Face detection helper built on MediaPipe Face Detection.

MediaPipe returns 6 key points per face:
    0 right eye, 1 left eye, 2 nose tip, 3 mouth, 4 right ear, 5 left ear

Run this file directly for a webcam demo:  python -m modules.face_detection
"""

import math

import cv2
import mediapipe as mp


class FaceDetection:
    def __init__(self, min_detection_confidence=0.8):
        self.mpFaceDetec = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetec = self.mpFaceDetec.FaceDetection(min_detection_confidence=min_detection_confidence)

    def process(self, img):
        """Run detection on a BGR frame. Use result.detections (None if no face)."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.faceDetec.process(img_rgb)

    def face_draw(self, img, result):
        """Draw a box and key points for every detected face."""
        if result.detections:
            for detection in result.detections:
                self.mpDraw.draw_detection(img, detection)
        return img

    @staticmethod
    def find_key_point_location(img, result, keypoint, face_no=0):
        """Pixel (x, y) of one key point on one face, or None if there is no such face."""
        if not result.detections or face_no >= len(result.detections):
            return None
        p_loc = result.detections[face_no].location_data.relative_keypoints[keypoint]
        h, w, _ = img.shape
        return int(p_loc.x * w), int(p_loc.y * h)

    def find_hand_to_keypoint_distance(self, img, lm_list, face_result, finger_point=8, face_point_index=2, draw=False):
        """Pixel distance between a hand landmark (default: index fingertip) and a face key point."""
        if not lm_list:
            return None
        face_point = self.find_key_point_location(img, face_result, face_point_index)
        if face_point is None:
            return None

        x1, y1 = lm_list[finger_point][1:]
        x2, y2 = face_point
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (250, 0, 0), 2)
        return int(math.hypot(x2 - x1, y2 - y1))


def main():
    # Hand tracking is optional; it is only used to show the fingertip-to-nose distance.
    try:
        from modules.hand_tracking import HandDetector
        hand = HandDetector()
    except ImportError:
        hand = None

    cap = cv2.VideoCapture(0)
    face_detec = FaceDetection()

    while True:
        success, img = cap.read()
        if not success:
            break

        result = face_detec.process(img)
        face_detec.face_draw(img, result)

        if hand:
            img = hand.find_hands(img)
            lm_list = hand.find_hand_pos(img)
            length = face_detec.find_hand_to_keypoint_distance(img, lm_list, result, 8, 2, draw=True)
            if length is not None:
                print("Fingertip to nose:", length)

        right_eye = face_detec.find_key_point_location(img, result, 0)
        if right_eye:
            cv2.circle(img, right_eye, 5, (250, 0, 250), 5)

        cv2.imshow("Face detection (press q to quit)", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
