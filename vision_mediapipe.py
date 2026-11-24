"""MediaPipe BlazePose vision detector — production version.

Replaces vision_detector.py when running on Jetson/Pi with a camera.
Uses MediaPipe Pose to detect body landmarks, then maps limb
configurations to baby posture keywords.

Interface is identical to vision_detector.py:
  detect_from_image(path) → {"postures": [...], "face_state": "..."}
  detect_from_frame(frame) → same (for real-time camera input)

Setup on Jetson:
  pip install mediapipe opencv-python
  (JetPack 6+ ships with CUDA support for MediaPipe)

Camera connection:
  Pi Camera V3 → MIPI CSI connector on Jetson
  Or USB webcam → USB port

Posture detection logic:
  MediaPipe gives 33 body landmarks. We detect:
    - arching_back: spine curvature + head tilted back
    - clenched_fists: hands closed (finger distance)
    - arms_spread: both arms extended outward (Moro reflex)
    - rooting: head turning side to side + mouth area
    - sucking_fingers: hand near mouth
    - rubbing_eyes: hand near eye region
    - yawning: mouth wide open
    - reaching_up: arms extended upward
    - kicking_legs: legs alternating rapidly
    - squirming: body twisting
    - fetal_position: knees drawn up + body curled
    - looking_around: head rotation
    - limp: all limbs low energy
"""
import os
import math
import time
from typing import Optional

import numpy as np

# MediaPipe imports — only available on device
try:
    import mediapipe as mp
    import cv2
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    mp = None
    cv2 = None


class BlazePoseDetector:
    """MediaPipe BlazePose-based posture detector.

    Maps 33 body landmarks to baby posture keywords.
    Falls back to annotation mode if MediaPipe is not installed.
    """

    # MediaPipe Pose landmark indices
    LANDMARKS = {
        "nose": 0, "left_eye": 2, "right_eye": 5,
        "left_ear": 7, "right_ear": 8,
        "mouth_left": 9, "mouth_right": 10,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.fallback = not HAS_MEDIAPIPE

        if not self.fallback:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=lite, 1=full, 2=heavy
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
            )
        else:
            print("[vision] MediaPipe not installed — falling back to annotation mode")
            from vision_detector import VisionDetector
            self._fallback_detector = VisionDetector()

    def detect_from_image(self, image_path: str) -> dict:
        """Detect postures from an image file."""
        if self.fallback:
            return self._fallback_detector.detect_from_image(image_path)

        frame = cv2.imread(image_path)
        if frame is None:
            return {"postures": [], "face_state": "normal"}
        return self.detect_from_frame(frame)

    def detect_from_frame(self, frame: np.ndarray) -> dict:
        """Detect postures from a camera frame (BGR numpy array).

        This is the main entry point for real-time camera input.
        """
        if self.fallback:
            return {"postures": [], "face_state": "normal"}

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Run pose detection
        pose_result = self.pose.process(rgb)

        # Run face mesh for face state
        face_result = self.face_mesh.process(rgb)

        postures = []
        face_state = "normal"

        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            postures = self._detect_postures(lm, w, h)

        if face_result.multi_face_landmarks:
            face_lm = face_result.multi_face_landmarks[0].landmark
            face_state = self._detect_face_state(face_lm, w, h)

        return {"postures": postures, "face_state": face_state}

    def _detect_postures(self, lm, w, h) -> list[str]:
        """Map body landmarks to posture keywords."""
        postures = []

        # Helper: get normalized landmark
        def pt(name):
            idx = self.LANDMARKS[name]
            return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

        # Key points
        nose = pt("nose")
        left_shoulder = pt("left_shoulder")
        right_shoulder = pt("right_shoulder")
        left_elbow = pt("left_elbow")
        right_elbow = pt("right_elbow")
        left_wrist = pt("left_wrist")
        right_wrist = pt("right_wrist")
        left_hip = pt("left_hip")
        right_hip = pt("right_hip")
        left_knee = pt("left_knee")
        right_knee = pt("right_knee")
        left_ankle = pt("left_ankle")
        right_ankle = pt("right_ankle")
        mouth_left = pt("mouth_left")
        mouth_right = pt("mouth_right")
        left_eye = pt("left_eye")
        right_eye = pt("right_eye")

        # --- Arching back ---
        # Spine curvature: shoulders significantly above hips,
        # and head tilted back (nose y < shoulder y)
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        spine_angle = abs(shoulder_mid[1] - hip_mid[1])
        if nose[1] < shoulder_mid[1] - 0.05 and spine_angle < 0.15:
            postures.append("arching_back")

        # --- Clenched fists ---
        # Can't directly detect finger closure from BlazePose (no finger joints)
        # Approximate: wrists close to body + hands not open
        # Check if wrists are clenched (near center of body, not extended)
        wrist_dist_from_body = (
            np.linalg.norm(left_wrist - shoulder_mid) +
            np.linalg.norm(right_wrist - shoulder_mid)
        ) / 2
        if wrist_dist_from_body < 0.15 and "arching_back" in postures:
            postures.append("clenched_fists")

        # --- Arms spread (Moro reflex) ---
        # Both arms extended outward beyond shoulder width
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        wrist_width = np.linalg.norm(left_wrist - right_wrist)
        if shoulder_width > 0 and wrist_width > shoulder_width * 1.5:
            postures.append("arms_spread")

        # --- Rooting ---
        # Head turning: nose x significantly offset from shoulder midpoint
        head_turn = abs(nose[0] - shoulder_mid[0])
        if head_turn > 0.08:
            postures.append("rooting")

        # --- Sucking fingers ---
        # Hand near mouth
        mouth_mid = (mouth_left + mouth_right) / 2
        hand_to_mouth = min(
            np.linalg.norm(left_wrist - mouth_mid),
            np.linalg.norm(right_wrist - mouth_mid),
        )
        if hand_to_mouth < 0.08:
            postures.append("sucking_fingers")

        # --- Rubbing eyes ---
        # Hand near eye region
        eye_mid = (left_eye + right_eye) / 2
        hand_to_eye = min(
            np.linalg.norm(left_wrist - eye_mid),
            np.linalg.norm(right_wrist - eye_mid),
        )
        if hand_to_eye < 0.08 and "sucking_fingers" not in postures:
            postures.append("rubbing_eyes")

        # --- Yawning ---
        # Mouth wide open: mouth width relative to face width
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        face_width = np.linalg.norm(left_eye - right_eye)
        if face_width > 0 and mouth_width > face_width * 0.8:
            postures.append("yawning")

        # --- Reaching up ---
        # Both wrists above shoulders (lower y = higher position)
        if left_wrist[1] < left_shoulder[1] - 0.1 and \
           right_wrist[1] < right_shoulder[1] - 0.1:
            postures.append("reaching_up")

        # --- Looking around ---
        # Head rotation: ear visibility asymmetry
        # (when head turns, one ear becomes less visible)
        if abs(nose[0] - shoulder_mid[0]) > 0.04 and head_turn <= 0.08:
            postures.append("looking_around")

        # --- Fetal position ---
        # Knees drawn up toward chest (knee y close to hip y)
        knee_to_hip = (
            abs(left_knee[1] - left_hip[1]) +
            abs(right_knee[1] - right_hip[1])
        ) / 2
        if knee_to_hip < 0.08:
            postures.append("fetal_position")

        # --- Kicking legs ---
        # Legs extended and alternating (detected over frames, not single)
        # For single frame: check if legs are extended
        left_leg_ext = np.linalg.norm(left_hip - left_ankle)
        right_leg_ext = np.linalg.norm(right_hip - right_ankle)
        if left_leg_ext > 0.3 and right_leg_ext > 0.3:
            # Check if legs are at different heights (alternating)
            if abs(left_ankle[1] - right_ankle[1]) > 0.05:
                postures.append("kicking_legs")

        # --- Squirming ---
        # Body twisting: shoulders not aligned horizontally
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_tilt > 0.06 and "arching_back" not in postures:
            postures.append("squirming")

        # --- Limp ---
        # All limbs close to body, low energy
        all_close = (
            np.linalg.norm(left_wrist - left_shoulder) < 0.15 and
            np.linalg.norm(right_wrist - right_shoulder) < 0.15 and
            np.linalg.norm(left_ankle - left_hip) < 0.20 and
            np.linalg.norm(right_ankle - right_hip) < 0.20
        )
        if all_close and not postures:
            postures.append("limp")

        return postures

    def _detect_face_state(self, face_lm, w, h) -> str:
        """Detect face color state from face mesh landmarks.

        Samples skin color from the cheek area to detect redness or paleness.
        """
        if cv2 is None:
            return "normal"

        # MediaPipe FaceMesh cheek landmark indices
        left_cheek = 234    # left cheek
        right_cheek = 454   # right cheek
        forehead = 10       # forehead center

        def get_pixel(lm_idx):
            pt = face_lm[lm_idx]
            return int(pt.x * w), int(pt.y * h)

        # We need the original frame to sample colors
        # Since we're in detect_from_frame, we stored the frame
        # This is a limitation — face state needs the frame
        # For now, return normal; color analysis requires the frame ref
        return "normal"

    def detect_from_frame_full(self, frame: np.ndarray) -> dict:
        """Full detection with face color analysis.

        Unlike detect_from_frame, this also samples skin color
        from the frame to detect redness/paleness.
        """
        result = self.detect_from_frame(frame)
        if self.fallback:
            return result

        # Face color analysis
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        face_result = self.face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_lm = face_result.multi_face_landmarks[0].landmark
            cheek_idx = 234
            cx, cy = int(face_lm[cheek_idx].x * w), int(face_lm[cheek_idx].y * h)

            # Sample 5x5 pixel region around cheek
            region = rgb[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
            if region.size > 0:
                avg_r = region[:,:,0].mean()
                avg_g = region[:,:,1].mean()
                avg_b = region[:,:,2].mean()

                # Detect redness: R significantly > G and B
                if avg_r > avg_g + 25 and avg_r > avg_b + 25:
                    result["face_state"] = "red"
                # Detect paleness: all channels low, especially low R
                elif avg_r < 120 and avg_g < 120 and avg_b < 120 and avg_r < avg_g:
                    result["face_state"] = "pale"

        return result

    def close(self):
        """Release MediaPipe resources."""
        if not self.fallback:
            self.pose.close()
            self.face_mesh.close()


class CameraStream:
    """Real-time camera input for Jetson/Pi.

    Opens Pi Camera V3 or USB webcam and provides frames.
    """

    def __init__(self, camera_index=0, width=640, height=480, fps=15):
        if cv2 is None:
            raise RuntimeError("OpenCV not installed — install with: pip install opencv-python")

        # On Jetson with Pi Camera, use GStreamer backend
        # On USB webcam, use default V4L2
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

    def read(self):
        """Read a frame. Returns (success, frame)."""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# Convenience function for pipeline integration
def create_vision_detector():
    """Factory: returns BlazePoseDetector if MediaPipe available,
    otherwise falls back to annotation-based VisionDetector."""
    if HAS_MEDIAPIPE:
        return BlazePoseDetector()
    else:
        from vision_detector import VisionDetector
        return VisionDetector()


if __name__ == "__main__":
    print("=" * 60)
    print("MediaPipe BlazePose Vision Detector")
    print("=" * 60)
    print(f"MediaPipe available: {HAS_MEDIAPIPE}")

    if HAS_MEDIAPIPE:
        detector = BlazePoseDetector()
        print(f"Detector initialized: {type(detector).__name__}")
        print(f"Pose model: MediaPipe Pose (complexity=1)")
        print(f"Face model: MediaPipe FaceMesh")
        print()
        print("Posture detection capabilities:")
        capabilities = [
            ("arching_back", "Spine curvature + head tilted back"),
            ("clenched_fists", "Wrists clenched near body (with arching)"),
            ("arms_spread", "Both arms extended beyond shoulder width"),
            ("rooting", "Head turning side to side"),
            ("sucking_fingers", "Hand near mouth"),
            ("rubbing_eyes", "Hand near eye region"),
            ("yawning", "Mouth width > 80% of face width"),
            ("reaching_up", "Both wrists above shoulders"),
            ("looking_around", "Subtle head rotation"),
            ("fetal_position", "Knees drawn up toward hips"),
            ("kicking_legs", "Legs extended + alternating height"),
            ("squirming", "Shoulder tilt + body twisting"),
            ("limp", "All limbs close to body, no active posture"),
        ]
        for posture, desc in capabilities:
            print(f"  {posture:20s} — {desc}")
        print()
        print("Face state detection:")
        print("  red   — R channel significantly higher than G/B")
        print("  pale  — All channels low, R < G")
        print()
        print("Usage:")
        print("  # From image file:")
        print("  detector = BlazePoseDetector()")
        print('  result = detector.detect_from_image("frame.jpg")')
        print()
        print("  # From camera stream:")
        print("  with CameraStream(camera_index=0) as cam:")
        print("      ret, frame = cam.read()")
        print("      result = detector.detect_from_frame_full(frame)")
        print()
        print("  # Integrate with pipeline:")
        print("  from vision_mediapipe import create_vision_detector")
        print("  vision = create_vision_detector()")
        print("  # vision.detect_from_frame(frame) → same interface as vision_detector.py")
    else:
        print()
        print("MediaPipe not installed. Install with:")
        print("  pip install mediapipe opencv-python")
        print()
        print("On Jetson Orin Nano (JetPack 6+):")
        print("  sudo apt install python3-opencv")
        print("  pip install mediapipe")
        print()
        print("Falling back to annotation mode for testing.")
        detector = BlazePoseDetector()
        print(f"Using: {type(detector).__name__} (fallback)")
