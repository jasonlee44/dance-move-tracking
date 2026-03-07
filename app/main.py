"""
Main entry point: load a video, analyze one frame, output detected pose.
Usage: python -m app.main <path_to_video>

Uses MediaPipe Tasks (PoseLandmarker). The pose model is downloaded to output/
on first run if not present.
"""
import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# MediaPipe Pose 33 landmark names (index 0–32)
POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
    "float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"


def get_pose_model_path() -> Path:
    """Return path to pose model, downloading to output/ if needed."""
    # Use project root: parent of app/
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "output" / POSE_MODEL_FILENAME
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {model_path} ...")
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(POSE_MODEL_URL, context=ssl_ctx) as resp:
        model_path.write_bytes(resp.read())
    print("Done.")
    return model_path


def load_frame(video_path: str, frame_index: int = 0) -> tuple[bool, "np.ndarray | None"]:
    """Load a single frame from the video. frame_index 0 = first frame."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def detect_pose(frame: "np.ndarray") -> list[dict] | None:
    """
    Run MediaPipe Pose (Tasks API) on one frame (BGR).
    Returns list of {name, x, y, z, visibility} or None if no pose detected.
    """
    model_path = get_pose_model_path()
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode= vision.RunningMode.IMAGE,
    )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_contiguous = np.ascontiguousarray(rgb)
    mp_img = mp.Image(
        image_format= mp.ImageFormat.SRGB,
        data=rgb_contiguous,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
        return None
    landmarks_list = result.pose_landmarks[0]
    out = []
    for i, lm in enumerate(landmarks_list):
        name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"landmark_{i}"
        out.append({
            "name": name,
            "x": lm.x if lm.x is not None else 0.0,
            "y": lm.y if lm.y is not None else 0.0,
            "z": lm.z if lm.z is not None else 0.0,
            "visibility": lm.visibility if lm.visibility is not None else 1.0,
        })
    return out


def format_pose(landmarks: list[dict]) -> str:
    """Format pose landmarks for console output: keypoint set, then per-point coords and confidence."""
    lines = [
        "Pose detected (MediaPipe Pose, 33 body keypoints)",
        "",
        "--- Keypoints (body landmarks) ---",
        "  Index  Landmark name",
        "  -----  ------------",
    ]
    for i, lm in enumerate(landmarks):
        lines.append(f"  {i:5}  {lm['name']}")
    lines.extend([
        "",
        "--- Per keypoint: x, y, z (normalized 0–1, origin top-left), confidence (0–1) ---",
        "  Landmark              x        y        z        confidence",
        "  -------------------  -------  -------  -------  ----------",
    ])
    for lm in landmarks:
        conf = lm.get("visibility", 1.0)
        lines.append(
            f"  {lm['name']:20} {lm['x']:.4f}   {lm['y']:.4f}   {lm['z']:.4f}   {conf:.4f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load video, analyze the first frame only, output pose.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    args = parser.parse_args()

    print(f"Loading video: {args.video_path}")
    ok, frame = load_frame(args.video_path, frame_index=0)
    if not ok or frame is None:
        print("Failed to read first frame.", file=sys.stderr)
        return 1

    print("Analyzing first frame only.")
    print(f"Frame shape: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    print("Running pose estimation...")
    landmarks = detect_pose(frame)
    if landmarks is None:
        print("No pose detected in this frame.")
        return 0

    print(format_pose(landmarks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
