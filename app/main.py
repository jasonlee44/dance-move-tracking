"""
Main entry point: load a video, run pose estimation, output detected pose(s).
Usage:
  python -m app.main <path_to_video>           # first frame only
  python -m app.main <path_to_video> --full   # entire video (good for ~5s clips)

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

# Subset of landmark names to keep (edit this to match what you need for dance moves).
# Only these will be returned from detect_pose() and used downstream.
# Use POSE_LANDMARK_NAMES above as reference; set to None to keep all 33.
LANDMARKS_TO_KEEP = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
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


def _result_to_landmarks(landmarks_list) -> list[dict]:
    """Convert MediaPipe pose_landmarks[0] to list of {name, x, y, z, visibility} (filtered by LANDMARKS_TO_KEEP)."""
    out = []
    keep_set = set(LANDMARKS_TO_KEEP) if LANDMARKS_TO_KEEP else None
    for i, lm in enumerate(landmarks_list):
        name = POSE_LANDMARK_NAMES[i] if i < len(POSE_LANDMARK_NAMES) else f"landmark_{i}"
        if keep_set is not None and name not in keep_set:
            continue
        out.append({
            "name": name,
            "x": lm.x if lm.x is not None else 0.0,
            "y": lm.y if lm.y is not None else 0.0,
            "z": lm.z if lm.z is not None else 0.0,
            "visibility": lm.visibility if lm.visibility is not None else 1.0,
        })
    return out


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
    return _result_to_landmarks(result.pose_landmarks[0])


def format_pose(landmarks: list[dict]) -> str:
    """Format pose landmarks for console output: keypoint set, then per-point coords and confidence."""
    n = len(landmarks)
    lines = [
        f"Pose detected (MediaPipe Pose, {n} keypoints)",
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


def extract_poses_from_video(
    video_path: str,
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[list[tuple[float, int, list[dict]]], float]:
    """
    Run pose estimation on every frame (or every stride-th frame) of the video.
    Returns (list of (timestamp_sec, frame_index, landmarks), fps).
    Uses a single PoseLandmarker for the whole video for efficiency.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    model_path = get_pose_model_path()
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
    )
    poses: list[tuple[float, int, list[dict]]] = []
    n_processed = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_index in range(0, total_frames, stride):
            if max_frames is not None and n_processed >= max_frames:
                break
            ok, frame = load_frame(video_path, frame_index)
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            result = landmarker.detect(mp_img)
            if not result.pose_landmarks:
                continue
            landmarks = _result_to_landmarks(result.pose_landmarks[0])
            t = frame_index / fps
            poses.append((t, frame_index, landmarks))
            n_processed += 1
    return poses, fps


def pose_change(a: list[dict], b: list[dict]) -> float:
    """
    Compute how much the pose changed between two frames (same landmark set).
    Returns mean Euclidean distance in normalized (x, y) space.
    Uses visibility as weight: low-visibility landmarks contribute less.
    """
    if not a or not b or len(a) != len(b):
        return float("inf")
    total = 0.0
    weight_sum = 0.0
    for la, lb in zip(a, b):
        if la["name"] != lb["name"]:
            return float("inf")
        w = (la.get("visibility", 1.0) + lb.get("visibility", 1.0)) / 2.0
        dx = la["x"] - lb["x"]
        dy = la["y"] - lb["y"]
        total += w * (dx * dx + dy * dy) ** 0.5
        weight_sum += w
    return total / weight_sum if weight_sum > 0 else 0.0


def segment_into_steps(
    poses: list[tuple[float, int, list[dict]]],
    change_threshold: float = 0.03,
) -> list[dict]:
    """
    Group pose sequence into steps: start a new step when pose change vs previous
    frame exceeds change_threshold (in normalized coords). Tune threshold by eye.
    Returns list of {step, start_time, end_time, start_frame, end_frame, keyframe_frame}.
    """
    if not poses:
        return []
    steps: list[dict] = []
    step_start_t, step_start_f, step_start_landmarks = poses[0][0], poses[0][1], poses[0][2]
    for i in range(1, len(poses)):
        t, frame_idx, landmarks = poses[i]
        change = pose_change(step_start_landmarks, landmarks)
        if change >= change_threshold:
            steps.append({
                "step": len(steps) + 1,
                "start_time": step_start_t,
                "end_time": poses[i - 1][0],
                "start_frame": step_start_f,
                "end_frame": poses[i - 1][1],
                "keyframe_frame": step_start_f,
            })
            step_start_t, step_start_f, step_start_landmarks = t, frame_idx, landmarks
    # last step
    steps.append({
        "step": len(steps) + 1,
        "start_time": step_start_t,
        "end_time": poses[-1][0],
        "start_frame": step_start_f,
        "end_frame": poses[-1][1],
        "keyframe_frame": step_start_f,
    })
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load video, run pose estimation (first frame or full video)."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run pose on entire video (default: first frame only). Good for short clips (e.g. ~5s).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame when using --full (default: 1). Use 2 or 3 to speed up.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="When using --full, stop after N frames (optional, for quick tests).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        metavar="T",
        help="Pose change threshold for starting a new step when using --full (default: 0.03). Increase for fewer steps.",
    )
    args = parser.parse_args()

    print(f"Loading video: {args.video_path}")
    if not args.full:
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

    # Full video mode
    print("Running pose on entire video (this may take a moment)...")
    poses, fps = extract_poses_from_video(
        args.video_path,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if not poses:
        print("No poses detected in any frame.")
        return 0
    duration_sec = poses[-1][0] + (1.0 / fps) if poses else 0.0
    print(f"Processed {len(poses)} frames over {duration_sec:.2f}s (fps={fps:.1f})")
    steps = segment_into_steps(poses, change_threshold=args.threshold)
    print()
    print("Steps (move segmentation by pose-change threshold):")
    print("  Step   Start(s)   End(s)   Keyframe")
    print("  ----   --------   ------   -------")
    for s in steps:
        print(f"  {s['step']:4}   {s['start_time']:8.2f}   {s['end_time']:6.2f}   {s['keyframe_frame']}")
    print()
    print("First frame pose:")
    print(format_pose(poses[0][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
