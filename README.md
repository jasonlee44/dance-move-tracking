# Dance Move Tracking

A desktop app that takes a video of dance choreography, runs pose estimation on it, and breaks the motion into steps with timestamps and keyframes so someone can learn the dance.

---

## What it does

- **Input:** A single video file (common formats, e.g. MP4, MOV; target up to ~1 minute).
- **Output:**
  - **First-frame mode (default):** Pose for the first frame only — a list of body landmarks (subset of 33) with normalized coordinates and confidence.
  - **Full-video mode (`--full`):** Pose on every frame, then segment into steps when pose change exceeds a threshold. Outputs a table of steps with start/end times and keyframe frame indices, plus the first frame’s pose.

**Pipeline:** Video → pose estimation (MediaPipe) → group poses into moves by change threshold → steps + keyframes.

---

## Project structure

```
dance-move-tracking/
├── app/
│   └── main.py          # Entry point: pose on first frame or full video, move segmentation
├── data/                 # Put your input videos here (or pass any path)
├── output/               # Pose model cache (pose_landmarker_lite.task)
├── requirements.txt
├── check_env.py          # Verify PyTorch + MediaPipe + deps
├── env_imports.py        # Reference imports for PyTorch projects
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+ (3.10–3.12 recommended).

### 1. Clone and go to the project

```bash
cd dance-move-tracking
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# Windows:  .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, OpenCV, MediaPipe, and other libs. On **Apple Silicon**, PyTorch can use MPS if available. For **CUDA** on Linux/Windows, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt`.

### 4. (Optional) Verify the environment

```bash
python check_env.py
```

Checks that PyTorch, MediaPipe, and other packages import correctly and reports CPU/CUDA/MPS.

---

## Running the app

From the project root with the virtual environment activated:

```bash
python -m app.main <path_to_video>           # First frame only
python -m app.main <path_to_video> --full   # Full video: pose every frame, then segment into steps
```

**Examples:**

```bash
python -m app.main data/my_dance.MOV
python -m app.main data/my_dance.MOV --full
python -m app.main data/my_dance.MOV --full --threshold 0.05   # Fewer steps (higher threshold)
python -m app.main data/my_dance.MOV --full --stride 2        # Every 2nd frame (faster)
```

**Options:**


| Option           | Description                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `--full`         | Run pose on the entire video and output a step breakdown (timestamps + keyframe indices).           |
| `--stride N`     | With `--full`, process every Nth frame (default: 1). Use 2 or 3 to speed up.                        |
| `--max-frames N` | With `--full`, stop after N frames (for quick tests).                                               |
| `--threshold T`  | With `--full`, pose-change threshold for starting a new step (default: 0.03). Higher = fewer steps. |


**First run:** The pose model (`pose_landmarker_lite.task`) is downloaded once into `output/` (requires internet and may take a moment).

**Output:**

- **Without `--full`:** First frame dimensions and the pose keypoints (subset of 33 landmarks) with normalized (x, y, z) and confidence.
- **With `--full`:** Number of frames processed, a table of steps (Step, Start(s), End(s), Keyframe), then the first frame’s full pose.

---

## Tech stack

- **Pose estimation:** MediaPipe Tasks (`mediapipe.tasks.vision.PoseLandmarker`), CPU delegate by default. Uses a configurable subset of landmarks (e.g. nose, shoulders, elbows, wrists, hips, knees, ankles, feet) for dance-relevant keypoints.
- **Video I/O:** OpenCV.
- **Move segmentation:** Rule-based grouping — a new step starts when mean pose change (weighted by visibility) between consecutive frames exceeds a threshold.

---

## Troubleshooting

- **SSL / certificate errors when downloading the pose model:** Ensure `certifi` is installed (`pip install certifi`). On macOS, you may need to run the “Install Certificates” command for your Python installer.
- **“No pose detected”:** Try a frame where the full body is visible and well lit.
- **GPU / OpenGL errors (e.g. NSOpenGLPixelFormat):** The app uses the CPU delegate for pose; if you still see GPU-related errors, try running in a normal terminal (not headless) or check your MediaPipe version.

