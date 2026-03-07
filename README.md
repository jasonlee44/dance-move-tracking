# Dance Move Tracking

A desktop-oriented project that takes a video of dance choreography, runs pose estimation on it, and (in progress) breaks the motion into steps with timestamps and keyframes so someone can learn the dance.

**Current scope:** Load a video, analyze the first frame with MediaPipe Pose, and print the detected body keypoints and confidence scores. Future work: segment the full video into moves and output a step-by-step breakdown.

---

## What it does

- **Input:** A single video file (common formats, e.g. MP4, MOV; target up to ~1 minute).
- **Output (current):** For the first frame, a list of 33 body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.) with normalized coordinates and confidence.
- **Planned output:** A list of steps with timestamps and keyframes for teaching the choreography.

**Pipeline:** Video → pose estimation (MediaPipe) → (planned) group poses into moves by change threshold → steps + keyframes.

---

## Project structure

```
dance-move-tracking/
├── app/
│   └── main.py          # Entry point: load video, run pose on first frame, print pose
├── data/                 # Put your input videos here (or pass any path)
├── output/               # Generated files (pose model cache, future step breakdowns)
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
python -m app.main <path_to_video>
```

**Examples:**

```bash
python -m app.main data/my_dance.MOV
python -m app.main /path/to/video.mp4
```

**First run:** The pose model (`pose_landmarker_lite.task`) is downloaded once into `output/` (requires internet and may take a moment).

**Output:** The script prints the first frame’s dimensions, then the 33 body keypoints with normalized (x, y, z) and confidence (0–1) for each landmark.

---

## Tech stack

- **Pose estimation:** MediaPipe Tasks (`mediapipe.tasks.vision.PoseLandmarker`), CPU delegate by default.
- **Video I/O:** OpenCV.
- **Future:** PyTorch for any learned move-grouping or sequence models; current pipeline uses the above plus rule-based grouping (planned).

---

## Troubleshooting

- **SSL / certificate errors when downloading the pose model:** Ensure `certifi` is installed (`pip install certifi`). On macOS, you may need to run the “Install Certificates” command for your Python installer.
- **“No pose detected”:** Try a frame where the full body is visible and well lit.
- **GPU / OpenGL errors (e.g. NSOpenGLPixelFormat):** The app uses the CPU delegate for pose; if you still see GPU-related errors, try running in a normal terminal (not headless) or check your MediaPipe version.
