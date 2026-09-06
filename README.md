# EgoHands Hand Detection

Real-time **hand bounding-box detection** from a webcam, with a detector fine-tuned on the
[EgoHands dataset](https://vision.soic.indiana.edu/projects/egohands/) (48 videos, 4,800
annotated frames).

OpenCV handles the camera, frames, drawing and display. The model handles finding the hand,
predicting its box, and scoring the detection.

<table>
<tr><td><b>Sample run of demo1.py</b></td></tr>
<tr><td><img src="DEMO1output.png" alt="demo1 output" style="width: 320px;"/></td></tr>
</table>

## Accuracy

Measured on 1,200 held-out frames from 12 videos the model never saw
(`evaluate_detector.py`):

| Metric | Value |
| --- | --- |
| **AP@0.50** | **0.9335** |
| AP@0.75 | 0.7447 |
| AP@[0.50:0.95] (COCO primary) | 0.6297 |
| Best F1 | 0.903 @ score 0.20 |
| Frames with every hand found, nothing spurious | 46.5% |

Recall by hand size: **large 0.992**, medium 0.833, **small 0.152**. Small distant hands are
the known weakness — at 320×320 input a hand under 32 px is below what the coarsest feature
map can resolve.

## Setup

PyTorch has no wheels for Python 3.14, and MediaPipe pins `numpy<2`, so build the
environment against Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then download the dataset. IU removed the original file, but the Internet Archive has the
genuine 1.33 GB archive:

```bash
curl -L -C - -o egohands_data.zip \
  "https://web.archive.org/web/20200713164330id_/http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
unzip -n egohands_data.zip -d .
```

That produces `_LABELLED_SAMPLES/` beside the code — 48 folders of 100 frames each.
`metadata.mat` (the annotations) is already in this repo.

MediaPipe additionally needs its model file:

```bash
mkdir -p models && curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

On macOS, grant camera access to your terminal in
**System Settings → Privacy & Security → Camera**, then restart it.

## Run the live demos

```bash
.venv/bin/python webcam_detect.py       # SSDlite fine-tuned on EgoHands
.venv/bin/python webcam_detect_mp.py    # MediaPipe HandLandmarker
.venv/bin/python webcam_detect_cv.py    # classical skin + contours, no model
```

Press `q` or `Esc` to quit. Useful flags: `--threshold`, `--camera`, `--no-mirror`,
`--landmarks` (MediaPipe), `--show-mask` (classical). Each also takes `--image FILE` to run
on a still instead of the camera.

**Which one to use.** `webcam_detect_mp.py` is better for a laptop webcam — hands facing the
camera, and it labels left vs right. `webcam_detect.py` is better on first-person footage,
where hands enter cropped at the frame edge; measured head to head on EgoHands validation
frames, it reaches 0.761 recall against MediaPipe's 0.443. `webcam_detect_cv.py` needs no
model at all but detects *skin-coloured blobs*, so it will happily box your face and sleeves.

## The full pipeline

```
WEBCAM -> OpenCV captures each frame
   -> 1. DETECT     where is the hand?        box + confidence
   -> 2. SEGMENT    which pixels are hand?    mask
   -> 3. POSE       21 landmarks              fingertip coordinates
   -> 4. TRACK      frame to frame            id, trajectory, velocity
   -> gesture / movement
```

```bash
.venv/bin/python run_pipeline.py                 # live
.venv/bin/python run_pipeline.py --record s.mp4  # save the session
.venv/bin/python run_pipeline.py --video s.mp4   # replay it, reproducibly
```

Press `q` or `Esc` to quit. Runs at ~29 fps end-to-end, ~50 fps for the stages alone
on an M1 Pro.

Stages 1 and 3 share a single MediaPipe runner. HandLandmarker returns the box and the
21 landmarks from one inference, so running it twice to keep the stages separate would
halve the frame rate for nothing: 86.3 ms -> 42.9 ms per frame. The stages stay
independently swappable; the cost is paid once.

### Measured accuracy

Scored against EgoHands ground truth, not just checked for crashes
(`pytest tests/ -m accuracy`):

| Stage | Against | Result |
| --- | --- | --- |
| 1 Detect | annotated boxes | P 0.930, R 0.435, mean IoU 0.758 |
| 2 Segment | annotated polygons | mask IoU mean 0.750, median 0.771 |
| 3 Pose | anatomical consistency | 42 poses, all valid |
| 4 Track | known synthetic motion | 300.0 px/s at both 15 and 60 fps |
| 4 Track | two hands crossing | no id swap |

Stage 1's recall reflects MediaPipe on *egocentric* footage, which is not its domain;
precision 0.930 is the number that matters. On a webcam it is far stronger.

## Google Colab

Open `hand_pipeline_colab.ipynb` and upload `hand_pipeline_colab.zip` when it asks.

Colab runs on a remote VM with no camera and no display, so `cv2.VideoCapture(0)` and
`cv2.imshow` both fail there. The notebook bridges to your browser's camera with
JavaScript `getUserMedia`, uses `cv2_imshow`, and re-encodes output video to H.264
(OpenCV's mp4v will not play in a browser).

## Choosing a detector

`webcam_detect_mp.py` (MediaPipe) for a webcam; `webcam_detect.py` (SSDlite trained
here) for first-person footage.

The reason is worth stating plainly. EgoHands is head-mounted footage in which the only
skin-coloured objects are hands, so a detector trained on it alone never learns "hand vs
other body part" -- it learns "skin-coloured blob = hand". That scores AP@0.50 = 0.9335
on EgoHands and still boxes a webcam user's face, elbow and feet with high confidence.
Training with COCO-Hand, HaGRID and generic hand-free negatives raised precision to
0.978 but has not been confirmed to fix it. `run_pipeline.py` defaults to MediaPipe,
which does not have this failure mode.

## Train and evaluate

```bash
.venv/bin/python train_detector.py                     # all 48 videos, ~17 min on Apple Silicon
.venv/bin/python train_detector.py --limit-videos 4 --epochs 2   # quick smoke test
.venv/bin/python train_detector.py --eval-only         # score the current checkpoint

.venv/bin/python evaluate_detector.py                  # full accuracy report
.venv/bin/python evaluate_detector.py --render 12      # + visual prediction vs truth grid
```

Training splits **by video, never by frame** — the 100 frames inside one clip are near
duplicates, so a frame-level split would leak validation data into training. It checkpoints
on best validation F1 rather than the last epoch.

## Tests

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q            # everything
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -m e2e -v -s  # pipeline only, ~30s
```

`tests/test_e2e_pipeline.py` runs the whole journey — dataset → training → checkpoint →
evaluation → inference → camera — and asserts the model *actually learns* by requiring a
freshly trained model to beat its own random-initialised baseline. It also gates AP@0.50
at 0.85, so an accuracy regression fails the suite.

## Querying the dataset

`get_meta_by()` returns video metadata as a pandas DataFrame:

```python
from get_meta_by import get_meta_by

get_meta_by()                                            # all 48 videos
get_meta_by('Location', 'COURTYARD')                     # 16 videos
get_meta_by('Activity', 'PUZZLE', 'Viewer', 'B, S')      # filters in any order
```

Filters: `Location` (OFFICE, COURTYARD, LIVINGROOM), `Activity` (CHESS, JENGA, PUZZLE,
CARDS), `Viewer` and `Partner` (B, S, T, H). Values accept comma-separated lists with or
without spaces. An unrecognised filter name raises `ValueError` rather than being ignored.

Per-frame accessors: `get_frame_path`, `get_bounding_boxes` (4×4 `[x, y, w, h]`, zero rows
for absent hands) and `get_segmentation_mask`. `demo1.py` shows all three.

## Files

| File | Purpose |
| --- | --- |
| `detection_dataset.py` | Lazy detection dataset over all 48 videos; video-level split |
| `train_detector.py` | Fine-tunes `ssdlite320_mobilenet_v3_large`; P/R/F1 eval |
| `evaluate_detector.py` | AP, threshold sweeps, per-slice breakdowns, failure analysis |
| `webcam_detect.py` | Live demo — trained SSDlite |
| `webcam_detect_mp.py` | Live demo — MediaPipe |
| `webcam_detect_cv.py` | Live demo — classical OpenCV, no model |
| `get_meta_by.py`, `get_frame_path.py`, `get_bounding_boxes.py`, `get_segmentation_mask.py` | Dataset accessors, ported from the original MATLAB toolkit |
| `demo1.py` | Shows a frame with its mask and bounding boxes |
| `tests/` | 252 tests, including the end-to-end pipeline |

## Maintainer

- **shivansh.s@utexas.edu**

## Credit

All videos and raw label data (the EgoHands dataset itself) come from:

```
@InProceedings{Bambach_2015_ICCV,
author = {Bambach, Sven and Lee, Stefan and Crandall, David J. and Yu, Chen},
title = {Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
```

The dataset accessors are a Python port of Indiana University's original MATLAB toolkit.
Detection uses `ssdlite320_mobilenet_v3_large` from torchvision (BSD-3-Clause) and,
optionally, MediaPipe HandLandmarker from Google (Apache-2.0).
