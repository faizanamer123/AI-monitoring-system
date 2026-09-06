# Hand Detection, Segmentation, Pose & Tracking

Real-time hand analysis from a webcam. OpenCV handles the camera, frames and drawing;
the models handle finding hands and understanding them.

```
WEBCAM -> OpenCV captures each frame
   |
   +-> 1. DETECT     where is the hand?        box + confidence
   +-> 2. SEGMENT    which pixels are hand?    per-pixel mask
   +-> 3. POSE       21 landmarks              fingertip coordinates
   +-> 4. TRACK      frame to frame            stable id, trajectory, velocity
   |
   +-> gesture / movement / trajectory
```

<img src="DEMO1output.png" alt="EgoHands demo output" width="320"/>

---

## Table of contents

1. [What this does right now](#what-this-does-right-now)
2. [Quick start](#quick-start)
3. [Full setup](#full-setup)
4. [Commands](#commands)
5. [The four stages](#the-four-stages)
6. [Datasets](#datasets)
7. [Accuracy](#accuracy)
8. [Google Colab](#google-colab)
9. [Project layout](#project-layout)
10. [Testing](#testing)
11. [Known issues](#known-issues)
12. [Credits](#credits)

---

## What this does right now

Point it at your webcam and it will, per frame:

- **find every hand** and draw a box with a confidence score
- **segment the hand pixels** and paint them as a translucent overlay
- **place 21 landmarks** (wrist, and four joints per finger) and draw the skeleton
- **keep a stable id** for each hand across frames, with its trajectory and speed in px/s
- **name the gesture**: `open_palm`, `fist`, `point`, `peace`, `thumbs_up`, `pinch`

Measured on an Apple M1 Pro: **~29 fps end-to-end**, ~50 fps for the stages alone
(the rest is camera I/O, which caps at 15-30 fps depending on lighting).

There are **three interchangeable detectors**. Which one you want depends on the footage:

| Detector | Use it for | Speed | Needs |
| --- | --- | --- | --- |
| **MediaPipe** (default) | webcams, anything front-facing | 52 fps | 7.5 MB model file |
| **SSDlite** (trained here) | first-person / head-mounted footage | 26 fps | the committed checkpoint |
| **Classical CV** | nothing serious; a no-model baseline | 179 fps | nothing |

> **Use MediaPipe on a webcam.** The SSDlite model is genuinely strong on egocentric
> footage but boxes faces, elbows and feet on a webcam — see [Known issues](#known-issues).
> `run_pipeline.py` already defaults to MediaPipe.

---

## Quick start

You need Python 3.11, about 60 MB of downloads, and a webcam.

```bash
git clone https://github.com/faizanamer123/AI-monitoring-system.git
cd AI-monitoring-system

# 1. environment (Python 3.11 specifically -- see Full setup for why)
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. the MediaPipe model (7.5 MB)
mkdir -p models
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# 3. run it
.venv/bin/python run_pipeline.py
```

A window opens. Hold up a hand. Press **`q`** or **`Esc`** to quit.

**No dataset download is needed to run the demo.** You only need EgoHands if you want to
retrain or re-evaluate the SSDlite detector.

**macOS:** the first run asks for camera permission on behalf of your terminal. If no
prompt appears and frames come back black, grant it under
**System Settings → Privacy & Security → Camera**, then restart the terminal.

---

## Full setup

### Why Python 3.11 specifically

Two hard constraints, and they interlock:

- **PyTorch publishes no wheels for Python 3.12+** on this stack. If your default
  `python3` is 3.13 or 3.14, `pip install torch` simply fails.
- **MediaPipe 0.10.x pins `numpy<2`**, which in turn forces `opencv-contrib-python 4.11`
  rather than `opencv-python 5.x` (which requires numpy≥2).

`requirements.txt` pins the whole interlocked set. Read the comments in it before
upgrading anything — they explain what breaks.

```bash
# macOS with Homebrew
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Always invoke `.venv/bin/python`, never a bare `python3`.

### Two known traps

**Install `opencv-contrib-python`, never `opencv-python`.** The two distributions
overlay the same `cv2/` package directory, so installing both corrupts the import and
uninstalling one deletes files the other needs.

**`pip check` will report "mediapipe is not supported on this platform".** Ignore it.
That reads a stale platform tag, not the binary; the shipped `.so` files are Mach-O
universal (x86_64 + arm64) and an arm64 interpreter loads the arm64 slice natively.
Verified running at ~52 fps, no Rosetta.

### Optional: the EgoHands dataset

Only needed for training or evaluation.

```bash
# 1.3 GB. Indiana University removed the original file; the Internet Archive has it.
curl -L -C - -o egohands_data.zip \
  "https://web.archive.org/web/20200713164330id_/http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
unzip -n egohands_data.zip -d .
```

That produces `_LABELLED_SAMPLES/` — 48 folders of 100 frames each. `metadata.mat`
(the annotations) is already committed. The Archive can stall; `-C -` resumes, and you
may need several attempts.

---

## Commands

Run everything **from the repository root** — `get_frame_path()` resolves paths via
`os.getcwd()`.

### Live demos

```bash
.venv/bin/python run_pipeline.py          # all four stages (recommended)
.venv/bin/python webcam_detect_mp.py      # MediaPipe detection only
.venv/bin/python webcam_detect.py         # SSDlite detection only
.venv/bin/python webcam_detect_cv.py      # classical OpenCV, no model
```

Useful flags for `run_pipeline.py`:

| Flag | Effect |
| --- | --- |
| `--record out.mp4` | save the annotated session |
| `--video in.mp4` | replay a file instead of the camera (reproducible tracking) |
| `--detector ssdlite` | use the trained model instead of MediaPipe |
| `--segmentation landmark\|skin\|hybrid\|none` | choose the mask method (default `hybrid`) |
| `--show mask,skeleton,box,trail,label` | pick overlays |
| `--max-hands 2` | cap detections |
| `--no-mirror` | disable the selfie flip |
| `--headless 20` | run 20 s with no window, print stats (for CI) |

Every `webcam_detect*.py` also accepts `--image FILE` to run on a still.

### Evaluation

```bash
.venv/bin/python evaluate_detector.py               # AP, threshold sweep, breakdowns
.venv/bin/python evaluate_detector.py --render 12   # + prediction-vs-truth image grid
.venv/bin/python evaluate_detector.py --split train # check the generalisation gap
.venv/bin/python train_detector.py --eval-only      # quick P/R/F1 at one threshold
```

### Training

```bash
.venv/bin/python train_detector.py                       # EgoHands only, ~17 min
.venv/bin/python train_detector.py --external            # + COCO-Hand + HaGRID, ~50 min
.venv/bin/python train_detector.py --limit-videos 4 --epochs 2   # 2-min smoke test
```

Notable options:

| Flag | Meaning |
| --- | --- |
| `--external` | mix in COCO-Hand-S and HaGRID |
| `--generic-negatives N` | hand-free crops from those images (default 6000) |
| `--neg-ratio N` | hard negatives mined per positive anchor (torchvision default 3) |
| `--zoom-min F --zoom-prob P` | zoom-crop augmentation, for matching webcam scale |
| `--resume CKPT` | fine-tune from a checkpoint instead of COCO weights |

Training **splits by video, never by frame**. The 100 labelled frames inside one clip
are near-duplicates, so a frame-level split would leak validation data into training and
report an accuracy that means nothing. It checkpoints on best validation F1, not the
last epoch.

### Dataset queries

```python
from get_meta_by import get_meta_by

get_meta_by()                                       # all 48 videos
get_meta_by('Location', 'COURTYARD')                # 16 videos
get_meta_by('Activity', 'PUZZLE', 'Viewer', 'B, S') # filters in any order
```

Filters: `Location` (OFFICE, COURTYARD, LIVINGROOM), `Activity` (CHESS, JENGA, PUZZLE,
CARDS), `Viewer` and `Partner` (B, S, T, H). Values accept comma-separated lists with or
without spaces. An unrecognised filter name raises `ValueError` rather than being
silently ignored.

Per-frame accessors: `get_frame_path`, `get_bounding_boxes` (4×4 `[x, y, w, h]`, zero
rows for absent hands), `get_segmentation_mask`. `demo1.py` demonstrates all three.

---

## The four stages

Each stage is a module under `pipeline/`, swappable independently. They communicate
through the dataclasses in `pipeline/types.py`.

**One coordinate rule:** every pixel coordinate is **full-frame**, except `HandMask.mask`,
which is ROI-local and carries its `origin` with it. Mixing the two is the classic way
this kind of pipeline breaks silently.

### 1. Detect — `pipeline/detector.py`

Frame in, `HandDetection(box, score, handedness)` out. Backends: `mediapipe` (default)
or `ssdlite`.

### 2. Segment — `pipeline/segmenter.py`

Box + landmarks in, `HandMask` out. Three methods:

- `landmark` — convex hull of the 21 points, dilated. Fast, robust, coarse edges.
- `skin` — YCrCb+HSV thresholding **inside the ROI**. Colour thresholding fails badly on
  a whole frame (it finds faces and walls), but inside an already-detected hand box the
  face isn't present, so colour becomes a strong signal again.
- `hybrid` *(default)* — landmark hull as a spatial prior, skin mask to refine the
  silhouette. Measured **IoU 0.750** against EgoHands' ground-truth polygons, versus
  0.733 for landmark-only and 0.722 for skin-only.

### 3. Pose — `pipeline/pose.py`

21 landmarks: wrist `0`; thumb `1-4`; index `5-8`; middle `9-12`; ring `13-16`;
pinky `17-20`. Fingertips are `4, 8, 12, 16, 20`.

> **Design note.** MediaPipe returns the box *and* the landmarks from one inference.
> Keeping stages 1 and 3 conceptually separate while running MediaPipe twice would have
> halved the frame rate for nothing, so they share a per-frame cached runner
> (`pipeline/mediapipe_hands.py`): **86.3 ms → 42.9 ms per frame**. The stages stay
> independently swappable; the cost is paid once.

### 4. Track — `pipeline/tracker.py`

Associates detections across frames by IoU (handedness biases the match), assigns stable
ids, and maintains trajectories.

**Velocity is in pixels per SECOND, not per frame.** A per-frame velocity silently means
different things at 15 fps and 60 fps, and this pipeline runs at both. Tracks survive
`max_missing` frames of non-detection so a momentary miss doesn't renumber everything.

Gesture classification (`pipeline/gestures.py`) derives finger extension geometrically
and **normalises by hand size**, so a hand near the camera classifies the same as one
far away. The thumb gets its own rule because it folds sideways across the palm rather
than curling in.

---

## Datasets

### EgoHands — the primary training set

48 videos × 100 annotated frames = **4,800 frames**, a balanced 4×3×4 design:

| Dimension | Values |
| --- | --- |
| Activity | CARDS, CHESS, JENGA, PUZZLE |
| Location | COURTYARD, LIVINGROOM, OFFICE |
| Viewer / Partner | B, S, T, H |

Each frame has polygon annotations for four hands: *own left, own right, other left,
other right*. `metadata.mat` (19 MB, committed) holds them; `_LABELLED_SAMPLES/` (1.2 GB,
gitignored) holds the JPEGs.

> **Two things to know before using it for motion work.**
>
> The 100 "labelled frames" per video are sampled roughly **0.7 seconds apart** (median
> gap 21 frames at 30 fps), so consecutive annotations are *not* consecutive in time.
> Frame-to-frame tracking on them is meaningless — record your own video instead.
>
> **6.8% of hand polygons extend up to 0.99 px outside the 1280×720 frame**, so a box
> clamped to the frame cannot contain them. Clamping wins; a box must not reference
> pixels that don't exist.

### Optional third-party sets

Used by `--external`. All verified live; none require a login.

| Dataset | Size | Content | Hand scale* |
| --- | --- | --- | --- |
| **COCO-Hand-S** | 1.30 GB | 4,534 COCO photos, 10,845 hands. Whole people, faces and feet visible and unlabelled | 0.062 |
| **HaGRID 30k @384p** | 1.01 GB | 31,833 images, one person frontal — the webcam framing | 0.163 |
| **Oxford Hand** | 250 MB | Unconstrained third-person photos. Good as an eval set | — |
| EgoHands | 1.3 GB | head-mounted, hands on a table | 0.207 |
| *your webcam* | — | — | *0.439* |

\* median hand size as a fraction of the frame.

```bash
mkdir -p datasets && cd datasets
curl -L -C - -O "http://vision.cs.stonybrook.edu/~supreeth/COCO-Hand.zip"
curl -L -C - -o hagrid-30k.zip \
  "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"
unzip -q -n COCO-Hand.zip -d COCO-Hand && unzip -q -n hagrid-30k.zip -d hagrid-30k
```

**Parsing traps, both of which silently corrupt training if missed:**

- COCO-Hand columns are `xmin, xmax, ymin, ymax` — **not** the conventional
  `xmin, ymin, xmax, ymax`. Read as xyxy, every box comes out with `x2 < x1`.
- HaGRID boxes are **normalised `[x, y, w, h]`**, not pixel xyxy.
- Take **COCO-Hand-S**, not COCO-Hand-Big: per its own README, Big paints black circles
  over hands its automatic detector missed, so a "hand-free" region there may contain a
  real hand.

`external_datasets.py` handles all three.

---

## Accuracy

### Detector, on 1,200 held-out EgoHands frames

The committed checkpoint reports **P 0.968 / R 0.747 / F1 0.843**. The EgoHands-only
model before third-party data scored:

```
AP@0.50           0.9335      <- threshold-free headline number
AP@0.75           0.7447
AP@[0.50:0.95]    0.6297      <- COCO primary metric
best F1           0.903 at score threshold 0.20
```

Recall by hand size: **large 0.992, medium 0.833, small 0.152**. Small distant hands are
the known weakness — under 32 px, below what the coarsest feature map resolves.

By slice (consistent, so it isn't overfit to one setting): LIVINGROOM 0.937, COURTYARD
0.936, OFFICE 0.914; PUZZLE 0.971, CARDS 0.940, CHESS 0.937, **JENGA 0.859** (hands are
constantly occluded behind the tower).

### Pipeline stages, scored against ground truth

From `pytest tests/ -m accuracy` — these measure whether results are *right*, not
whether the code runs:

| Stage | Measured against | Result |
| --- | --- | --- |
| 1 Detect | annotated boxes | P 0.930, R 0.435, mean IoU 0.758 |
| 2 Segment | annotated polygons | mask IoU mean 0.750, median 0.771 |
| 3 Pose | anatomical consistency | 42 poses, all valid |
| 4 Track | known synthetic motion | 300.0 px/s at both 15 and 60 fps |
| 4 Track | two hands crossing | no id swap |

Stage 1's recall of 0.435 is MediaPipe on *egocentric* footage, which is not its domain.
Precision 0.930 is the number that matters there. On a webcam it is far stronger; on the
same EgoHands frames the trained SSDlite reaches 0.761 recall against MediaPipe's 0.443.

---

## Google Colab

Open **`hand_pipeline_colab.ipynb`** in Colab and upload **`hand_pipeline_colab.zip`**
when cell 2 asks for it. Both are in this repository.

### Colab is not your laptop

Colab runs on a **remote virtual machine** with no camera and no screen:

| Works locally | On Colab |
| --- | --- |
| `cv2.VideoCapture(0)` | **fails** — the VM has no webcam. The notebook bridges to your browser's camera with JavaScript `getUserMedia` |
| `cv2.imshow(...)` | **fails** — no display. Use `cv2_imshow` |
| `python run_pipeline.py` | **fails** — it opens a window. Call the stages directly, as the notebook does |

### What to upload where

Everything lands in `/content`, which is **wiped when the runtime disconnects** —
re-run the setup cells after a reconnect.

| File | Size | How it gets there | Needed for |
| --- | --- | --- | --- |
| `hand_pipeline_colab.zip` | 52 KB | **you upload** (cell 2) | everything |
| `models/hand_landmarker.task` | 7.5 MB | notebook downloads it | everything |
| `checkpoints/hand_detector.pth` | 8.7 MB | **you upload**, optional | only `--detector ssdlite` |
| `_LABELLED_SAMPLES/` | 1.2 GB | notebook downloads it | training only |

**For the live demo you need only the first two rows.**

### Notebook sections

1. Install dependencies, upload the code
2. Fetch the MediaPipe model, verify imports
3. Build the pipeline once, reused by later cells
4. **Webcam** — JavaScript bridge, capture a frame, run all four stages, show the result
5. **Video** — upload a clip, process every frame, report per-track path length, play it back
6. *(optional)* download EgoHands and train

Section 5 is where motion becomes measurable: a still image has no trajectory. Note the
re-encode step — OpenCV's `mp4v` will not play in a browser, so the notebook converts to
H.264 with ffmpeg.

### Colab troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `NotFoundError` on the .task file | model not downloaded | re-run section 3 |
| Camera cell hangs | browser permission denied | allow camera, reload the page |
| numpy version error after install | mediapipe pins `numpy<2` | Runtime → Restart, re-run from section 2 |
| Video player is black | `mp4v` isn't browser-playable | run the ffmpeg re-encode cell |
| Everything vanished | runtime disconnected, `/content` wiped | re-run sections 2 and 3 |

---

## Project layout

```
run_pipeline.py              the live demo -- start here
pipeline/
  types.py                   shared dataclasses; the coordinate contract
  mediapipe_hands.py         one MediaPipe runner shared by stages 1 and 3
  detector.py                stage 1
  segmenter.py               stage 2
  pose.py                    stage 3
  tracker.py                 stage 4
  gestures.py                gesture classification
  visualize.py               overlay rendering

webcam_detect.py             standalone demo, trained SSDlite
webcam_detect_mp.py          standalone demo, MediaPipe
webcam_detect_cv.py          standalone demo, classical OpenCV

train_detector.py            fine-tune SSDlite
evaluate_detector.py         AP, threshold sweeps, failure analysis
detection_dataset.py         EgoHands as a detection dataset
external_datasets.py         COCO-Hand / HaGRID loaders + generic negatives

get_meta_by.py               query videos by location/activity/viewer/partner
get_frame_path.py            video + frame index -> JPEG path
get_bounding_boxes.py        hand boxes
get_segmentation_mask.py     hand masks
demo1.py                     dataset demo: frame + mask + boxes
metadata.mat                 EgoHands annotations (committed)

tests/                       583 tests
hand_pipeline_colab.ipynb    Colab notebook
hand_pipeline_colab.zip      upload this alongside it
```

---

## Testing

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -q              # all 583
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -m e2e -v -s    # full pipeline, ~30 s
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -m accuracy -v -s  # stage accuracy, ~8 s
```

`MPLBACKEND=Agg` prevents matplotlib from opening blocking windows.

Two suites are worth knowing about:

**`test_e2e_pipeline.py`** runs the real journey — dataset → train → checkpoint →
evaluate → infer → camera — and its central assertion is that a freshly trained model
must **beat its own random-initialised baseline**. A training loop that runs cleanly but
doesn't learn passes every shape check, every smoke test and every CI job, and ships a
useless model. It also gates AP@0.50 at 0.85, so an accuracy regression fails the suite.

**`test_pipeline_accuracy.py`** scores each stage against ground truth rather than
checking it executes.

---

## Known issues

**The SSDlite detector boxes faces, elbows and feet on a webcam.** Measured: it fires on
~99% of frames containing a face and no hand, and no confidence threshold separates the
two — at 0.70 the false-positive rate (77%) exceeds the true-detection rate (62%).

This is a property of the training data, not a bug. EgoHands is head-mounted footage in
which the **only** skin-coloured objects are hands, so the model never had to learn "hand
vs other body part" — it learned the shortcut "skin-coloured blob of roughly this size =
hand". That is 100% correct on EgoHands and wrong on any frame containing a face.

Mitigations attempted: hard negatives, COCO-Hand + HaGRID, generic hand-free crops from
thousands of people, zoom augmentation to match webcam scale, and raising torchvision's
hard-negative mining ratio. Precision rose from 0.961 to 0.978, but the false-positive
behaviour is **not confirmed fixed**.

**Use MediaPipe on a webcam** (the default). The SSDlite model remains the better choice
for first-person footage, which is what it was trained for.

Smaller items: small hands (<32 px) have 0.152 recall; box localisation is loose above
IoU 0.80; `webcam_detect_cv.py` detects *skin-coloured blobs*, not hands, so it will box
your face and sleeves — that's the approach, not a defect.

---

## Credits

The EgoHands dataset and all its raw annotations come from:

```bibtex
@InProceedings{Bambach_2015_ICCV,
  author    = {Bambach, Sven and Lee, Stefan and Crandall, David J. and Yu, Chen},
  title     = {Lending A Hand: Detecting Hands and Recognizing Activities
               in Complex Egocentric Interactions},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month     = {December},
  year      = {2015}
}
```

The dataset accessors (`get_meta_by`, `get_frame_path`, `get_bounding_boxes`,
`get_segmentation_mask`) are a Python port of Indiana University's original MATLAB
toolkit, originating from [shondle/EgoHands_Dataset](https://github.com/shondle/EgoHands_Dataset)
(maintainer: shivansh.s@utexas.edu).

Third-party components: `ssdlite320_mobilenet_v3_large` from torchvision (BSD-3-Clause);
MediaPipe HandLandmarker from Google (Apache-2.0); COCO-Hand and HaGRID under their own
respective licences.
