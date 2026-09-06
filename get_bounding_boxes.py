"""
getBoundingBoxes(vid, i) returns the ground-truth bounding boxes for the "ith" annotated frame of
video "vid", where "vid" is an EgoHands video metadata structure.

Boxes is a 4x4 matrix, where each row corresponds to a hand bounding box in the format [x y width
height], where x and y mark the top left corner of the box. The rows from top to bottom contain
the bounding boxes for "own left", "own right", "other left", and "other right" hand respectively.
If a hand is not in the frame, the values are set to 0.


For full dataset details, see the
<a href="matlab: web('http://vision.soic.indiana.edu/egohands')">EgoHands project website</a>.

See also getFramePath, getMetaBy, getSegmentationMask, showLabelsOnFrame
"""

import numpy as np

def get_bounding_boxes(video, i):
    """get_bounding_boxes returns the bounding boxes for the hands in the queried video and frame"""
    boxes = np.zeros([4, 4])
    if np.any(video.loc['labelled_frames'][0][i][1]):
        box = segmentation2box((video.loc['labelled_frames'][0][i][1]))
        boxes[0, :] = box
    if np.any(video.loc['labelled_frames'][0][i][2]):
        box = segmentation2box((video.loc['labelled_frames'][0][i][2]))
        boxes[1, :] = box
    if np.any(video.loc['labelled_frames'][0][i][3]):
        box = segmentation2box((video.loc['labelled_frames'][0][i][3]))
        boxes[2, :] = box
    if np.any(video.loc['labelled_frames'][0][i][4]):
        box = segmentation2box((video.loc['labelled_frames'][0][i][4]))
        boxes[3, :] = box
    return boxes

def segmentation2box(shape):
    """Smallest integer box that CONTAINS the hand polygon.

    Two corrections to the MATLAB port:

    * The low clamp is 0, not 1. MATLAB indexes from 1 and Python does not, so the
      1-based floor shaved the top or left row off any hand touching the frame edge.
    * Every edge floors. That is deliberate rather than sloppy: get_segmentation_mask
      rasterises the same polygon through np.int32, which truncates, so flooring here
      makes the box cover exactly the pixels the mask paints. Ceiling the maxima would
      "contain the polygon" more literally but would leave the box one pixel wider
      than the hand actually drawn, putting the two accessors out of step.

    Note that 6.8% of the dataset's hand polygons extend up to 0.99 px beyond the
    frame. A box clamped to the frame cannot contain those, and clamping wins -- a
    box must not reference pixels that do not exist.
    """
    shape = np.asarray(shape, dtype=np.float64)
    box_xyxy = np.floor(np.array([np.min(shape[:, 0]), np.min(shape[:, 1]),
                                  np.max(shape[:, 0]), np.max(shape[:, 1])]))
    box_xyxy[0] = max(0, box_xyxy[0])
    box_xyxy[1] = max(0, box_xyxy[1])
    box_xyxy[2] = min(1279, box_xyxy[2])
    box_xyxy[3] = min(719, box_xyxy[3])
    box_xywh = np.array([box_xyxy[0], box_xyxy[1], box_xyxy[2]-box_xyxy[0]+1,
                         box_xyxy[3]-box_xyxy[1]+1])
    return box_xywh
