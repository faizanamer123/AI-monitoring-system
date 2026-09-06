"""Shared fixtures for the EgoHands end-to-end test suite.

Every module here runs against the REAL project: the real metadata.mat, the real
_LABELLED_SAMPLES frames, and the real trained checkpoint. Nothing mocks the data
layer, because the bugs worth catching live in how those pieces fit together.

Run with:   .venv/bin/python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# get_frame_path builds paths from os.getcwd(), so tests must run from the repo root
os.chdir(REPO)

DATASET = REPO / "_LABELLED_SAMPLES"
CHECKPOINT = REPO / "checkpoints" / "hand_detector.pth"
MP_MODEL = REPO / "models" / "hand_landmarker.task"


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def python_bin():
    """The venv interpreter, for tests that shell out to a CLI entry point."""
    return str(REPO / ".venv" / "bin" / "python")


@pytest.fixture(scope="session")
def videos():
    """All 48 video rows. Session-scoped: loading metadata.mat is slow."""
    from get_meta_by import get_meta_by
    return get_meta_by()


@pytest.fixture(scope="session")
def sample_frame_path():
    """Path to one real labelled frame, or skip if the dataset is not downloaded."""
    if not DATASET.is_dir():
        pytest.skip("_LABELLED_SAMPLES not present")
    for folder in sorted(DATASET.iterdir()):
        if folder.is_dir():
            frames = sorted(folder.glob("frame_*.jpg"))
            if frames:
                return frames[0]
    pytest.skip("no frames found under _LABELLED_SAMPLES")


needs_dataset = pytest.mark.skipif(not DATASET.is_dir(), reason="_LABELLED_SAMPLES not present")
needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="no trained checkpoint")
needs_mp_model = pytest.mark.skipif(not MP_MODEL.is_file(), reason="no mediapipe model file")
