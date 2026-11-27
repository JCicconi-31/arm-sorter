"""Test della pipeline colore su immagini sintetiche (niente webcam)."""

import cv2
import numpy as np

from arm_sorter.vision import find_object, largest_blob

# soglie per il giallo, le stesse di partenza di tools/color_tuner.py
LO = (20, 100, 100)
HI = (35, 255, 255)
YELLOW_BGR = (0, 255, 255)


def blank(h=240, w=320):
    return np.zeros((h, w, 3), np.uint8)


def test_finds_centroid_of_yellow_square():
    img = blank()
    cv2.rectangle(img, (100, 50), (160, 110), YELLOW_BGR, -1)  # centro (130, 80)
    blob, mask, _ = find_object(img, LO, HI)
    assert blob is not None
    assert abs(blob.cx - 130) <= 1 and abs(blob.cy - 80) <= 1
    assert mask.dtype == np.uint8 and mask.shape == img.shape[:2]


def test_ignores_other_colors():
    img = blank()
    cv2.rectangle(img, (100, 50), (160, 110), (255, 0, 0), -1)  # blu
    blob, _, _ = find_object(img, LO, HI)
    assert blob is None


def test_picks_largest_of_two():
    img = blank()
    cv2.rectangle(img, (10, 10), (40, 40), YELLOW_BGR, -1)     # piccolo
    cv2.rectangle(img, (150, 100), (250, 200), YELLOW_BGR, -1)  # grande, centro (200, 150)
    blob, _, _ = find_object(img, LO, HI)
    assert blob is not None
    assert abs(blob.cx - 200) <= 1 and abs(blob.cy - 150) <= 1


def test_small_blob_is_treated_as_noise():
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (10, 10), (25, 25), 255, -1)  # ~225 px < 500
    assert largest_blob(mask) is None
    assert largest_blob(mask, min_area=100) is not None


def test_empty_mask_returns_none():
    assert largest_blob(np.zeros((50, 50), np.uint8)) is None
