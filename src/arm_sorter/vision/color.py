"""Rilevamento di un oggetto colorato tramite soglie HSV.

Funzioni pure (nessuna finestra, nessuna webcam) così si possono testare
con immagini sintetiche. La parte interattiva sta in tools/color_tuner.py.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# elemento strutturante per apertura/chiusura, 5x5 toglie il rumore senza mangiare i bordi
KERNEL = np.ones((5, 5), np.uint8)

# sotto quest'area (px) le macchie sono di solito rumore o riflessi
MIN_AREA = 500


@dataclass(frozen=True)
class Blob:
    """Oggetto trovato: baricentro e rettangolo di contorno, in pixel."""

    cx: int
    cy: int
    x: int
    y: int
    w: int
    h: int
    area: float


def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Ammorbidisce con GaussianBlur (sennò i bordi tremano) e converte in HSV.

    HSV (Tinta, Saturazione, Luminosità) perché la tinta cambia poco tra luce
    e ombra, in BGR invece cambiano tutti i canali.
    """
    return cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)


def color_mask(hsv: np.ndarray, lo, hi) -> np.ndarray:
    """Maschera binaria: bianco i pixel dentro [lo, hi], nero il resto."""
    mask = cv2.inRange(hsv, np.asarray(lo), np.asarray(hi))
    # prima apertura poi chiusura, al contrario il rumore verrebbe ingrandito invece che tolto
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)   # toglie i puntini
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)  # chiude i buchi
    return mask


def largest_blob(mask: np.ndarray, min_area: float = MIN_AREA) -> Blob | None:
    """Tiene solo la macchia bianca più grande (si assume sia l'oggetto)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= min_area:
        return None
    # baricentro con i momenti: m00 = area, m10/m00 = media x, m01/m00 = media y
    # più stabile del centro del rettangolo se l'oggetto è storto o coperto
    m = cv2.moments(c)
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    x, y, w, h = cv2.boundingRect(c)
    return Blob(cx, cy, x, y, w, h, area)


def find_object(frame_bgr: np.ndarray, lo, hi, min_area: float = MIN_AREA):
    """Pipeline completa. Restituisce (blob o None, maschera, immagine HSV)."""
    hsv = to_hsv(frame_bgr)
    mask = color_mask(hsv, lo, hi)
    return largest_blob(mask, min_area), mask, hsv
