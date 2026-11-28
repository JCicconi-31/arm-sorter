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


@dataclass(frozen=True)
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def __post_init__(self):
        
        for name, lo, hi, limit in (
            ("h", self.h_min, self.h_max, 179),
            ("s", self.s_min, self.s_max, 255),
            ("v", self.v_min, self.v_max, 255),
        ):
            if not (0 <= lo <= limit) or not (0 <= hi <= limit):
                raise ValueError(
                    f"{name}_min e {name}_max devono essere in [0, {limit}], ricevuto ({lo}, {hi})"
                )
            if name != "h" and lo > hi:
                    raise ValueError(f"{name}_min ({lo}) maggiore di {name}_max ({hi})")

def to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
    """Ammorbidisce con GaussianBlur (sennò i bordi tremano) e converte in HSV.

    HSV (Tinta, Saturazione, Luminosità) perché la tinta cambia poco tra luce
    e ombra, in BGR invece cambiano tutti i canali.
    """
    return cv2.cvtColor(cv2.GaussianBlur(frame_bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)


def color_mask(hsv: np.ndarray, r: HSVRange) -> np.ndarray:
    """ Crezione della maschera in base al colore che rileva"""
    if r.h_min <= r.h_max:
        lo_1 = np.array((r.h_min, r.s_min, r.v_min))
        hi_1 = np.array((r.h_max, r.s_max, r.v_max))
        mask = cv2.inRange(hsv, lo_1, hi_1)
    else:
        # la hue è un cerchio (0–179) e il rosso sta a cavallo dello 0, quindi due intervalli        
        lo_1 = np.array((0, r.s_min, r.v_min))
        hi_1 = np.array((r.h_max, r.s_max, r.v_max))
        lo_2 = np.array((r.h_min, r.s_min, r.v_min))
        hi_2 = np.array((179, r.s_max, r.v_max))
        mask = cv2.bitwise_or(cv2.inRange(hsv, lo_1, hi_1), cv2.inRange(hsv, lo_2, hi_2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)   # toglie i puntini
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)  # toglie i buchi
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


def find_object(frame_bgr: np.ndarray, r: HSVRange, min_area: float = MIN_AREA):
    """Pipeline completa. Restituisce (blob o None, maschera, immagine HSV)."""
    hsv = to_hsv(frame_bgr)
    mask = color_mask(hsv, r)
    return largest_blob(mask, min_area), mask, hsv
