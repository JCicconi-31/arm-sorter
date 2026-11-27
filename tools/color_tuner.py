"""Regolazione dal vivo delle soglie HSV sulla webcam.

Uso:  python tools/color_tuner.py [indice_webcam]
"""

import sys

import cv2
import numpy as np

from arm_sorter.vision import find_object

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = webcam portatile
cap = cv2.VideoCapture(cam_index)
# controllo disponibilità cam, meglio fermarsi subito che fallire dopo senza errore
if not cap.isOpened():
    raise SystemExit("Webcam non trovata: prova a cambiare l'indice (1, 2...)")

# creazione dei controlli, le soglie dipendono dalla luce quindi si regolano dal vivo
cv2.namedWindow("controlli")
# Valori iniziali pensati per un oggetto giallo (H arriva a 179 perché OpenCV usa metà gradi)
for name, val, top in [("H min", 20, 179), ("H max", 35, 179),
                       ("S min", 100, 255), ("S max", 255, 255),
                       ("V min", 100, 255), ("V max", 255, 255)]:
    cv2.createTrackbar(name, "controlli", val, top, lambda _: None)

# lettura soglie dalle barre
g = lambda n: cv2.getTrackbarPos(n, "controlli")

hsv = None  # così il clic prima del primo frame non dà errore

# clic sull'oggetto -> stampa i valori HSV reali, più veloce che andare a tentativi
def stampa_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and hsv is not None:
        print("HSV:", hsv[y, x])  # [riga, colonna] -> [y, x]
cv2.namedWindow("webcam")
cv2.setMouseCallback("webcam", stampa_hsv)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    lo = np.array([g("H min"), g("S min"), g("V min")])
    hi = np.array([g("H max"), g("S max"), g("V max")])
    blob, mask, hsv = find_object(frame, lo, hi)

    if blob is not None:
        #rettangolo intorno all'oggetto identificato (magenta, si vede anche sul giallo)
        cv2.rectangle(frame, (blob.x, blob.y), (blob.x + blob.w, blob.y + blob.h), (255, 0, 255), 2)
        #cerchio al baricentro oggetto
        cv2.circle(frame, (blob.cx, blob.cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"centro: ({blob.cx}, {blob.cy}) px", (blob.x, blob.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("webcam", frame)
    cv2.imshow("maschera", mask)  # serve per regolare le soglie, è quello che "vede" l'algoritmo
    # waitKey serve anche ad aggiornare le finestre
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# rilascio webcam così non resta occupata
cap.release()
cv2.destroyAllWindows()
