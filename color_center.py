import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 0 = webcam portatile
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

# elemento strutturante per apertura/chiusura, 5x5 toglie il rumore senza mangiare i bordi
kernel = np.ones((5, 5), np.uint8)

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

    # l'immagine viene ammorbidita con GaussianBlur (sennò i bordi tremano) e convertita in HSV (Hue, Saturation, Value) -> (Tinta, Saturazione, Luminosità)
    # HSV perché la tinta cambia poco tra luce e ombra, in BGR invece cambiano tutti i canali
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
    lo = np.array([g("H min"), g("S min"), g("V min")])
    hi = np.array([g("H max"), g("S max"), g("V max")])

    #Ogni pixel che rientra nel range viene marcato di bianco, mentre il resto di nero, così si crea la maschera dell'immagine.
    mask = cv2.inRange(hsv, lo, hi)
    # prima apertura poi chiusura, al contrario il rumore verrebbe ingrandito invece che tolto
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # toglie i puntini
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # chiude i buchi

    # trova i bordi di ogni macchia bianca sulla maschera e tiene solo la più grande (si assume sia l'oggetto)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:  # ignora le macchie piccole, di solito sono rumore o riflessi
            # baricentro con i momenti: m00 = area, m10/m00 = media x, m01/m00 = media y
            # più stabile del centro del rettangolo se l'oggetto è storto o coperto
            m = cv2.moments(c)
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            x, y, w, h = cv2.boundingRect(c)
            #rettangolo intorno all'oggetto identificato (magenta, si vede anche sul giallo)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            #cerchio al baricentro oggetto
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"centro: ({cx}, {cy}) px", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("webcam", frame)
    cv2.imshow("maschera", mask)  # serve per regolare le soglie, è quello che "vede" l'algoritmo
    # waitKey serve anche ad aggiornare le finestre
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# rilascio webcam così non resta occupata
cap.release()
cv2.destroyAllWindows()