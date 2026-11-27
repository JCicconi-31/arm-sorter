# arm-sorter

Braccio robotico 4-DOF stampato in 3D che smista oggetti colorati, guidato da una webcam fissa dall'alto.
Il laptop fa tutto il calcolo (visione, calibrazione, cinematica); l'ESP32 riceve solo angoli via seriale.

## Obiettivo del progetto

Progetto di apprendimento: prima capire, poi scrivere. Spiega il perché delle scelte,
non riscrivere grandi parti di codice senza che l'utente lo chieda, e preferisci
modifiche piccole che l'utente possa seguire.

## Struttura

- `src/arm_sorter/` — il pacchetto Python (layout src)
  - `vision/` — rilevamento oggetti (oggi: soglie HSV)
  - `calibration/`, `kinematics/`, `sim/` — step futuri, per ora solo un README
- `tools/` — script interattivi (webcam, finestre, slider). Usano il pacchetto, non contengono logica.
- `tests/` — pytest, con immagini sintetiche: nessun test deve richiedere webcam o hardware.
- `firmware/` — codice ESP32 (Arduino framework), fuori dal pacchetto Python.
- `docs/` — immagini e GIF per il README.

## Ambiente

- Python >= 3.12 (la `.venv` locale è 3.14). Dipendenze solo in `pyproject.toml`, niente `requirements.txt`.
- Installazione: `pip install -e ".[dev]"` dentro la `.venv`.
- Test: `python -m pytest`.
- Su Windows la venv si attiva con `.venv\Scripts\Activate.ps1`.

## Convenzioni

- La logica va in funzioni pure dentro `src/arm_sorter/` (input array → output dati), così è testabile.
  Finestre OpenCV, `VideoCapture` e `waitKey` restano negli script in `tools/`.
- Commenti e docstring in italiano, nomi di variabili/funzioni in inglese.
- I commenti spiegano il *perché* (es. perché HSV, perché apertura prima di chiusura), non il *cosa*.
- Immagini in BGR (convenzione OpenCV), coordinate pixel come `(x, y)` ma indicizzazione array `[y, x]`.
- Hue OpenCV va da 0 a 179 (mezzi gradi).
- Unità fisiche: cm sul tavolo, gradi per gli angoli dei giunti verso l'ESP32.
- Ogni nuova funzione in `src/` ha almeno un test in `tests/`.
- `ROADMAP.md`, `Simulazioni3D/` e `demo.mp4` sono privati e ignorati da git: non committarli.
