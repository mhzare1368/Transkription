# Murmur – lokale Transkription

Murmur transkribiert Audio- und Videodateien mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
direkt auf dem eigenen Rechner. Die Aufnahmen verlassen den Computer nicht.

| Datei | Aufgabe |
| --- | --- |
| `start-murmur.bat` | Startet alles per Doppelklick (Windows) |
| `server.py` | Lokaler Server: nimmt Dateien an und transkribiert sie |
| `index.html` | Die Oberfläche im Browser |
| `requirements.txt` | Benötigte Python-Pakete |

## Starten

**Windows:** `start-murmur.bat` doppelklicken. Beim ersten Start werden die Python-Pakete
installiert (Python 3.10 oder neuer muss vorhanden sein), danach öffnet sich der Browser von
selbst, sobald der Server bereit ist. Zum Beenden das schwarze Fenster schließen.

**Von Hand (alle Systeme):**

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m uvicorn server:app --port 8000
```

Dann <http://localhost:8000> öffnen. Unter Windows liegt Python in `.venv\Scripts\`.

Beim ersten Verwenden eines Modus wird das zugehörige Whisper-Modell heruntergeladen
(`large-v3` hat rund 3 GB). Das passiert nur einmal.

## Modi

| Modus | Modell | Geeignet für |
| --- | --- | --- |
| Sprint | `base` | schnelle Rohfassung |
| Steady | `small` | guter Kompromiss |
| Precise | `large-v3` | beste Genauigkeit, auf dem Prozessor langsam |

Tipp: `large-v3-turbo` ist deutlich schneller als `large-v3` bei fast gleicher Genauigkeit.
Dazu in `start-murmur.bat` das `rem` vor `set MURMUR_PRECISE_MODEL=large-v3-turbo` entfernen.
Für „Transcribe to English“ ist das Turbo-Modell nicht gedacht.

## Einstellungen

Alle Einstellungen sind Umgebungsvariablen und optional. In `start-murmur.bat` trägt man sie
als `set NAME=wert` ein.

| Variable | Standard | Bedeutung |
| --- | --- | --- |
| `MURMUR_SPRINT_MODEL`, `MURMUR_STEADY_MODEL`, `MURMUR_PRECISE_MODEL` | `base`, `small`, `large-v3` | Whisper-Modell je Modus |
| `MURMUR_DEVICE` | `auto` | `cpu` oder `cuda`; `auto` nimmt die NVIDIA-Grafikkarte, wenn sie nutzbar ist |
| `MURMUR_COMPUTE` | `auto` | Rechengenauigkeit; `auto` wählt `int8` auf dem Prozessor und `float16` auf der Grafikkarte |
| `MURMUR_THREADS` | `auto` | Prozessor-Threads pro Transkription (etwa einer pro echtem Kern) |
| `MURMUR_BATCH` | `auto` | Stapelgröße für gebündeltes Dekodieren; `auto` = 8 auf der Grafikkarte, aus auf dem Prozessor |
| `MURMUR_WORKERS` | `1` | Wie viele Dateien gleichzeitig transkribiert werden; weitere warten |
| `MURMUR_MAX_MODELS` | `2` | Wie viele Modelle im Arbeitsspeicher bleiben |
| `MURMUR_MAX_BYTES` | 5 GB | Größte erlaubte Datei |
| `MURMUR_PORT` | `8000` | Port des Servers (nur `start-murmur.bat`) |
| `HF_TOKEN` | – | Schaltet die Sprechererkennung frei, siehe unten |

Mit `MURMUR_BATCH` größer als 1 wird die Stille-Erkennung immer verwendet, weil das gebündelte
Dekodieren die Aufnahme daran in Stücke teilt.

## Sprechererkennung

Die Option „Recognize speakers“ braucht [pyannote.audio](https://github.com/pyannote/pyannote-audio):

1. Paket installieren: `.venv\Scripts\pip install pyannote.audio`
2. Auf <https://huggingface.co> anmelden und die Nutzungsbedingungen von
   `pyannote/speaker-diarization-3.1` und `pyannote/segmentation-3.0` bestätigen.
3. Ein Zugriffstoken erstellen und in `start-murmur.bat` als `set HF_TOKEN=hf_...` eintragen.

Solange das nicht eingerichtet ist, ist die Option in der Oberfläche ausgegraut.

## Wenn etwas nicht klappt

- **„address already in use“:** Murmur läuft bereits. Den offenen Browser-Tab weiterverwenden
  oder das andere Fenster schließen.
- **Jede Datei schlägt mit „couldn't be read as audio or video“ fehl:** Vermutlich ist PyAV 19
  installiert, mit dem faster-whisper (bis 1.2.1) nicht zusammenarbeitet.
  Abhilfe: `.venv\Scripts\pip install "av<19"`.
- **Fehler mit `cublas` oder `cudnn` auf einem Rechner mit NVIDIA-Karte:** Die CUDA-Bibliotheken
  fehlen. Entweder installieren oder mit `set MURMUR_DEVICE=cpu` auf den Prozessor wechseln.
- **Seite neu geladen oder Browser geschlossen:** Laufende Transkriptionen gehen nicht verloren,
  die Oberfläche verbindet sich wieder mit dem Auftrag. Nach einem Neustart des Servers wird die
  Datei automatisch erneut gesendet; sonst hilft „Retry“ im Menü `⋯` der Datei.
