"""Murmur local engine.

Serves the Murmur web app and transcribes audio/video with faster-whisper.
Run:  uvicorn server:app --port 8000   then open http://localhost:8000

Settings (environment variables, all optional):
  MURMUR_SPRINT_MODEL / MURMUR_STEADY_MODEL / MURMUR_PRECISE_MODEL   Whisper model per mode
  MURMUR_DEVICE        auto | cpu | cuda                 (auto: the GPU when CTranslate2 can use it)
  MURMUR_COMPUTE       auto | int8 | float16 | ...       (auto: int8 on CPU, float16 on GPU)
  MURMUR_THREADS       CPU threads per transcription     (auto: about one per physical core)
  MURMUR_BATCH         auto | 0 | N   batched decoding   (auto: 8 on GPU, off on CPU; needs faster-whisper 1.1+)
  MURMUR_WORKERS       transcriptions running at once    (default 1; more jobs wait in a queue)
  MURMUR_MAX_MODELS    Whisper models kept in memory     (default 2)
  MURMUR_MAX_BYTES     upload limit in bytes             (default 5 GB)
  MURMUR_JOB_TTL       seconds a finished job is kept    (default 86400)
  MURMUR_PORT / MURMUR_OPEN_BROWSER=1                    open the app in the browser once the server is up
  HF_TOKEN             enables pyannote speaker recognition
  MURMUR_DIAR_MODEL    pyannote pipeline                 (default pyannote/speaker-diarization-3.1)
"""
import bisect
import gc
import importlib.util
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import ctranslate2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel, decode_audio

try:  # faster-whisper 1.1+
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None

ROOT = Path(__file__).parent
SAMPLE_RATE = 16000
PEAK_COUNT = 720
SENTENCE_END = ".?!…。？！"
DIAR_SHARE = 85          # percent of the progress bar used by Whisper when speakers are recognised too


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _pick_device() -> str:
    want = os.getenv("MURMUR_DEVICE", "auto").lower()
    if want != "auto":
        return want
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _pick_compute(device: str) -> str:
    want = os.getenv("MURMUR_COMPUTE", "auto").lower()
    if want not in ("auto", "default", ""):
        return want
    try:
        supported = ctranslate2.get_supported_compute_types(device)
    except Exception:
        return "default"
    for kind in (("float16", "int8_float16", "float32") if device == "cuda" else ("int8", "float32")):
        if kind in supported:
            return kind
    return "default"


def _pick_threads() -> int:
    # CTranslate2 only uses 4 threads unless told otherwise. One per physical core is
    # fastest; hyper-threads don't help, so assume half of the logical cores are real.
    n = _env_int("MURMUR_THREADS", 0)
    if n > 0:
        return n
    if os.getenv("OMP_NUM_THREADS"):
        return 0                               # leave the choice to OMP_NUM_THREADS
    return min(16, max(4, (os.cpu_count() or 4) // 2))


MODE_MODELS = {
    "sprint": os.getenv("MURMUR_SPRINT_MODEL", "base"),
    "steady": os.getenv("MURMUR_STEADY_MODEL", "small"),
    "precise": os.getenv("MURMUR_PRECISE_MODEL", "large-v3"),
}
DEVICE = _pick_device()
COMPUTE = _pick_compute(DEVICE)
THREADS = _pick_threads()
WORKERS = max(1, _env_int("MURMUR_WORKERS", 1))
MAX_MODELS = max(1, _env_int("MURMUR_MAX_MODELS", 2))
MAX_BYTES = _env_int("MURMUR_MAX_BYTES", 5 * 1024**3)
JOB_TTL = _env_int("MURMUR_JOB_TTL", 24 * 3600)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
DIAR_MODEL = os.getenv("MURMUR_DIAR_MODEL", "pyannote/speaker-diarization-3.1")


def _pick_batch() -> int:
    if BatchedInferencePipeline is None:
        return 0
    want = os.getenv("MURMUR_BATCH", "auto").lower()
    if want in ("auto", ""):
        return 8 if DEVICE == "cuda" else 0
    try:
        return max(0, int(want))
    except ValueError:
        return 0


BATCH = _pick_batch()

_models: OrderedDict = OrderedDict()   # model name -> WhisperModel, least recently used first
_model_lock = threading.RLock()
_diar = None
_diar_lock = threading.Lock()
jobs: dict = {}
_jobs_lock = threading.Lock()
_queue: queue.Queue = queue.Queue()     # jobs wait here; WORKERS daemon threads work through them
_page = {"mtime": None, "html": ""}


class Cancelled(Exception):
    pass


def log(msg: str):
    print(f"[murmur] {msg}", flush=True)


# ---------- models ----------

def _use_cpu(reason: str):
    """Give up on the graphics card for the rest of this run."""
    global DEVICE, COMPUTE, BATCH
    log(f"the graphics card can't be used ({reason}); continuing on the processor")
    with _model_lock:
        DEVICE, BATCH = "cpu", 0
        COMPUTE = _pick_compute("cpu")
        _models.clear()
    gc.collect()


def _load_model(name: str) -> WhisperModel:
    try:
        return WhisperModel(name, device=DEVICE, compute_type=COMPUTE, cpu_threads=THREADS)
    except Exception as exc:
        if DEVICE == "cpu":
            raise
        _use_cpu(str(exc))
        return WhisperModel(name, device=DEVICE, compute_type=COMPUTE, cpu_threads=THREADS)


def get_model(mode: str) -> WhisperModel:
    name = MODE_MODELS.get(mode, MODE_MODELS["precise"])
    with _model_lock:
        if name in _models:
            _models.move_to_end(name)
            return _models[name]
        while len(_models) >= MAX_MODELS:      # free memory before loading the next model
            old, _ = _models.popitem(last=False)
            log(f"unloading model '{old}'")
        gc.collect()
        _models[name] = _load_model(name)
        return _models[name]


# ---------- speaker recognition ----------

def diarization_status() -> str:
    if not HF_TOKEN:
        return "off (set HF_TOKEN and install pyannote.audio)"
    try:
        found = importlib.util.find_spec("pyannote.audio") is not None   # no heavy torch import
    except (ImportError, ValueError):
        found = False
    return "pyannote" if found else "off (pip install pyannote.audio)"


def get_diarizer():
    global _diar
    with _diar_lock:
        if _diar is None:
            import torch
            from pyannote.audio import Pipeline
            try:
                pipe = Pipeline.from_pretrained(DIAR_MODEL, token=HF_TOKEN)             # pyannote 4
            except TypeError:
                pipe = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)    # pyannote 3
            if pipe is None:
                raise RuntimeError(f"Could not load {DIAR_MODEL}. Accept its terms on huggingface.co and check HF_TOKEN.")
            if torch.cuda.is_available():
                pipe.to(torch.device("cuda"))
            _diar = pipe
        return _diar


def diarize(audio: np.ndarray, count: str) -> list:
    """Run pyannote on the audio that is already decoded. Returns [(start, end, label)] sorted by start."""
    import torch
    kwargs = {"num_speakers": int(count)} if str(count).isdigit() else {}
    result = get_diarizer()({"waveform": torch.from_numpy(audio)[None], "sample_rate": SAMPLE_RATE}, **kwargs)
    annotation = getattr(result, "speaker_diarization", result)   # pyannote 4 wraps the annotation
    return sorted((t.start, t.end, label) for t, _, label in annotation.itertracks(yield_label=True))


class _Turns:
    """Speaker turns with a fast 'who talks most in this span' lookup."""

    def __init__(self, turns: list):
        self.turns = sorted(turns)
        self.starts = [t[0] for t in self.turns]
        self.longest = max((e - s for s, e, _ in self.turns), default=0.0)

    def speaker(self, start: float, end: float, nearest: float = 1.0):
        lo = bisect.bisect_left(self.starts, start - self.longest)
        hi = bisect.bisect_right(self.starts, end + nearest)
        overlap: dict = {}
        near, near_gap = None, nearest
        for s, e, label in self.turns[lo:hi]:
            ov = min(end, e) - max(start, s)
            if ov > 0:
                overlap[label] = overlap.get(label, 0.0) + ov
            elif -ov <= near_gap:
                near, near_gap = label, -ov
        return max(overlap, key=overlap.get) if overlap else near


def _phrases(words: list) -> list:
    """Split a segment's words into sentences, the only places where the speaker label may change."""
    out, cur = [], []
    for w in words:
        cur.append(w)
        if w["text"].rstrip()[-1:] in SENTENCE_END:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def assign_speakers(segments: list, turns: list) -> list:
    """Label segments with speakers S1, S2, ... in order of appearance.

    Segments that carry word timings are split where the speaker changes between sentences,
    so a reply doesn't end up glued to the question before it.
    """
    lookup, names, out = _Turns(turns), {}, []

    def name(label):
        return None if label is None else names.setdefault(label, f"S{len(names) + 1}")

    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append({**seg, "speaker": name(lookup.speaker(seg["start"], seg["end"]))})
            continue
        parts = []
        for phrase in _phrases(words):
            label = lookup.speaker(phrase[0]["start"], phrase[-1]["end"])
            if parts and (label is None or label == parts[-1][0]):
                parts[-1][1].extend(phrase)
            else:
                parts.append([label, list(phrase)])
        if len(parts) > 1 and parts[0][0] is None:      # leading words nobody claimed
            parts[1][1][:0] = parts[0][1]
            parts.pop(0)
        for label, ws in parts:
            text = "".join(w["text"] for w in ws).strip()
            if text:
                out.append({"start": round(ws[0]["start"], 2), "end": round(ws[-1]["end"], 2), "text": text, "speaker": name(label)})
    return out


# ---------- jobs ----------

def waveform_peaks(audio: np.ndarray, n: int = PEAK_COUNT) -> list:
    """Loudness envelope for the player's waveform, normalised to 0..1."""
    block = len(audio) // n
    if block < 1:
        return []
    chunks = audio[: block * n].reshape(n, block)
    peaks = np.maximum(chunks.max(axis=1), -chunks.min(axis=1))
    top = float(peaks.max())
    return [] if top <= 0 else [round(float(v), 2) for v in peaks / top]


def purge_jobs():
    cutoff = time.time() - JOB_TTL
    with _jobs_lock:
        for job_id in [k for k, j in jobs.items() if j.get("finished") and j["finished"] < cutoff]:
            del jobs[job_id]


def public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    if job["status"] == "queued":
        with _jobs_lock:
            out["ahead"] = sum(1 for j in jobs.values() if j["status"] == "queued" and j["_n"] < job["_n"])
    return out


def _check(job: dict):
    if job["_cancel"]:
        raise Cancelled()


def _transcribe(job: dict, tag: str, audio: np.ndarray, mode: str, language: str, task: str, vad: bool, diarizing: bool) -> list:
    name = MODE_MODELS.get(mode, MODE_MODELS["precise"])
    duration = len(audio) / SAMPLE_RATE
    job.update(phase="Loading model", progress=0)
    if name not in _models:
        log(f"{tag}: loading Whisper model '{name}' (the first use downloads it, please wait)")
    model = get_model(mode)
    _check(job)

    options = dict(
        language=None if language == "auto" else language,
        task="translate" if task == "translate" else "transcribe",
        beam_size=5 if mode == "precise" else 1,
        word_timestamps=diarizing,               # lets us split lines where the speaker changes
    )
    job["phase"] = "Transcribing"
    if BATCH > 1:
        # Batched decoding needs the voice activity filter to cut the audio into chunks.
        seg_iter, info = BatchedInferencePipeline(model=model).transcribe(
            audio, batch_size=BATCH, vad_filter=True, without_timestamps=False, **options)
    else:
        seg_iter, info = model.transcribe(audio, vad_filter=vad, **options)
    log(f"{tag}: language {info.language}, audio length {duration / 60:.1f} min, model '{name}' on {DEVICE}/{COMPUTE}")
    job["language"] = info.language

    share = DIAR_SHARE if diarizing else 99      # diarization takes the rest of the progress bar
    out, last_logged = [], 0
    try:
        for s in seg_iter:
            _check(job)
            text = s.text.strip()
            if not text:
                continue
            seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": text, "speaker": None}
            if diarizing and s.words:
                seg["words"] = [{"start": w.start, "end": w.end, "text": w.word} for w in s.words]
            out.append(seg)
            job["progress"] = min(share, int(s.end / duration * share))
            if job["progress"] - last_logged >= 10:
                last_logged = job["progress"]
                log(f"{tag}: {job['progress']}%")
    finally:
        if hasattr(seg_iter, "close"):
            seg_iter.close()
    return out


def run_job(job_id: str, path: str, language: str, mode: str, task: str, vad: bool, speakers: bool, speaker_count: str):
    job = jobs.get(job_id)
    tag = job_id[:6]
    try:
        if job is None:
            return
        _check(job)
        t0 = time.time()
        job.update(status="processing", phase="Reading audio")
        try:
            audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        except Exception as exc:
            reason = getattr(exc, "strerror", None) or str(exc)       # PyAV errors carry the temp path, skip it
            raise RuntimeError(f"This file couldn't be read as audio or video ({reason}).") from exc
        if len(audio) == 0:
            raise RuntimeError("This file has no audio track.")
        try:                                   # the upload isn't needed any more
            os.remove(path)
        except OSError:
            pass
        duration = len(audio) / SAMPLE_RATE
        job.update(duration=round(duration, 2), peaks=waveform_peaks(audio))
        _check(job)

        diarizing = speakers and diarization_status() == "pyannote"
        if speakers and not diarizing:
            job["warning"] = "Speaker recognition is off on this server. Set HF_TOKEN and install pyannote.audio to enable it."
            log(f"{tag}: {job['warning']}")
        try:
            out = _transcribe(job, tag, audio, mode, language, task, vad, diarizing)
        except RuntimeError as exc:
            # Typical on Windows: an NVIDIA card is found, but cuBLAS/cuDNN are missing and the
            # first real computation fails. Carry on with the processor instead of giving up.
            if DEVICE == "cpu" or not any(k in str(exc).lower() for k in ("cuda", "cublas", "cudnn")):
                raise
            _use_cpu(str(exc))
            out = _transcribe(job, tag, audio, mode, language, task, vad, diarizing)

        if diarizing and out:
            job.update(progress=DIAR_SHARE, phase="Recognizing speakers")
            log(f"{tag}: recognizing speakers")
            try:
                out = assign_speakers(out, diarize(audio, speaker_count))
            except Exception as exc:           # keep the transcript even if diarization breaks
                job["warning"] = f"Speaker recognition failed: {exc}"
                log(f"{tag}: {job['warning']}")
                traceback.print_exc()
        for seg in out:
            seg.pop("words", None)
        _check(job)
        job.update(status="done", progress=100, phase="Done", segments=out)
        took = time.time() - t0
        log(f"{tag}: done in {took:.0f}s ({duration / max(took, 0.01):.1f}x real time), {len(out)} segments")
    except Cancelled:
        job.update(status="cancelled", phase="Cancelled")
        log(f"{tag}: cancelled")
    except Exception as exc:  # report any failure to the client
        job.update(status="error", error=str(exc) or exc.__class__.__name__)
        log(f"{tag}: FAILED: {exc}")
        traceback.print_exc()
    finally:
        if job is not None:
            job["finished"] = time.time()
        try:
            os.remove(path)
        except OSError:
            pass


def _sweep_temp():
    """Uploads a crashed or closed server left behind (only old ones: another instance may be running)."""
    cutoff = time.time() - 24 * 3600
    for f in Path(tempfile.gettempdir()).glob("murmur-*"):
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def _worker():
    while True:
        args = _queue.get()
        try:
            run_job(*args)
        except BaseException:                  # a worker must survive anything a job throws
            traceback.print_exc()


# ---------- web app ----------

@asynccontextmanager
async def lifespan(_: FastAPI):
    batch = f", batch {BATCH}" if BATCH > 1 else ""
    threads = f"{THREADS} CPU threads" if THREADS else "CPU threads from OMP_NUM_THREADS"
    log(f"device {DEVICE}, compute {COMPUTE}, {threads}{batch}, speaker recognition: {diarization_status()}")
    _sweep_temp()
    for i in range(WORKERS):
        threading.Thread(target=_worker, name=f"murmur-{i}", daemon=True).start()
    if os.getenv("MURMUR_OPEN_BROWSER") == "1":
        url = f"http://localhost:{os.getenv('MURMUR_PORT', '8000')}"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    yield
    for job in list(jobs.values()):
        job["_cancel"] = True


app = FastAPI(title="Murmur local engine", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    src = ROOT / "index.html"
    mtime = src.stat().st_mtime
    if _page["mtime"] != mtime:
        head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>'
        _page.update(mtime=mtime, html=head + src.read_text(encoding="utf-8") + "</body></html>")
    return HTMLResponse(_page["html"], headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    with _jobs_lock:
        active = sum(1 for j in jobs.values() if j["status"] in ("queued", "processing"))
    return {
        "engine": "faster-whisper", "models": MODE_MODELS, "device": DEVICE, "compute": COMPUTE,
        "threads": THREADS, "batch": BATCH, "diarization": diarization_status(),
        "active_jobs": active, "max_bytes": MAX_BYTES,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    mode: str = Form("precise"),
    speakers: str = Form("0"),
    speaker_count: str = Form("auto"),
    task: str = Form("transcribe"),
    vad: str = Form("0"),
):
    if (getattr(file, "size", None) or 0) > MAX_BYTES:
        raise HTTPException(413, "File is larger than the server's upload limit.")
    suffix = Path(file.filename or "upload").suffix or ".bin"
    fd, path = tempfile.mkstemp(prefix="murmur-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            await run_in_threadpool(shutil.copyfileobj, file.file, out, 4 * 1024 * 1024)
        size = os.path.getsize(path)
        if size > MAX_BYTES:
            raise HTTPException(413, "File is larger than the server's upload limit.")
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    purge_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "phase": "Waiting", "_cancel": False, "_n": time.monotonic()}
    log(f"{job_id[:6]}: received '{file.filename}' ({size / 1e6:.1f} MB), mode {mode}, language {language}")
    _queue.put((job_id, path, language, mode, task, vad == "1", speakers == "1", speaker_count))
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    return public(job)


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    """Cancel a job that is waiting or running, or drop the stored result of a finished one."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job.get("finished"):
        with _jobs_lock:
            jobs.pop(job_id, None)
        return {"status": "removed"}
    job["_cancel"] = True
    return {"status": "cancelling"}
