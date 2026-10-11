"""Descarga reproducible, carga y deduplicación del corpus SMS Spam Collection (UCI #228)."""
import csv
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
              "master/data/sms.tsv")
DEFAULT_PATH = "data/raw/sms_spam.csv"
LABELS = ("ham", "spam")


def _read_bytes(url: str) -> bytes:
    if not url.startswith(("https://", "http://")):
        raise ValueError("La fuente debe ser una URL HTTP(S)")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _parse(raw: bytes) -> pd.DataFrame:
    if raw[:2] == b"PK":  # ZIP de UCI: contiene SMSSpamCollection y readme
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            raw = zf.read("SMSSpamCollection")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), sep="\t", header=None, names=["label", "text"],
                       quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def download_corpus(url: str = UCI_URL, destination: str = DEFAULT_PATH,
                    fallback: str | None = MIRROR_URL) -> Path:
    """Descarga el corpus y lo guarda como CSV con columnas label y text."""
    try:
        frame = _parse(_read_bytes(url))
    except Exception as exc:
        if not fallback:
            raise
        print(f"Fuente principal no disponible ({exc}); usando espejo.")
        frame = _parse(_read_bytes(fallback))
    if frame.empty:
        raise ValueError("El corpus descargado está vacío")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_corpus(path: str = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def dedup_key(text: str) -> str:
    """Clave para detectar duplicados y casi duplicados: minúsculas, dígitos -> 0, espacios colapsados."""
    text = text.lower()
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos (mismo texto y etiqueta) y añade el grupo de casi duplicados."""
    out = frame.drop_duplicates(subset=["label", "text"]).copy()
    out["group"] = out["text"].map(dedup_key)
    return out.reset_index(drop=True)
