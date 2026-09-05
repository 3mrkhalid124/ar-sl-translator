"""محرّك مترجم لغة الإشارة العربية — كل المنطق (لا شيء من الواجهة هنا).

USAGE (أدوات تحقق ذاتي في الطرفية):
  python engine.py --fetch-data     # M2: نزّل+تحقق+استخرج الداتا وصور القاموس (npz مخزَّن)
  python engine.py --train          # M3: تدريب CNN + حفظ checkpoint + class_map.json
  python engine.py --selftest       # تشغيل كل الاختبارات الصغيرة المتاحة (بدون إنترنت/كاميرا)
"""

import json
import logging
import logging.handlers
import queue
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = ROOT / "logs"
DICT_DIR = ASSETS_DIR / "dict"
HAND_MODEL = ASSETS_DIR / "hand_landmarker.task"

PARQUET = DATA_DIR / "arasl.parquet"
ARASL_NPZ = DATA_DIR / "arasl.npz"
SIGNS_PNG = DATA_DIR / "Signs_32_New.png"

# المصدر: مرآة HF لـ ArASL2018 (نفس 54,049 صورة، نفس class 0..31، CC BY 4.0).
# السبب: روابط Mendeley المباشرة تعيد 403/Cloudflare لطلبات غير-متصفح (يثبت 2026-09-05).
HF_PARQUET_URL = ("https://huggingface.co/datasets/pain/ArASL_Database_Grayscale/"
                  "resolve/main/data/train-00000-of-00001-aa6a48ea2f282316.parquet")
HF_PARQUET_SIZE = 30479019
SIGNS_MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/y7pckrw6z2/"
                      "files/263ffdd1-9599-4bd8-afaf-64918244050e/file_downloaded")

CLASS_NAMES = [
    "عين", "ال", "ألف", "باء", "دال", "ظاء", "ضاد", "فاء", "قاف", "غين",
    "هاء", "حاء", "جيم", "كاف", "خاء", "لام-ألف", "لام", "ميم", "نون", "راء",
    "صاد", "سين", "شين", "طاء", "تاء", "ثاء", "ذال", "تاء مربوطة", "واو",
    "ياء همزة", "ياء", "زاي",
]
CLASS_SYMS = [
    "ع", "ال", "أ", "ب", "د", "ظ", "ض", "ف", "ق", "غ",
    "هـ", "ح", "ج", "ك", "خ", "لا", "ل", "م", "ن", "ر",
    "ص", "س", "ش", "ط", "ت", "ث", "ذ", "ة", "و", "ئ", "ي", "ز",
]

EXPECTED_IMAGES = 54049
EXPECTED_CLASSES = 32
IMG_SIZE = 64

CONF_THRESHOLD = 0.85
DEBOUNCE_FRAMES = 3
SILENCE_SECONDS = 2.5
MAX_WORD_LEN = 40

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 30

_log_queue = queue.Queue(-1)


def setup_logging() -> None:
    """Logging لا-حظري (QueueHandler + Listener thread) — لا يُبطئ الحلقة الحية."""
    if getattr(setup_logging, "_listener", None) is not None:
        return
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
    file.setFormatter(fmt)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root = logging.getLogger("arsl")
    root.setLevel(logging.INFO)
    if not root.handlers:
        qh = logging.handlers.QueueHandler(_log_queue)
        root.addHandler(qh)
        listener = logging.handlers.QueueListener(_log_queue, file, stream)
        listener.start()
        setup_logging._listener = listener


def log() -> logging.Logger:
    setup_logging()
    return logging.getLogger("arsl")


def _maybe_download(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        return
    log().info("تنزيل %s ...", dest.name)
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if dest.stat().st_size != expected_size:
        raise RuntimeError(f"حجم غير صحيح لـ {dest.name}: {dest.stat().st_size}")
    log().info("تنزيل OK: %s (%d bytes)", dest.name, dest.stat().st_size)


# ---------------------------------------------------------------- البيانات

def _cell_bytes(cell):
    if isinstance(cell, dict) and "bytes" in cell:
        return cell["bytes"]
    if isinstance(cell, bytes):
        return cell
    raise TypeError(f"خلية صورة غير معروفة: {type(cell)}")


def _decode_image(cell: bytes):
    """فك أي صورة → رمادي مقاس موحّد 64×64 (الداتا تحتوي 64×64 و 256×256 و 1024×768)."""
    raw = _cell_bytes(cell)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("صورة غير قابلة للفك")
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def fetch_dataset(force: bool = False) -> None:
    """M2: نزّل parquet، حلّل كل الصور، تحقق، وخزّن npz (عال Retries: يستخدم npz القائم)."""
    if ARASL_NPZ.exists() and not force:
        log().info("npz موجود — تخطي التنزيل (%d rows)", _npz_stack(ARASL_NPZ)[1])
        extract_dictionary_samples()
        return
    _maybe_download(HF_PARQUET_URL, PARQUET, HF_PARQUET_SIZE)

    df = pd.read_parquet(PARQUET)
    raw_lbl = df["label"] if "label" in df.columns else df.iloc[:, -1]
    raw_img = df["image"] if "image" in df.columns else df.iloc[:, 0]

    if str(raw_lbl.dtype) in {"object", "string"}:
        lbl = raw_lbl.astype("category").cat.codes.to_numpy()
    else:
        lbl = raw_lbl.astype(int).to_numpy()

    images = np.empty((len(df), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    ok = np.ones(len(df), dtype=bool)
    bad = 0
    for i in range(len(df)):
        try:
            images[i, :, :, 0] = _decode_image(raw_img.iloc[i]).astype(np.float32) / 255.0
        except Exception:
            ok[i] = False
            bad += 1
    if bad:
        log().warning("%d صفوف غير قابلة للفك (تُتجاوز)", bad)
        images = images[ok]
        lbl = lbl[ok]

    uniq = np.unique(lbl)
    assert len(uniq) == EXPECTED_CLASSES, f"أصناف {uniq}"
    assert images.shape[0] == EXPECTED_IMAGES, f"عدد الصور {images.shape[0]}"
    assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 1)

    np.savez_compressed(ARASL_NPZ, images=images, labels=lbl)
    _signs_reference()
    extract_dictionary_samples()
    log().info("داتا سليمة: %d صورة | أصناف %d | تخزين %s", images.shape[0], len(uniq), ARASL_NPZ.name)


def _npz_stack(path: Path):
    z = np.load(path)
    return z["images"], z["labels"]


def _signs_reference() -> None:
    """لوحة الـ32 إشارة (اختيارية؛ إن فشلت تُترك لعرض الأيقونات لكل صنف)."""
    try:
        _maybe_download(SIGNS_MENDELEY_URL, SIGNS_PNG, 2613952)
    except Exception as exc:
        log().warning("لوحة الإشارات المرجعية فشلت (تبقى اختيارية): %s", exc)


def load_dataset(force_rebuild: bool = False):
    """(images float32 N,64,64,1 في [0,1], labels int N) — من npz المخزَّن أو إعادة البناء."""
    if not ARASL_NPZ.exists() or force_rebuild:
        fetch_dataset()
    return _npz_stack(ARASL_NPZ)


def extract_dictionary_samples() -> None:
    """لكل صنف صورة مرجعية → assets/dict/class_XX.png (مصدر تبويب القاموس)."""
    DICT_DIR.mkdir(exist_ok=True)
    images, labels = _npz_stack(ARASL_NPZ)
    for i in range(EXPECTED_CLASSES):
        out = DICT_DIR / f"class_{i:02d}.png"
        if out.exists():
            continue
        mask = labels == i
        if not mask.any():
            log().warning("لا صور للصنف %d", i)
            continue
        img = (images[mask][0, :, :, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(out), img)
    log().info("صور القاموس جاهزة: %d", len(list(DICT_DIR.glob("*.png"))))


if __name__ == "__main__":
    setup_logging()
    if "--fetch-data" in sys.argv:
        fetch_dataset()
        imgs, labels = load_dataset()
        print(f"[SELFTEST] fetch_data: PASS shape={imgs.shape} classes={len(np.unique(labels))} "
              f"dict_pngs={len(list(DICT_DIR.glob('*.png')))}")
    else:
        print("استخدام: --fetch-data | --train | --selftest")