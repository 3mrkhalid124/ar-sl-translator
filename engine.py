"""محرّك مترجم لغة الإشارة العربية — كل المنطق (لا شيء من الواجهة هنا).

USAGE (أدوات تحقق ذاتي في الطرفية):
  python engine.py --fetch-data     # M2: نزّل+تحقق+استخرج الداتا وصور القاموس (npz مخزَّن)
  python engine.py --train          # M3: تدريب CNN + حفظ checkpoint + class_map.json
  python engine.py --camera         # M4: اختبار مباشر للكاميرا + كشف اليد + تنصيف CNN
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
import torch

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
GROQ_MODEL = "allam-2-7b"  # عربي-مختص (بديل llama-3.3-70b-versatile الذي أزيل من الكتالوج 2026)
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


# ---------------------------------------------------------------- CNN

MODEL_PATH = MODELS_DIR / "cnn.pt"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"
TRAIN_FRAC = 0.85
MAX_EPOCHS = 3  # أقصى عدد epochs في هذه الجلسة الواحدة (توقف مبكر عند val ≥ 0.95)
BATCH_SIZE = 512
LR = 2e-3
TARGET_VAL_ACC = 0.95
CPU_THREADS = 6  # قياس: MKL أسرع عند 6 خيوط (نوى فيزيائية) على i7-10850H


def _build_cnn():
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.Conv2d(24, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 16
                nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 8
            )
            self.head = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(64 * 8 * 8, 32),
            )

        def forward(self, x):
            x = self.features(x)
            return self.head(x.flatten(1))

    return CNN()


def _write_class_map() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    CLASS_MAP_PATH.write_text(
        json.dumps({str(i): {"sym": s, "name": n} for i, (s, n) in enumerate(zip(CLASS_SYMS, CLASS_NAMES))},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def train_cnn() -> None:
    """M3: تدريب → التحقق على 15% held-out → حفظ cnn.pt + class_map.json. Goal: val ≥ 95%.
    يستأنف من models/cnn.pt إن وُجد (نفس المعمارية). الحد الأقصى: MAX_EPOCHS epochs؛
    توقف مبكر تلقائي عند أول epoch يصل فيه val_acc ≥ TARGET_VAL_ACC."""
    import torch
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(CPU_THREADS)
    images, labels = load_dataset()
    x = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)  # (N,1,64,64)
    x_train, x_val, y_train, y_val = train_test_split(
        x, labels, test_size=1 - TRAIN_FRAC, stratify=labels, random_state=42)

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    model = _build_cnn()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = torch.nn.CrossEntropyLoss()
    train_n = x_train.shape[0]
    best_acc = 0.0
    reached = False

    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            log().info("استئناف من %s — weights محمّلة (نفس المعمارية)", MODEL_PATH.name)
        except Exception as exc:
            log().warning("تعذر تحميل %s (%s) — بدء من الصفر", MODEL_PATH.name, exc)

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        perm = torch.randperm(train_n)
        t0 = time.time()
        for i in range(0, train_n, BATCH_SIZE):
            ids = perm[i:i + BATCH_SIZE]
            xb, yb = x_train[ids], y_train[ids]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            pred = out.argmax(1)
            total += yb.numel()
            correct += (pred == yb).sum().item()
            loss_sum += loss.item() * yb.numel()
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            vcorrect = 0
            for i in range(0, x_val.shape[0], BATCH_SIZE):
                out = model(x_val[i:i + BATCH_SIZE])
                vcorrect += (out.argmax(1) == y_val[i:i + BATCH_SIZE]).sum().item()
        val_acc = vcorrect / y_val.shape[0]

        log().info("epoch %02d | train acc %.4f | val acc %.4f | loss %.3f | %.1fs",
                   ep, train_acc, val_acc, loss_sum / total, time.time() - t0)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
        if val_acc >= TARGET_VAL_ACC:
            log().info("الوصول للهدف (val ≥ %.2f) عند epoch %d — توقف مبكر", TARGET_VAL_ACC, ep)
            reached = True
            break

    if not reached:
        log().warning("val acc النهائي %.4f < الهدف %.2f بعد %d epochs", best_acc,
                      TARGET_VAL_ACC, MAX_EPOCHS)

    _per_class_val(x_val, y_val)
    _write_class_map()
    log().info("أُنجز التدريب: best val %.4f | checkpoint %s | calc %s", best_acc, MODEL_PATH.name, CLASS_MAP_PATH.name)
    if reached:
        print(f"[SELFTEST] train: PASS best_val_acc={best_acc:.4f} target={TARGET_VAL_ACC}")
    else:
        print(f"[SELFTEST] train: FAIL best_val_acc={best_acc:.4f} target={TARGET_VAL_ACC}")


def _per_class_val(x_val, y_val) -> None:
    import torch
    model = _build_cnn().eval()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    with torch.no_grad():
        preds = model(x_val).argmax(1).numpy()
    accs = []
    for i in range(EXPECTED_CLASSES):
        mask = y_val.numpy() == i
        accs.append(round(float((preds[mask] == i).mean()), 3))
    log().info("دقة كل صنف (val): %s", dict(zip(CLASS_SYMS, accs)))
    worst = sorted(zip(CLASS_SYMS, accs), key=lambda t: t[1])[:3]
    log().warning("أضعف 3 أصناف: %s", worst)


def _get_model():
    """يحمّل cnn.pt مرة واحدة ويخزّنه (حاسم لـ fps في الحلقة الحية)."""
    model = getattr(_get_model, "cache", None)
    if model is None:
        model = _build_cnn().eval()
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        _get_model.cache = model
    return model


def classify(patch: np.ndarray):
    """CNN inference: patch float32 (64,64) أو (64,64,1) في [0,1] → (class_idx, confidence)."""
    import torch
    p = patch if patch.ndim == 2 else patch[:, :, 0]
    x = torch.from_numpy(p.astype(np.float32)[None, None]).float()
    with torch.no_grad():
        probs = torch.softmax(_get_model()(x), dim=1)[0].numpy()
    idx = int(probs.argmax())
    return idx, float(probs[idx])


# ---------------------------------------------------------------- M4: كاميرا + يد

ENG_LABELS = [
    "Ayn", "Al", "Alef", "Baa", "Dal", "Thaa", "Dhad", "Faa", "Qaf", "Ghain",
    "Haa", "Hha", "Jeem", "Kaf", "Khaa", "Lam-Alef", "Lam", "Meem", "Noon", "Raa",
    "Sad", "Seen", "Sheen", "Taa", "Tta", "Thaal", "Thal", "Ta-Marb", "Waw",
    "Yaa-Ha", "Yaa", "Zayn",
]
HAND_PAD = 1.4  # هامش حول كف اليد قبل التكبير — يقارب نسبة اليد لعين الصور التدريبية
ARSL_GRAYSCALE_ON_BLACK = True  # صور ArASL غالباً يد بيضاء على خلفية سوداء


def _get_hand_detector():
    """HandLandmarker (Tasks API VIDEO) — singleton كـ _get_model (حاسم لـ fps)."""
    det = getattr(_get_hand_detector, "cache", None)
    if det is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        det = mp_vision.HandLandmarker.create_from_options(options)
        _get_hand_detector.cache = det
    return det


def detect_hand(frame, ts_ms: int):
    """M4: كشف كف اليد الأولى → قائمة النقاط (x,y معيّرة) أو [] عند غيابها.
    يتطلب timestamp متزايد لكل إطار (RunningMode.VIDEO)."""
    import mediapipe as mp
    det = _get_hand_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = det.detect_for_video(mp_image, ts_ms)
    return result.hand_landmarks[0] if result and result.hand_landmarks else []


def crop_hand_patch(frame, landmarks, pad: float = HAND_PAD):
    """قص مربع حول اليد → رمادي (64,64) float في [0,1] + bbox (x0,y0,x1,y1).
    طبيعية القطبية: إن كانت الخلفية أفتح من اليد تُعكس الصورة (مطابقة لـ ArASL)."""
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * pad
    if size < 4:
        return None, None
    x0 = max(0, int(cx - size / 2))
    y0 = max(0, int(cy - size / 2))
    x1 = min(w, int(x0 + size))
    y1 = min(h, int(y0 + size))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, None
    roi = frame[y0:y1, x0:x1]
    gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE),
                      interpolation=cv2.INTER_AREA)
    norm = gray.astype(np.float32) / 255.0
    if ARSL_GRAYSCALE_ON_BLACK and norm.mean() > 0.5:
        norm = 1.0 - norm
    return norm, (x0, y0, x1, y1)


def process_frame(frame, ts_ms: int):
    """M4: إطار → (overlay_frame, result)
    result: {hand, idx, label, label_en, conf, bbox} — ما يستهلكه تبويب Streamlit."""
    overlay = frame.copy()
    result = {"hand": False, "idx": None, "label": None, "label_en": None,
              "conf": 0.0, "bbox": None}
    landmarks = detect_hand(frame, ts_ms)
    if landmarks:
        result["hand"] = True
        patch, bbox = crop_hand_patch(frame, landmarks)
        if patch is not None:
            idx, conf = classify(patch)
            result.update(idx=idx, label=CLASS_NAMES[idx], label_en=ENG_LABELS[idx],
                          conf=conf, bbox=bbox)
            h, w = frame.shape[:2]
            for lm in landmarks:
                cv2.circle(overlay, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            x0, y0, x1, y1 = bbox
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(overlay, f"{result['label_en']} {conf:.2f}", (x0, max(16, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return overlay, result


def run_camera(source=0):
    """اختبار M4 في الطرفية: نافذة مباشرة + إخراج السجل. ESC للإيقاف."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("الكاميرا غير متاحة (source=%s)" % source)
    log().info("كاميرا مفتوحة — اهتزاز/توقف بإغلاقها أو ESC")
    t_start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log().warning("إطار فاشل")
                continue
            ts = int((time.monotonic() - t_start) * 1000)
            overlay, result = process_frame(frame, ts)
            if result["hand"]:
                log().info("%s (%.2f)", result["label"], result["conf"])
            cv2.imshow("ArASL — اختبار الكاميرا", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------- M5/M6: سلسلة إشارات + كلمة

class SignSequencer:
    """M5+M6: التزام بالحرف عبر debounce (ثقة ≥ عتبة لعدة إطارات)، بناء كلمة، إغلاقها عند الصمت.
    لا يُعاد الالتزام بنفس الحرف حتى انقطاع اليد أو تغيّر الحرف (يمنع تكراراً صناعياً)."""

    def __init__(self, conf_threshold=CONF_THRESHOLD, debounce=DEBOUNCE_FRAMES,
                 silence_seconds=SILENCE_SECONDS, max_word=MAX_WORD_LEN):
        self.conf_threshold = conf_threshold
        self.debounce = debounce
        self.silence_seconds = silence_seconds
        self.max_word = max_word
        self._votes = {}
        self._last_idx = None
        self._last_conf = 0.0
        self._committed_idx = None
        self.word = []
        self.last_hand_time = 0.0
        self.finalized_word = None

    def feed(self, hand_seen: bool, idx=None, conf=0.0, ts=0.0):
        events = {"committed": None, "word": "".join(CLASS_SYMS[i] for i in self.word), "finalized": None}
        if hand_seen and idx is not None:
            self.last_hand_time = ts
            if idx == self._last_idx and conf >= self.conf_threshold:
                self._votes[idx] = self._votes.get(idx, 0) + 1
                self._last_conf = conf
            elif conf >= self.conf_threshold:
                self._votes = {idx: 1}
                self._last_idx = idx
                self._last_conf = conf
            else:
                self._votes = {}
                self._last_idx = idx
            if idx != self._committed_idx and self._votes.get(idx, 0) >= self.debounce:
                if len(self.word) >= self.max_word:
                    self._finalize()
                self.word.append(idx)  # نصيحة: نخزّن الرمز
                events["committed"] = (idx, CLASS_SYMS[idx], self._last_conf)
                self._committed_idx = idx
                self._votes[idx] = 0
            events["word"] = "".join(CLASS_SYMS[i] for i in self.word)
        else:
            self._last_idx = None
            self._committed_idx = None
            if self.word and (self.last_hand_time == 0.0 or ts - self.last_hand_time >= self.silence_seconds):
                self._finalize()
                events["finalized"] = self.finalized_word
                events["word"] = ""
        return events

    def _finalize(self):
        self.finalized_word = "".join(CLASS_SYMS[i] for i in self.word)
        self.word = []


# ---------------------------------------------------------------- M7: Groq تصحيح

def _load_groq_key() -> str:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def correct_word(raw: str):
    """M7: Groq llama-3.3-70b → تصحيح كلمة عربية. بلا مفتاح/إنترنت → raw بلا كسر."""
    key = _load_groq_key()
    if not key or not raw:
        log().info("Groq: مفتاح غير متاح — raw %r", raw)
        return {"text": raw, "source": "raw", "api": False}
    t0 = time.time()
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "أنت مصحّح لغوي عربي. سيصلك نص من حروف إشارة عربية قد يكون هجاءً "
                        "متصلاً بلا تشكيل. صحّحه إلى كلمة/جملة عربية سليمة وردّ بها فقط.")},
                    {"role": "user", "content": raw},
                ],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=GROQ_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        log().info("Groq: %r -> %r (%.1fs)", raw, text, time.time() - t0)
        return {"text": text, "source": "groq", "api": True}
    except Exception as exc:
        log().warning("Groq فشل (%s) — raw %r", exc, raw)
        return {"text": raw, "source": "raw", "api": False}


# ---------------------------------------------------------------- M8: صوت

def synthesize_speech(text: str):
    """M8: gTTS ar → bytes mp3 (يُشغَّل عبر st.audio). فشل → None بلا كسر."""
    if not text or not text.strip():
        return None
    try:
        from gtts import gTTS
        import io
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang="ar").write_to_fp(buf)
        buf.seek(0)
        log().info("TTS: %d bytes لـ %r", buf.getbuffer().nbytes, text)
        return buf.getvalue()
    except Exception as exc:
        log().warning("gTTS فشل (%s) — نص بلا صوت", exc)
        return None


# ---------------------------------------------------------------- خط أنابيب حي (مستهلك الواجهة)

class LivePipeline:
    """M4+M5+M6 (+M7 عبر correct_word) في نقاط جاهزة للواجهة. process_frame محسوب مرة/إطار."""

    def __init__(self, conf_threshold=CONF_THRESHOLD, debounce=DEBOUNCE_FRAMES,
                 silence_seconds=SILENCE_SECONDS, auto_correct=True):
        self.seq = SignSequencer(conf_threshold, debounce, silence_seconds)
        self.t0 = time.monotonic()
        self.auto_correct = auto_correct

    def update(self, frame):
        ts = time.monotonic() - self.t0
        overlay, result = process_frame(frame, int(ts * 1000))
        events = self.seq.feed(result["hand"], result["idx"], result["conf"], ts)
        out = {**result, "events": events, "overlay": overlay,
               "word": events["word"], "corrected": None, "audio": None}
        if events["finalized"]:
            out["corrected"] = (correct_word(events["finalized"]) if self.auto_correct
                                else {"text": events["finalized"], "source": "raw", "api": False})
            out["audio"] = synthesize_speech(out["corrected"]["text"])
        return out


def run_selftest() -> int:
    """فحص شامل بلا كاميرا/إنترنت حتمي: ملخص PASS/FAIL لكل مقطع."""
    ok = True
    def check(name: str, cond) -> None:
        nonlocal ok
        print(f"[SELFTEST] {name}: {'PASS' if cond else 'FAIL'}")
        ok = ok and bool(cond)

    check("artifacts.npz", ARASL_NPZ.exists())
    check("artifacts.cnn", MODEL_PATH.exists())
    check("artifacts.class_map", CLASS_MAP_PATH.exists())
    check("artifacts.hand_model", HAND_MODEL.exists())
    check("artifacts.dict_pngs", len(list(DICT_DIR.glob("*.png"))) == EXPECTED_CLASSES)

    images, labels = load_dataset()
    check("data.shape", images.shape == (EXPECTED_IMAGES, IMG_SIZE, IMG_SIZE, 1))

    idx, conf = classify(images[0][:, :, 0])
    check("classify.smoke", 0 <= idx < EXPECTED_CLASSES and 0.0 <= conf <= 1.0)

    try:
        import json as _j
        cm = _j.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
        check("class_map.len", len(cm) == EXPECTED_CLASSES)
    except Exception:
        check("class_map.len", False)

    fake_lm = [type("LM", (), {"x": 0.3 + (i / 21) * 0.4, "y": 0.4 + ((i % 5) / 5) * 0.2})()
               for i in range(21)]
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    patch, bbox = crop_hand_patch(black, fake_lm)
    check("crop.patch", patch is not None and patch.shape == (IMG_SIZE, IMG_SIZE))
    overlay, res = process_frame(black, 1)
    check("frame.no_hand", res["hand"] is False)

    seq = SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    ev = seq.feed(True, 2, 1.0, 0.10)
    ev = seq.feed(True, 2, 1.0, 0.20)
    ev = seq.feed(True, 2, 1.0, 0.30)
    check("seq.commit", ev["committed"] is not None and ev["word"] == CLASS_SYMS[2])
    ev = seq.feed(True, 3, 1.0, 0.40)
    ev = seq.feed(True, 3, 1.0, 0.50)
    ev = seq.feed(True, 3, 1.0, 0.60)
    check("seq.second_letter", ev["word"] == CLASS_SYMS[2] + CLASS_SYMS[3])
    ev = seq.feed(False, None, 0.0, 5.00)
    check("seq.finalize", ev["finalized"] == CLASS_SYMS[2] + CLASS_SYMS[3])

    check("groq.key", bool(_load_groq_key()))  # متاح أم لا — معلومة فقط
    check("tts.internet", synthesize_speech("اختبار") is not None)  # إنترنت؛ قابلة للفشل المسموح

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _ = mp_vision.HandLandmarkerOptions
        check("mediapipe.tasks_api", True)
    except Exception:
        check("mediapipe.tasks_api", False)

    print(f"[SELFTEST] overall: {'ALL PASS' if ok else 'SOME FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):  # تفادي cp1252 عند طباعة العربية في PowerShell
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    setup_logging()
    if "--fetch-data" in sys.argv:
        fetch_dataset()
        imgs, labels = load_dataset()
        print(f"[SELFTEST] fetch_data: PASS shape={imgs.shape} classes={len(np.unique(labels))} "
              f"dict_pngs={len(list(DICT_DIR.glob('*.png')))}")
    elif "--train" in sys.argv:
        train_cnn()
        images, labels = load_dataset()
        idx, conf = classify(images[0][:, :, 0])
        print(f"[SELFTEST] classify smoke: idx={idx} en={ENG_LABELS[idx]} "
              f"true={ENG_LABELS[int(labels[0])]} match={(idx == int(labels[0]))}")
    elif "--camera" in sys.argv:
        run_camera()
    elif "--selftest" in sys.argv:
        sys.exit(run_selftest())
    else:
        print("استخدام: --fetch-data | --train | --camera | --selftest")