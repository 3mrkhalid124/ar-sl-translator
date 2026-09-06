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
import os
import queue
import sys
import time
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

# المصدر: مرآة HF لـ ArASL2018 (نفس 54,049 صورة، نفس class 0..31، CC BY 4.0).
# السبب: روابط Mendeley المباشرة تعيد 403/Cloudflare لطلبات غير-متصفح (يثبت 2026-09-05).
HF_PARQUET_URL = ("https://huggingface.co/datasets/pain/ArASL_Database_Grayscale/"
                  "resolve/main/data/train-00000-of-00001-aa6a48ea2f282316.parquet")
HF_PARQUET_SIZE = 30479019

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

CONF_THRESHOLD = 0.90  # Priority 1: 0.85 → 0.90 (يخفف false positives مع التحقق الهندسي)
DEBOUNCE_FRAMES = 5  # Priority 1: 3 → 5
SILENCE_SECONDS = 2.5
MAX_WORD_LEN = 40

# --- [Diagnosis 1] تتبّع تشخيصي مؤقت: يطبع الأرقام الفعلية لكل إطار عند ENGINE_TRACE=1 (بلا كلفة حين يغيب) ---
_TRACE = os.getenv("ENGINE_TRACE") == "1"


def _trace(*args):
    if _TRACE:
        print("[TRACE]", *args, flush=True)

# أداء (Performance — Polish9): الكاميرا مقيّدة بعرض معالجة واحد + تواقيت متزايدة عالمياً.
LIVE_MAX_WIDTH = 640  # تصغير مبكر لفريم الكاميرا قبل المعالجة/الإرسال (كشف اليد شبه ثابت التكلفة)
PROFILE_EVERY = 60  # كل كم إطار يُسجَّل متوسط زمن كل مرحلة في السجل (measure, لا تخمين)

# أنماط الأصابع (مراسي مؤكدة من الصور المرجعية الحقيقية في assets/dict/ — بلا افتراض إضافي):
#   fingers: عدد الأصابع الأربعة الممدودة المتوقع أو None بلا قيد
#   thumb:   "fold"/"ext"/"any" أو None بلا قيد
#   القاعدة في validate_geometry: فرق عدد أصابع ≥ 2 يُرفض؛ تعارض إبهام وحده لا يُرفض.
SIGNPAT = {
    7: {"fingers": 0, "thumb": "ext"},   # فاء: قبضة + إبهام جانبي
    21: {"fingers": 4, "thumb": "ext"},  # سين: كف مفتوح (4 أصابع + إبهام)
    25: {"fingers": 3, "thumb": "fold"}, # ثاء: W — سبابة+وسطى+بنصر ممدودة، إبهام وخنصر مطويان
    30: {"fingers": 1, "thumb": "fold"}, # ياء: سبابة لأعلى والباقي قبضة
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "allam-2-7b"  # عربي-مختص (بديل llama-3.3-70b-versatile الذي أزيل من الكتالوج 2026)
GROQ_TIMEOUT = 30

_log_queue = queue.Queue(-1)


def setup_logging() -> None:
    """Logging لا-حظري (QueueHandler + Listener thread) — لا يُبطئ الحلقة الحية."""
    if getattr(setup_logging, "_listener", None) is not None:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    try:
        LOGS_DIR.mkdir(exist_ok=True)
        file = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
        file.setFormatter(fmt)
    except OSError as exc:
        logging.error("تعذر تهيئة ملف السجل (%s) — Stream فقط", exc)
        file = None
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root = logging.getLogger("arsl")
    root.setLevel(logging.INFO)
    if not root.handlers:
        qh = logging.handlers.QueueHandler(_log_queue)
        root.addHandler(qh)
        handlers = [file] if file is not None else []
        listener = logging.handlers.QueueListener(_log_queue, *(handlers + [stream]))
        listener.start()
        setup_logging._listener = listener


def log() -> logging.Logger:
    """يؤمّن الإعداد ثم يعيد Logger 'arsl' (مركزي للسجل)."""
    setup_logging()
    return logging.getLogger("arsl")


def _maybe_download(url: str, dest: Path, expected_size: int) -> None:
    """ينزّل URL إلى dest إن كان غائباً أو بحجم مختلف؛ يتحقق من الحجم المتوقع."""
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
    """يستخرج بايتات الصورة من خلية parquet (dict-bytes أو bytes مباشرة)."""
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
    extract_dictionary_samples()
    log().info("داتا سليمة: %d صورة | أصناف %d | تخزين %s", images.shape[0], len(uniq), ARASL_NPZ.name)


def _npz_stack(path: Path):
    """(images, labels) من ملف npz (تحميل كسول عبر np.load)."""
    z = np.load(path)
    return z["images"], z["labels"]


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
    """معمارية CNN العربية (1×64×64 → 32 class). تُستورد من engine_en مع استبدال رأسها فقط."""
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
    """يكتب class_map.json (رمز/اسم لكل صنف) — يُقرأ في تبويب القاموس."""
    MODELS_DIR.mkdir(exist_ok=True)
    CLASS_MAP_PATH.write_text(
        json.dumps({str(i): {"sym": s, "name": n} for i, (s, n) in enumerate(zip(CLASS_SYMS, CLASS_NAMES))},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def train_cnn(epochs: int | None = None, target: float | None = None) -> None:
    """M3: تدريب → التحقق على 15% held-out → حفظ cnn.pt + class_map.json. Goal: val ≥ 95%.
    يستأنف من models/cnn.pt إن وُجد (نفس المعمارية). الحد الأقصى: epochs (افتراضاً MAX_EPOCHS)؛
    توقف مبكر تلقائي عند أول epoch يصل فيه val_acc ≥ tv (افتراضاً TARGET_VAL_ACC، قابل للرفع).
    best_acc يبدأ من val الحالي للـ checkpoint إن وُجد — لا يُستبدل checkpoint إلا بنموذج أفضل.
    تكبير مستهدف: الفئات التي acc_checkpoint < 0.91 تُكرَّر 2x في pool كل epoch (إن وُجدت)."""
    import torch
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(CPU_THREADS)
    epochs = epochs or MAX_EPOCHS
    tv = TARGET_VAL_ACC if target is None else target
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
    y_train_np = y_train.numpy()
    best_acc = 0.0
    reached = False
    weak_cls: list = []

    if MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            log().info("استئناف من %s — weights محمّلة (نفس المعمارية)", MODEL_PATH.name)
            model.eval()
            per_cls = np.zeros(EXPECTED_CLASSES)
            cls_cnt = np.zeros(EXPECTED_CLASSES)
            vcorrect = 0
            with torch.no_grad():
                for i in range(0, x_val.shape[0], BATCH_SIZE):
                    xb = x_val[i:i + BATCH_SIZE]
                    yb = y_val[i:i + BATCH_SIZE]
                    pred = model(xb).argmax(1).numpy()
                    ybn = yb.numpy()
                    vcorrect += int((ybn == pred).sum())
                    for c in range(EXPECTED_CLASSES):
                        m = ybn == c
                        per_cls[c] += int((pred[m] == c).sum())
                        cls_cnt[c] += int(m.sum())
            best_acc = vcorrect / y_val.shape[0]
            accs = per_cls / np.maximum(cls_cnt, 1)
            weak_cls = [int(c) for c in range(EXPECTED_CLASSES) if accs[c] < 0.91]
            log().info("checkpoint الحالي val=%.4f — بداية best_acc منه؛ لن يُستبدل إلا بأفضل", best_acc)
            log().info("دقة الفئات (checkpoint الحالي): %s",
                       dict(zip(CLASS_SYMS, [round(float(a), 3) for a in accs])))
            if weak_cls:
                log().warning("فئات ضعيفة مستهدفة بزيادة التكرار 2x (acc<0.91): %s",
                              [CLASS_SYMS[c] for c in weak_cls])
            else:
                log().info("لا فئات ≤ 0.91 — تدريب عادي متوازن")
        except Exception as exc:
            log().warning("تعذر تحميل %s (%s) — بدء من الصفر", MODEL_PATH.name, exc)

    base_pool = np.arange(train_n)
    if weak_cls:
        extra_idx = np.concatenate([np.where(y_train_np == c)[0] for c in weak_cls])
        pool = np.concatenate([base_pool, extra_idx])
    else:
        pool = base_pool
    pool_n = pool.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        perm = np.random.permutation(pool_n)
        ag_rng = np.random.default_rng(11 + ep)
        t0 = time.time()
        for i in range(0, pool_n, BATCH_SIZE):
            ids = torch.from_numpy(pool[perm[i:i + BATCH_SIZE]])
            xb = torch.from_numpy(_augment_batch(x_train[ids].numpy(), ag_rng)).float()
            yb = y_train[ids]
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
        if val_acc >= tv:
            log().info("الوصول للهدف (val ≥ %.2f) عند epoch %d — توقف مبكر", tv, ep)
            reached = True
            break

    if not reached:
        log().warning("val acc النهائي %.4f < الهدف %.2f بعد %d epochs", best_acc,
                      tv, epochs)

    _per_class_val(x_val, y_val)
    _write_class_map()
    log().info("أُنجز التدريب: best val %.4f | checkpoint %s | calc %s", best_acc, MODEL_PATH.name, CLASS_MAP_PATH.name)
    if reached:
        print(f"[SELFTEST] train: PASS best_val_acc={best_acc:.4f} target={tv}")
    else:
        print(f"[SELFTEST] train: FAIL best_val_acc={best_acc:.4f} target={tv}")


def _per_class_val(x_val, y_val) -> None:
    """دقة كل صنف على val + أضعف 3 أصناف (تقرير تدريب فقط)."""
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
    """يحمّل cnn.pt مرة واحدة ويخزّنه (حاسم لـ fps في الحلقة الحية).
    مفقود/تالف → RuntimeError برسالة إجرائية واضحة (لا traceback خام)."""
    model = getattr(_get_model, "cache", None)
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"نموذج CNN مفقود: {MODEL_PATH.name} — شغّل: python engine.py --train")
        try:
            model = _build_cnn().eval()
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        except Exception as exc:
            raise RuntimeError(
                f"تعذر تحميل {MODEL_PATH.name} ({exc}) — أعد التدريب: python engine.py --train") from exc
        _get_model.cache = model
    return model


MARGIN_THRESHOLD = 0.05  # margin-check عام: فرق احتمال top1−top2 المقبول (يطابق EN)


def _augment_batch(xb: np.ndarray, rng) -> np.ndarray:
    """زيادة تدريب on-the-fly: قلب قطبية عشوائي + تدوير/إزاحة/مقياس + تبويض/تباين/ضباب.
    الدليل (قياس offline): القطبية المعكوسة كانت acc 0.156/0.208 (عربي/إنجليزي) وperturbed 0.281
    → يجب أن يتدرب النموذج على نفس الـ transforms بدل الاعتماد على هيئة واحدة."""
    import cv2
    cv2.setNumThreads(0)  # يمنع ذعرها خيوطه مع MKL أثناء التدريب (قياس الزمن: 0.06s/دفعة)
    B = xb.shape[0]
    out = np.empty_like(xb)
    for i in range(B):
        p = xb[i, 0]
        if rng.random() < 0.5:
            p = 1.0 - p
        ang = (rng.random() * 2 - 1) * 12.0
        sc = 0.9 + rng.random() * 0.2
        M = cv2.getRotationMatrix2D((32, 32), ang, sc)
        p = cv2.warpAffine(p, M, (64, 64), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        p = p * (0.75 + rng.random() * 0.5)
        p = p + (rng.random() * 2 - 1) * 0.12
        if rng.random() < 0.5:
            p = cv2.GaussianBlur(p, (0, 0), 0.4 + rng.random() * 0.8)
        out[i, 0] = np.clip(p, 0, 1)
    return out


def classify_softmax(patch: np.ndarray):
    """same as classify لكن يُرجع أيضاً (top1−top2) margin — يُستخدم للـ margin-check."""
    import torch
    p = patch if patch.ndim == 2 else patch[:, :, 0]
    x = torch.from_numpy(p.astype(np.float32)[None, None]).float()
    with torch.no_grad():
        probs = torch.softmax(_get_model()(x), dim=1)[0].numpy()
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    return idx, float(probs[idx]), float(probs[idx] - probs[order[1]])


def margin_accept(margin: float) -> bool:
    """gate عام للعربي: يقبل فقط التصنيفات التي تفوق فيها top1 توب2 بـ >= MARGIN_THRESHOLD."""
    return float(margin) >= MARGIN_THRESHOLD


def classify(patch: np.ndarray):
    """CNN inference: patch float32 (64,64) أو (64,64,1) في [0,1] → (class_idx, confidence)."""
    idx, conf, _margin = classify_softmax(patch)
    return idx, conf


# ---------------------------------------------------------------- M4: كاميرا + يد

_last_ts_ms = 0


def _next_ts_ms() -> int:
    """Timestamp متزايد بصرامة عالمياً (كل العمليات تُشارك HandLandmarker واحداً).
    VIDEO mode يشترط ts متزايداً؛ timestamp نسبي إلى t0 يَتصفَّر مع كل LivePipeline جديد
    (تبديل لغة!) فيُخطئ MediaPipe. إصلاح Performance (Polish9) — قِيس بالبنش لا تخميناً."""
    global _last_ts_ms
    _last_ts_ms = max(_last_ts_ms + 1, int(time.monotonic() * 1000))
    return _last_ts_ms


def downscale_live(frame, max_width: int = LIVE_MAX_WIDTH):
    """تصغير مبكر لفريم الكاميرا إلى max_width قبل المعالجة/الإرسال.
    القياس (Polish9): detect في MediaPipe يكبّر داخلياً فكلفته شبه ثابتة، لكن
    imencode PNG + الإرسال ينفجران مع الدقة (1080p: 25ms/53KB) → تصغير قبلها."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, int(round(h * max_width / w))),
                      interpolation=cv2.INTER_AREA)


def _profile_time(prof, key, start):
    """يسجّل زمن مرحلة في prof إن وُجد (بدون كلفة حين غيابه)."""
    if prof is not None:
        prof[key] = (time.perf_counter() - start) * 1000


class FrameProfiler:
    """يجمع أزمنة المراحل عبر LivePipeline(+EN) ويُسجّل متوسطها كل window إطاراً في السجل."""

    def __init__(self, window: int = PROFILE_EVERY, tag: str = "live"):
        self.window = window
        self.tag = tag
        self.count = 0
        self.sums = {}

    def note(self, prof: dict) -> None:
        if not prof:
            return
        self.count += 1
        for k, v in prof.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        if self.count >= self.window:
            avg = {k: v / self.count for k, v in self.sums.items()}
            total = sum(avg.values())
            log().info("profile[%s] avg_ms/frame total=%.1f  %s",
                       self.tag, total, "  ".join(f"{k}={v:.1f}" for k, v in avg.items()))
            self.count = 0
            self.sums = {}

ENG_LABELS = [
    "Ayn", "Al", "Alef", "Baa", "Dal", "Thaa", "Dhad", "Faa", "Qaf", "Ghain",
    "Haa", "Hha", "Jeem", "Kaf", "Khaa", "Lam-Alef", "Lam", "Meem", "Noon", "Raa",
    "Sad", "Seen", "Sheen", "Taa", "Tta", "Thaal", "Thal", "Ta-Marb", "Waw",
    "Yaa-Ha", "Yaa", "Zayn",
]
HAND_PAD = 1.4  # هامش حول كف اليد قبل التكبير — يقارب نسبة اليد لعين الصور التدريبية
ARSL_GRAYSCALE_ON_BLACK = True  # صور ArASL غالباً يد بيضاء على خلفية سوداء
_TRAIN_MEAN = 0.646  # وسطي السطوع لـ arasl.npz كاملاً (مرجع التوثيق — لا يُستخدم في التحويل الآن)
_TRAIN_STD = 0.261


def normalize_live(norm: np.ndarray) -> np.ndarray:
    """مواءمة قطبية لقطة اليد الحية نحو مظهر التدريب (خلفية مشرقة) — IRON بلا إعادة سطوع.
    الدليل (قياس): canonical الداكن mean<0.5 → flip → يعادل canonical الفاتح acc 0.6875/1.0 بدل 0.156/0.208
    للعكس؛ وإعادة السطوع الإحصائي كان مدمراً (in-domain 0.966→0.777) فاستُبعدت بالقياس."""
    if norm.mean() < 0.5:
        norm = 1.0 - norm
    return norm.astype(np.float32)


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
        det = None
        try:
            det = mp_vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"فشل تهيئة كاشف اليد (تأكد من سلامة {HAND_MODEL.name}): {exc}") from exc
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


# فهارس مفاصل اليد (MediaPipe)
_LM = {"wrist": 0, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
       "index_mcp": 5, "index_pip": 6, "index_tip": 8,
       "middle_pip": 10, "middle_tip": 12,
       "ring_pip": 14, "ring_tip": 16,
       "pinky_pip": 18, "pinky_tip": 20}
_FINGER_LINKS = [  # (pip, tip) لكل إصبع بترتيب: سبابة/وسطى/بنصر/خنصر
    (_LM["index_pip"], _LM["index_tip"]),
    (_LM["middle_pip"], _LM["middle_tip"]),
    (_LM["ring_pip"], _LM["ring_tip"]),
    (_LM["pinky_pip"], _LM["pinky_tip"]),
]


def finger_features(landmarks):
    """ميزات هندسية من 21 نقطة: أصابع الأربعة ممدودة + الإبهام + العدد.
    القاعدة: الإصبع ممدود إذا كان طرفه أبعد من المعصم من مفصل PIP (بهامش 1.1×).
    لا نقاط/أقل من 21 → أصفار (لا معلومات هندسية)."""
    if not landmarks or len(landmarks) < 21:
        return {"fingers": [False, False, False, False], "thumb": False, "num_ext": 0}
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    wrist = pts[_LM["wrist"]]
    def _dist(a, b):
        return float(np.hypot(*(a - b)))
    fingers = [bool(_dist(pts[t], wrist) > _dist(pts[p], wrist) * 1.1) for p, t in _FINGER_LINKS]
    thumb = bool(_dist(pts[_LM["thumb_tip"]], wrist) > _dist(pts[_LM["thumb_ip"]], wrist) * 1.1)
    return {"fingers": fingers, "thumb": thumb, "num_ext": sum(fingers)}


def synthetic_landmarks(fingers=(0, 0, 0, 0), thumb=False, cx=0.5):
    """21 نقطة (x,y) بمسافات صريحة تصدق قاعدة عدّ الأصابع (المعصم أسفل الوسط).
    فِهرسة تتبع _FINGER_LINKS تماماً — تُستعمل في فحوص selftest العربية والإنجليزية."""
    pts = {0: (cx, 0.85)}
    for k, ext in enumerate(fingers):
        b = 0.62 - 0.05 * k          # قاعدة الإصبع
        m = 5 + 4 * k                # فهرس MCP
        for i, y in ((m, b + 0.06), (m + 1, b), (m + 3, b - 0.30 if ext else b + 0.04)):
            pts[i] = (cx, y)
    pts[2] = (cx, 0.61)
    pts[3] = (cx, 0.55)
    pts[4] = (cx + 0.22, 0.45) if thumb else (cx, 0.59)
    return [type("LM", (), {"x": x, "y": y})() for i in range(21)
            for x, y in [pts.get(i, (cx - i * 0.005, 0.6 - i * 0.005))]]


def validate_geometry(landmarks, cls_idx: int, patterns=SIGNPAT):
    """طبقة التحقق الهندسي (Priority 1): إن كان للنمط المعلوم للصنف قيد، نعبّر رفضاً عن تعارضٍ
    واضح بين الهندسة وتصنيف CNN (فرق عدد الأصابع ≥ 2 أو تعارض إبهام حاسم) → 'غير معروف'.
    بلا نقاط (يد فارغة) → قَبول صامت (لا معلومات تُرفض).
    patterns: جدول {cls_idx: {fingers, thumb}} — يأخذ الجدول العربي افتراضياً،
    والجدول الإنجليزي (SIGNPAT_EN) يُمرَّر من engine_en."""
    if not landmarks or cls_idx not in patterns:
        return True
    rule = patterns[cls_idx]
    if not rule:
        return True
    f = finger_features(landmarks)
    ok_fingers = rule.get("fingers") is None or abs(f["num_ext"] - rule["fingers"]) <= 1
    ok_thumb = rule.get("thumb") is None or rule["thumb"] == "any" or rule["thumb"] == f["thumb"]
    if not ok_fingers and not ok_thumb:
        return False
    if not ok_fingers:
        return False
    if not ok_thumb and abs(f["num_ext"] - rule.get("fingers", f["num_ext"])) >= 2:
        return False
    return True


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
    return normalize_live(norm), (x0, y0, x1, y1)


def process_frame(frame, ts_ms: int, prof=None):
    """M4: إطار → (overlay_frame, result)
    result: {hand, unknown, idx, label, label_en, conf, bbox}.
    prof (dict اختياري): يُملأ بأزمنة المراحل (detect/crop/classify/geometry/draw) بالسجلات.
    Priority 1: لو رفض التحقق الهندسي تصنيف CNN → 'غير معروف' (idx=None, conf=0) بدل تثبيت حرف غلط.
    إطار None/فارغ → نتيجة فارغة (لا crash)."""
    result = {"hand": False, "unknown": False, "idx": None, "label": None,
              "label_en": None, "conf": 0.0, "bbox": None}
    if frame is None or getattr(frame, "size", 0) == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8), result
    _t0 = time.perf_counter()
    overlay = frame.copy()
    _profile_time(prof, "copy", _t0)
    _t0 = time.perf_counter()
    landmarks = detect_hand(frame, ts_ms)
    _profile_time(prof, "detect", _t0)
    if landmarks:
        result["hand"] = True
        _t0 = time.perf_counter()
        patch, bbox = crop_hand_patch(frame, landmarks)
        _profile_time(prof, "crop", _t0)
        if patch is not None:
            _t0 = time.perf_counter()
            idx, conf, margin = classify_softmax(patch)
            _profile_time(prof, "classify", _t0)
            _t0 = time.perf_counter()
            ok_geo = validate_geometry(landmarks, idx)
            _profile_time(prof, "geometry", _t0)
            ok_mar = margin_accept(margin)
            _trace("raw_conf=", f"{conf:.4f}", "class=", CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else None,
                   "geo=", "accept" if ok_geo else "REJECT",
                   "margin=", f"{margin:.4f}", "margin_ok=", ok_mar)
            if not ok_geo or not ok_mar:
                idx, conf = None, 0.0
                result["unknown"] = True
            result.update(idx=idx, label=None if idx is None else CLASS_NAMES[idx],
                          label_en=None if idx is None else ENG_LABELS[idx],
                          conf=conf, bbox=bbox)
            _trace("result=", {k: result[k] for k in ("hand", "unknown", "idx", "label", "conf")})
            _t0 = time.perf_counter()
            h, w = frame.shape[:2]
            for lm in landmarks:
                cv2.circle(overlay, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            x0, y0, x1, y1 = bbox
            if result["unknown"]:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 215, 255), 2)
                cv2.putText(overlay, "غير معروف / مش واضح", (x0, max(16, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            else:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(overlay, f"{result['label_en']} {conf:.2f}", (x0, max(16, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            _trace("overlay_text=", f"{result['label_en']} {conf:.2f}" if not result['unknown'] else "لا نص",
                   "shown_any_conf_below_th", result["unknown"] is False and conf < CONF_THRESHOLD)
            _profile_time(prof, "draw", _t0)
    return overlay, result


def run_camera(source=0):
    """اختبار M4 في الطرفية: نافذة مباشرة + إخراج السجل. ESC للإيقاف."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("الكاميرا غير متاحة (source=%s)" % source)
    log().info("كاميرا مفتوحة — اهتزاز/توقف بإغلاقها أو ESC")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log().warning("إطار فاشل")
                continue
            ts = _next_ts_ms()
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
        """إطار واحد: يصرّف التزام/كلمة/إنهاء، ويعيد events (committed/word/finalized)."""
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
            _trace("feed", "hand+idx", "conf=", f"{conf:.4f}", "th=", self.conf_threshold,
                   "commit=", "YES" if events["committed"] else "no",
                   "votes=", dict(self._votes), "word=", repr(events["word"]))
        elif hand_seen:
            # Priority 1 — يد مرئية بلا تصنيف (رفض هندسي → 'غير معروف'): تُحدَّث last_hand_time فقط،
            # تُصفَّر الأصوات، بلا التزام حرف وبلا إنهاء كلمة مبكر حتى تختفي اليد أو يتوضح الوضع.
            _trace("feed", "hand-only(unknown)", "no_idx", "no_commit", "word=", repr(events["word"]))
            self.last_hand_time = ts
            self._votes = {}
            self._last_idx = None
            return events
        else:
            self._last_idx = None
            self._committed_idx = None
            if self.word and (self.last_hand_time == 0.0 or ts - self.last_hand_time >= self.silence_seconds):
                self._finalize()
                events["finalized"] = self.finalized_word
                events["word"] = ""
        return events

    def _finalize(self):
        """يقفل الكلمة الحالية في finalized_word ويصفّر المخزن."""
        self.finalized_word = "".join(CLASS_SYMS[i] for i in self.word)
        self.word = []

    def backspace(self):
        """تحكم يدوي: حذف آخر حرف مُلتزم في الكلمة الحية (فوري، بلا انتظار صمت).
        يعيد تهيئة بوابات عدم-التكرار حتى يمكن إعادة التزام نفس الحرف بعد الحذف."""
        if self.word:
            self.word.pop()
        self._committed_idx = None
        self._last_idx = None
        self._votes = {}

    def clear(self):
        """تحكم يدوي: مسح كل الكلمة الحية (فوري، بلا انتظار صمت)."""
        self.word = []
        self._committed_idx = None
        self._last_idx = None
        self._votes = {}

    def finalize_now(self):
        """تحكم يدوي: يغلق الكلمة الحالية ويصفّر المخزن — يعيدها نصاً أو None (فورية، بلا صمت)."""
        if not self.word:
            return None
        self._finalize()
        return self.finalized_word

    def force_commit(self, idx, conf=0.0):
        """تحكم يدوي: يُثبّت الحرف المعروض فوراً بلا debounce ولا اشتراط ثقة (P2).
        يحترم حد max_word (يغلق الكلمة تلقائياً عند الامتلاء)، ويُحدّث بوابة عدم-التكرار
        حتى لا يُعاد نفس الحرف تلقائياً. يعيد events كبنية feed()."""
        events = {"committed": None, "word": "".join(CLASS_SYMS[i] for i in self.word), "finalized": None}
        if self.word and len(self.word) >= self.max_word:
            self._finalize()
            events["finalized"] = self.finalized_word
        self.word.append(idx)
        self._committed_idx = idx
        self._last_idx = idx
        self._votes = {}
        events["committed"] = (idx, CLASS_SYMS[idx], conf or self._last_conf)
        events["word"] = "".join(CLASS_SYMS[i] for i in self.word)
        _trace("force_commit", "idx=", idx, "conf=", f"{conf:.4f}", "word=", repr(events["word"]),
               "finalized=", events["finalized"])
        return events


# ---------------------------------------------------------------- M7: Groq تصحيح

def _load_groq_key() -> str:
    """يقرأ GROQ_API_KEY (env/.env) — بلا مفتاح يعيد '' (ثم raw fallback)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def correct_word(raw: str, language: str = "ar"):
    """M7: Groq → تصحيح كلمة. language='ar'/'en' يبدّل الـ prompt (نفس وظيفة التصحيح للغتين).
    بلا مفتاح/إنترنت → raw بلا كسر. خلفي-متوافق (بدون وسيط = 'ar')."""
    key = _load_groq_key()
    if not key or not raw:
        log().info("Groq: مفتاح غير متاح — raw %r", raw)
        return {"text": raw, "source": "raw", "api": False}
    sys_prompt = (
        "You are an English spelling corrector. You receive a string of fingerspelled "
        "ASL letters (may be run-together, no spaces). Correct it into a proper English "
        "word/sentence and reply with only that."
        if language == "en" else
        "أنت مصحّح لغوي عربي. سيصلك نص من حروف إشارة عربية قد يكون هجاءً "
        "متصلاً بلا تشكيل. صحّحه إلى كلمة/جملة عربية سليمة وردّ بها فقط.")
    t0 = time.time()
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": raw},
                ],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=GROQ_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        log().info("Groq[%s]: %r -> %r (%.1fs)", language, raw, text, time.time() - t0)
        return {"text": text, "source": "groq", "api": True}
    except Exception as exc:
        log().warning("Groq[%s] فشل (%s) — raw %r", language, exc, raw)
        return {"text": raw, "source": "raw", "api": False}


# ---------------------------------------------------------------- M8: صوت

def synthesize_speech(text: str, lang: str = "ar"):
    """M8: gTTS → bytes mp3 (يُشغَّل عبر st.audio). lang='ar'/'en'. فشل → None بلا كسر."""
    if not text or not text.strip():
        return None
    try:
        from gtts import gTTS
        import io
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang=lang).write_to_fp(buf)
        buf.seek(0)
        log().info("TTS[%s]: %d bytes لـ %r", lang, buf.getbuffer().nbytes, text)
        return buf.getvalue()
    except Exception as exc:
        log().warning("gTTS[%s] فشل (%s) — نص بلا صوت", lang, exc)
        return None


# ---------------------------------------------------------------- خط أنابيب حي (مستهلك الواجهة)

class LivePipeline:
    """M4+M5+M6 (+M7 عبر correct_word) في نقاط جاهزة للواجهة. process_frame محسوب مرة/إطار."""

    def __init__(self, conf_threshold=CONF_THRESHOLD, debounce=DEBOUNCE_FRAMES,
                 silence_seconds=SILENCE_SECONDS, auto_correct=True, enable_profile=True):
        self.seq = SignSequencer(conf_threshold, debounce, silence_seconds)
        self.t0 = time.monotonic()
        self.auto_correct = auto_correct
        self._profiler = FrameProfiler(tag="ar") if enable_profile else None
        self.words = []   # كلمات مكتملة منفصلة — تُعرض وتُنطق بمسافات واضحة (P3)
        self.sentence = ""

    def push_word(self, raw: str):
        """P3: يسجّل كلمة مكتملة في قائمة words ويحقّق الجملة كاملة (تصحيح Groq على كل الكلمات
        مجتمعة + نطق الجملة بمسافات). أداة يدوية وتلقائية (الصمت) في مسار واحد. بلا كلمة → None."""
        if not raw:
            return None
        self.words.append(raw)
        self.sentence = " ".join(self.words)
        corrected = (correct_word(self.sentence) if self.auto_correct
                     else {"text": self.sentence, "source": "raw", "api": False})
        audio = synthesize_speech(corrected["text"])
        return {"raw": raw, "sentence": self.sentence, "corrected": corrected, "audio": audio}

    def reset_words(self):
        """P3: إعادة تعيين كلمات/جملة مكتملة (يستدعيها مسح الكل)."""
        self.words = []
        self.sentence = ""

    def update(self, frame):
        """إطار BGR → result + events + word + (corrected/audio عند الإنهاء) + error (أو None).
        timestamp زمن السلسلة نسبي (صمت)؛ timestamp الكاشف عالمي متزايد (Video mode)."""
        try:
            ts = time.monotonic() - self.t0
            prof = {}
            overlay, result = process_frame(frame, _next_ts_ms(), prof)
            events = self.seq.feed(result["hand"], result["idx"], result["conf"], ts)
            out = {**result, "events": events, "overlay": overlay, "timings": prof, "word": events["word"],
                   "corrected": None, "audio": None, "error": None,
                   "words": list(self.words), "sentence": self.sentence}
            if events["finalized"]:
                r = self.push_word(events["finalized"])
                if r:
                    out["corrected"], out["audio"] = r["corrected"], r["audio"]
                out["words"], out["sentence"] = list(self.words), self.sentence
            if self._profiler:
                self._profiler.note(prof)
            return out
        except Exception as exc:
            # أي فشل في مسار المعالجة (كاميرا/موديل/كاشف) → نتيجة خطأ صريحة بدل إسقاط التطبيق
            log().error("LivePipeline.update تعثر: %s", exc)
            return {"hand": False, "unknown": False, "idx": None, "label": None,
                    "label_en": None, "conf": 0.0, "bbox": None,
                    "events": {"committed": None, "word": "", "finalized": None},
                    "overlay": np.zeros((240, 320, 3), dtype=np.uint8), "word": "",
                    "timings": {}, "corrected": None, "audio": None,
                    "words": list(self.words), "sentence": self.sentence,
                    "error": f"تعذّرت معالجة الإطار ({exc}) — تحقق من الكاميرا والنموذج."}


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

    check("ar.margin.gate", margin_accept(0.08) is True and margin_accept(0.01) is False)
    check("crop.polarity.light", normalize_live(np.full((64, 64), 0.8, np.float32)).mean() > 0.5)
    check("crop.polarity.dark", normalize_live(np.full((64, 64), 0.2, np.float32)).mean() > 0.5)

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

    # ---- Priority 1: طبقة التحقق الهندسي (الأوضاع الأربعة المؤكدة) ----
    iu = synthetic_landmarks((1, 0, 0, 0), thumb=False)  # سبابة لأعلى → ياء (30)
    op = synthetic_landmarks((1, 1, 1, 1), thumb=True)  # كف مفتوح → سين (21)
    ft = synthetic_landmarks((0, 0, 0, 0), thumb=True)  # قبضة + إبهام جانبي → فاء (7)
    w3 = synthetic_landmarks((1, 1, 1, 0), thumb=False)  # W ثلاث أصابع → ثاء (25)
    fi_iu, fi_op, fi_ft, fi_w3 = (finger_features(x) for x in (iu, op, ft, w3))
    check("geo.features.counts",
          (fi_iu["num_ext"] == 1 and fi_iu["thumb"] is False and
           fi_op["num_ext"] == 4 and fi_op["thumb"] is True and
           fi_ft["num_ext"] == 0 and fi_ft["thumb"] is True and
           fi_w3["num_ext"] == 3 and fi_w3["thumb"] is False))
    check("geo.yyaa.index_up.accept", validate_geometry(iu, 30) is True)
    check("geo.seen_on_index.reject", validate_geometry(iu, 21) is False)
    check("geo.seen.palm.accept", validate_geometry(op, 21) is True)
    check("geo.thaaw_on_palm.tolerated", validate_geometry(op, 25) is True)  # فرق 1 ضمن التسامح
    check("geo.yyaa_on_palm.reject", validate_geometry(op, 30) is False)     # فرق 3 → رفض
    check("geo.faa.fist_thumb.accept", validate_geometry(ft, 7) is True)
    check("geo.seen_on_fist.reject", validate_geometry(ft, 21) is False)
    check("geo.thaaw.accept", validate_geometry(w3, 25) is True)
    check("geo.yyaa_on_w.reject", validate_geometry(w3, 30) is False)
    check("geo.unanchored.pass", validate_geometry(w3, 22) is True)  # شين بلا مرساة → لا رفض زائد

    seq2 = SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    seq2.feed(True, 2, 1.0, 0.10); seq2.feed(True, 2, 1.0, 0.20)
    seq2.feed(True, 2, 1.0, 0.30)
    ev = seq2.feed(True, None, 0.0, 5.00)    # يد مستمرة بلا تصنيف زمناً طويلاً
    check("seq.unknown.holds_word", ev["committed"] is None and ev["finalized"] is None
          and ev["word"] == CLASS_SYMS[2])
    ev = seq2.feed(False, None, 0.0, 7.00)   # اليد اختفت → إنهاء بعد الصمت
    check("seq.unknown.then_finalize", ev["finalized"] == CLASS_SYMS[2])

    # ---- P2: تثبيت يدوي + إغلاق كلمة يدوي (فورية، بلا debounce/صمت) ----
    seq3 = SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0, max_word=2)
    ev = seq3.force_commit(2, 0.0)            # ثبّت فوراً بلا votes
    check("seq.force_commit.fast", ev["committed"] is not None and ev["word"] == CLASS_SYMS[2])
    ev = seq3.force_commit(3, 0.0)            # امتلأ (max_word=2) → يغلق ويبدأ حرفاً جديداً
    check("seq.force_commit.full", ev["finalized"] is None and ev["word"] == CLASS_SYMS[2] + CLASS_SYMS[3])
    ev = seq3.force_commit(4, 0.0)
    check("seq.force_commit.overflow", ev["finalized"] == CLASS_SYMS[2] + CLASS_SYMS[3]
          and ev["word"] == CLASS_SYMS[4])
    ev = seq3.feed(True, 4, 1.0, 0.10); ev = seq3.feed(True, 4, 1.0, 0.20); ev = seq3.feed(True, 4, 1.0, 0.30)
    check("seq.force_commit.gate", ev["committed"] is None and ev["word"] == CLASS_SYMS[4])  # نفس الحرف لا يتكرر
    seq4 = SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    _ = seq4.feed(True, 2, 1.0, 0.10); _ = seq4.feed(True, 2, 1.0, 0.20); _ = seq4.feed(True, 2, 1.0, 0.30)
    raw = seq4.finalize_now()                 # إغلاق يدوي فوري بلا صمت
    check("seq.finalize_now", raw == CLASS_SYMS[2] and seq4.word == [])
    check("seq.finalize_now.empty", seq4.finalize_now() is None)

    # ---- P3: كلمات منفصلة + جملة بمسافات (تصحيح Groq على الجملة كاملة — بلا مفتاح → raw) ----
    pl3 = LivePipeline(auto_correct=False)
    r = pl3.push_word("سلام")
    check("p3.push.one", r and r["sentence"] == "سلام" and r["corrected"]["text"] == "سلام")
    r = pl3.push_word("عليكم")
    check("p3.push.sentence", r and r["sentence"] == "سلام عليكم"
          and r["corrected"]["source"] == "raw")
    check("p3.words.list", pl3.words == ["سلام", "عليكم"])
    pl3.reset_words()
    check("p3.reset", pl3.words == [] and pl3.sentence == "")
    pl3b = LivePipeline(auto_correct=False)
    r = pl3b.push_word("")
    check("p3.push.empty", r is None)

    # ---- Polish9: أداء/تواقيت كاشف اليد (قياس فعلي لا تخمين) ----
    seq_ts = [_next_ts_ms(), _next_ts_ms(), _next_ts_ms(), _next_ts_ms()]
    check("live.ts.monotonic", seq_ts == sorted(set(seq_ts)) and len(set(seq_ts)) == 4)
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    check("live.downscale.small.unchanged", downscale_live(small, 640) is small)
    ds = downscale_live(big, 640)
    check("live.downscale.big.max640", ds is not None and ds.shape[1] == 640 and ds.shape[0] == 360)
    prof = {}
    _, rp = process_frame(small, _next_ts_ms(), prof)
    check("live.profile.stages", rp["hand"] is False and prof.get("detect", 0) > 0
          and prof.get("copy", 0) > 0)
    fpr = FrameProfiler(window=2, tag="t")
    fpr.note({"detect": 1.0, "crop": 0.5}); fpr.note({"detect": 1.0, "crop": 0.5, "classify": 2.0})
    check("live.profile.window", fpr.count == 0)  # بعد window=2 يُصفَّر العدّاد

    check("groq.key", bool(_load_groq_key()))  # متاح أم لا — معلومة فقط
    check("tts.internet", synthesize_speech("اختبار") is not None)  # إنترنت؛ قابلة للفشل المسموح

    # ---- Polish2: حواف + انقطاع مفاجئ (محاكاة منفصلة بلا إنترنت حقيقي) ----
    _, rn = process_frame(None, 1)
    check("frame.empty.none", rn["hand"] is False)
    p_empty = LivePipeline().update(None)
    check("pipeline.empty_frame", p_empty["error"] is None and p_empty["hand"] is False
          and p_empty["word"] == "")

    e_lm = [type("LM", (), {"x": 0.01 + (i / 21) * 0.07, "y": 0.02 + ((i % 5) / 5) * 0.10})
            for i in range(21)]  # يد جزئية قرب زاوية الفريم
    e_patch, e_bbox = crop_hand_patch(black, e_lm)
    check("edge.hand.bbox", e_patch is not None and e_bbox is not None
          and e_bbox[0] >= 0 and e_bbox[2] <= 640 and e_bbox[3] <= 480)
    short = [e_lm[0]] * 5
    check("edge.short.landmarks", validate_geometry(short, 7) is True
          and finger_features(short)["num_ext"] == 0)

    _orig_post = requests.post
    try:
        requests.post = lambda *a, **k: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("محاكاة انقطاع الشبكة"))
        r_off = correct_word("سلم")
        check("groq.fallback.offline", r_off["source"] == "raw" and r_off["text"] == "سلم")
    finally:
        requests.post = _orig_post

    class _FakeResp401:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("401 unauthorized (مفتاح خاطئ)")
        def json(self):
            return {}

    _orig_post2 = requests.post
    try:
        requests.post = lambda *a, **k: _FakeResp401()
        r_bad = correct_word("سلم")
        check("groq.fallback.bad_key", r_bad["source"] == "raw" and r_bad["api"] is False)
    finally:
        requests.post = _orig_post2

    try:
        import gtts
        _orig_w = gtts.gTTS.write_to_fp
        def _boom(_self, _fp):
            raise OSError("محاكاة انقطاع gTTS")
        gtts.gTTS.write_to_fp = _boom
        check("tts.fallback.offline", synthesize_speech("اختبار") is None)
    finally:
        gtts.gTTS.write_to_fp = _orig_w

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _ = mp_vision.HandLandmarkerOptions
        check("mediapipe.tasks_api", True)
    except Exception:
        check("mediapipe.tasks_api", False)

    try:
        import engine_en as _EN
        _EN.en_checks(check)
    except Exception as exc:
        log().warning("engine_en فشل في selftest: %s", exc)
        check("en.module", False)

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
        _n = None
        _t = None
        _i = sys.argv.index("--train")
        if _i + 1 < len(sys.argv) and sys.argv[_i + 1].isdigit():
            _n = int(sys.argv[_i + 1])
        if "--target" in sys.argv:
            _ti = sys.argv.index("--target")
            if _ti + 1 < len(sys.argv):
                try:
                    _t = float(sys.argv[_ti + 1])
                except ValueError:
                    pass
        train_cnn(epochs=_n, target=_t)
        images, labels = load_dataset()
        idx, conf = classify(images[0][:, :, 0])
        print(f"[SELFTEST] classify smoke: idx={idx} en={ENG_LABELS[idx]} "
              f"true={ENG_LABELS[int(labels[0])]} match={(idx == int(labels[0]))}")
    elif "--camera" in sys.argv:
        run_camera()
    elif "--selftest" in sys.argv:
        sys.exit(run_selftest())
    else:
        print("استخدام: --fetch-data | --train [epochs] [--target VAL] | --camera | --selftest")