"""English (ASL) mode — منفصل تماماً عن Arabic (engine.py). لا يلمس أي كود/نموذج/بيانات عربية.
قاعدة: أي شك في التعارض = ملف/ثابت/نموذج منفصل، لا تعديل للموجود العربي.

قرارات موثقة:
- الداتا: Sign Language MNIST (Kaggle datamunge/sign-language-mnist) → data/asl_mnist.npz
  بنفس فورمات arasl.npz: (N,64,64,1) float [0,1] رمادي على خلفية سوداء، labels 0..23 كثيفة.
- الحجم: أعيد تحجيم 28×28 → 64×64 (INTER_CUBIC) للاتساق مع معمارية CNN العربية (ar._build_cnn) بلا تعديلها.
- الفئات: A–Y (بدون J, Z) = 24 فئة كثيفة، CLASS_EN_SYMS[i] = الحرف.
- المعمارية: نفس ar._build_cnn بالضبط (استيراد، لا إعادة كتابة).
- العتبات: نسخ قيم Arabic (BATCH 512 / CPU 6 / LR 2e-3 / TRAIN_FRAC 0.85 / MAX_EPOCHS 3 /
  TARGET_VAL 0.95 / CONF 0.90 / DEBOUNCE 5 / صمت 2.5) كـ *_EN مستقلة قابلة للتعديل.
- التحقق P1 إنجليزي: margin-check عام (top1 - top2 ≥ MARGIN_THRESHOLD_EN) فقط؛
  جدول SIGNPAT إنجليزي مؤجَّل (يُسجَّل في PROJECT_MAP [ENGLISH_SUPPORT] للتوسع لاحقاً).
"""

import sys
import time
import json
from collections import deque
from pathlib import Path

import numpy as np
import cv2

import engine as ar  # إعادة الاستخدام فقط — لا تعديل للعربي

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
DICT_EN_DIR = ASSETS_DIR / "dict_en"

ASL_MNIST_NPZ = DATA_DIR / "asl_mnist.npz"
MODEL_EN_PATH = MODELS_DIR / "cnn_en.pt"
CLASS_MAP_EN_PATH = MODELS_DIR / "class_map_en.json"

EN_CLASSES = 24
IMG_SIZE = ar.IMG_SIZE  # 64 — نفس بنية CNN العربية
EN_IMAGES = 27455 + 7172  # train + test من d.Set Sign Language MNIST

CLASS_EN_SYMS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # A–Y بدون J وZ (24 فئة)
CLASS_EN_NAMES = CLASS_EN_SYMS

# عتبات منفصلة (نسخ قيم Arabic كمقصد، قابلة للتوسع/التعديل لاحقاً)
BATCH_SIZE_EN = ar.BATCH_SIZE
CPU_THREADS_EN = ar.CPU_THREADS
LR_EN = ar.LR
TRAIN_FRAC_EN = ar.TRAIN_FRAC
MAX_EPOCHS_EN = 3
TARGET_VAL_ACC_EN = 0.95
CONF_THRESHOLD_EN = 0.90
DEBOUNCE_FRAMES_EN = 5
VOTE_WINDOW_EN = 7  # P4: نافذة انزلاقية للأغلبية — آخر 7 إطارات
VOTE_MAJORITY_EN = 5  # P4: التزام الحرف عندما يبلغ عدده 5 من آخر 7
SILENCE_SECONDS_EN = 2.5
MARGIN_THRESHOLD_EN = 0.05  # margin-check عام: فرق احتمال top1−top2 المقبول

# تحقق هندسي إنجليزي (SIGNPAT_EN، التجميعات فقط — لا حسابات حرف-حرف):
#   قياس على asl_mnist.npz نفسها أثبت أن رسومات MNIST منحازة منهجياً (B≈2 عمود لا 4،
#   A≈3 لا 0، W≈4 لا 3) فأعمدة الأصابع لا تصلح أساساً صارماً لوحدها؛ الملمس الموثوق الوحيد
#   منها الفصل «أصابع مرفوعة (topY≈0–3)» عن «قبضة/مطوية (topY≈6–9)». لذلك نضبط بوابات
#   عدد الأصابع فقط (فئة الأجزاء الأربعة) من القيم الكنسية الكنسية لـ ASL، بتسامح ±1
#   (نفس قاعدة العربية)، والإبهام "any" (إبهامه غير موثوق القياس في الرسومات). تم التعميد
#   على حروف يدويّة غامضة (C, G, O, P, Q, X) → لا قيد (margin-check وحده). التحقق
#   الصارم حرف-بحرف يُؤجَّل إلى اختبار الكاميرا (EN-6).
SIGNPAT_EN = {
    0: {"fingers": 0, "thumb": "any"},    # A — قبضة + إبهام مطوي
    1: {"fingers": 4, "thumb": "any"},    # B — أربعة أصابع ممدودة، الإبهام مطوي
    3: {"fingers": 1, "thumb": "any"},    # D — سبابة فقط
    4: {"fingers": 0, "thumb": "any"},    # E — قبضة (جميعها مطوية)
    5: {"fingers": 3, "thumb": "any"},    # F — وسطى+بنصر+خنصر ممدودة (سبابة تلامس الإبهام)
    8: {"fingers": 1, "thumb": "any"},    # I — خنصر فقط
    9: {"fingers": 2, "thumb": "any"},    # K — سبابة+وسطى ممدودتان
    10: {"fingers": 1, "thumb": "any"},   # L — سبابة فقط + إبهام
    11: {"fingers": 0, "thumb": "any"},   # M — ثلاث مطوية تحت الإبهام
    12: {"fingers": 0, "thumb": "any"},   # N — إصبعان مطويان
    16: {"fingers": 2, "thumb": "any"},   # R — سبابة+وسطى متقاطعتان
    17: {"fingers": 0, "thumb": "any"},   # S — قبضة كاملة
    18: {"fingers": 0, "thumb": "any"},   # T — قبضة + إبهام بين سبابة ووسطى
    19: {"fingers": 2, "thumb": "any"},   # U — سبابة+وسطى
    20: {"fingers": 2, "thumb": "any"},   # V — سبابة+وسطى متباعدتان
    21: {"fingers": 3, "thumb": "any"},   # W — سبابة+وسطى+بنصر
    23: {"fingers": 1, "thumb": "any"},   # Y — خنصر + إبهام
}

KAGGLE_DATASET = "datamunge/sign-language-mnist"
KAGGLE_TRAIN_REL = "sign_mnist_train/sign_mnist_train.csv"
KAGGLE_TEST_REL = "sign_mnist_test/sign_mnist_test.csv"


# ---------------------------------------------------------------- البيانات (Step 1)

def load_en_dataset(force_rebuild: bool = False):
    """(images (N,64,64,1) float [0,1], labels (N,) كثيفة 0..23)."""
    if not ASL_MNIST_NPZ.exists():
        fetch_en_data(force_rebuild)
    z = np.load(ASL_MNIST_NPZ)
    return z["images"], z["labels"]


def _read_mnist_csv(path: Path):
    """CSV خام (28×28) → (x, كثافة 0..23) (يزيل فجوة J)."""
    import pandas as pd
    df = pd.read_csv(path)
    y = df["label"].to_numpy(dtype=np.int64)
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28)
    dense = np.where(y > 9, y - 1, y).astype(np.int64)  # يزيل فجوة J(label 9)
    return x, dense


def fetch_en_data(force: bool = False) -> None:
    """Kaggle Sign Language MNIST → data/asl_mnist.npz + assets/dict_en/*.png + models/class_map_en.json."""
    if ASL_MNIST_NPZ.exists() and not force:
        ar.log().info("asl_mnist.npz موجود مسبقاً")
        if (len(list(DICT_EN_DIR.glob("*.png"))) != EN_CLASSES
                or not CLASS_MAP_EN_PATH.exists()):
            z = np.load(ASL_MNIST_NPZ)
            _write_dict_en(z["images"], z["labels"])
            _write_class_map_en()
            ar.log().info("أُعيد بناء dict_en/class_map_en من npz الموجود")
        return
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub غير مثبَّت — شغّل: pip install kagglehub (يقرأ ~/.kaggle/access_token)") from exc

    try:
        base = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise RuntimeError(
            "فشل تنزيل الداتا من Kaggle — تأكد من internet و~/.kaggle/access_token "
            f"(الصحيح: KGAT_...): {exc}") from exc
    tr_csv = base / KAGGLE_TRAIN_REL
    te_csv = base / KAGGLE_TEST_REL
    if not (tr_csv.exists() and te_csv.exists()):
        raise RuntimeError(f"ملفات CSV غير موجودة في {base}: {tr_csv}, {te_csv}")

    xt, yt = _read_mnist_csv(tr_csv)
    xe, ye = _read_mnist_csv(te_csv)
    x = np.concatenate([xt, xe]).astype(np.float32)
    y = np.concatenate([yt, ye]).astype(np.int64)
    ar.log().info("ASL MNIST: %d صورة (train %d + test %d)، فئات %d",
                  x.shape[0], xt.shape[0], xe.shape[0], len(np.unique(y)))

    resized = np.empty((x.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    for i in range(x.shape[0]):
        up = cv2.resize(x[i].astype(np.uint8), (IMG_SIZE, IMG_SIZE),
                        interpolation=cv2.INTER_CUBIC)  # uint8 ← لا تجاوز نطاق
        resized[i, :, :, 0] = up.astype(np.float32) / 255.0  # نفس تطبيع arasl.npz ([0,1])

    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(ASL_MNIST_NPZ, images=resized, labels=y)
    ar.log().info("حُفظ: %s (%d MB)", ASL_MNIST_NPZ.name,
                  (Path(ASL_MNIST_NPZ).stat().st_size // (1 << 20)))

    _write_dict_en(resized, y)
    _write_class_map_en()
    print(f"[SELFTEST] fetch_en: PASS shape={resized.shape} classes={len(np.unique(y))} "
          f"dict_en_pngs={len(list(DICT_EN_DIR.glob('*.png')))}")


def _write_dict_en(images, labels) -> None:
    """ملف مرجعي واحد لكل فئة (أقرب صورة إلى مركز الفئة) في assets/dict_en/class_XX.png."""
    DICT_EN_DIR.mkdir(parents=True, exist_ok=True)
    for c in range(EN_CLASSES):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        mean = images[idx].mean(axis=0)
        best = idx[int(np.abs(images[idx] - mean).sum(axis=(1, 2, 3)).argmin())]
        png = DICT_EN_DIR / f"class_{c:02d}.png"
        cv2.imwrite(str(png), (images[best][:, :, 0] * 255).astype(np.uint8))


def _write_class_map_en() -> None:
    """يكتب class_map_en.json (حرف EN لكل صنف كثيف) — يُقرأ في تبويب القاموس الإنجليزي."""
    MODELS_DIR.mkdir(exist_ok=True)
    CLASS_MAP_EN_PATH.write_text(
        json.dumps({str(i): {"sym": CLASS_EN_SYMS[i], "name": CLASS_EN_NAMES[i]}
                    for i in range(EN_CLASSES)}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- التدريب (Step 2)

def train_en() -> None:
    """نفس معمارية/Pipeline التدريب العربي على بيانات ASL → models/cnn_en.pt (لا يمس العربي)."""
    import torch
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(CPU_THREADS_EN)
    images, labels = load_en_dataset()
    ar.log().info("داتا EN: %s | فئات %d", images.shape, len(np.unique(labels)))
    x = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)  # (N,1,64,64)
    x_train, x_val, y_train, y_val = train_test_split(
        x, labels, test_size=1 - TRAIN_FRAC_EN, stratify=labels, random_state=42)

    model = ar._build_cnn()  # نفس المعمارية العربية بالضبط — استيراد لا إعادة كتابة
    if model.head[1].out_features != EN_CLASSES:
        # رأس واحد فقط بعدد فئات الإنجليزي (بقية المعمارية مطابقة تماماً)
        model.head = torch.nn.Sequential(
            model.head[0], torch.nn.Linear(64 * 8 * 8, EN_CLASSES))

    if MODEL_EN_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_EN_PATH, weights_only=True))
            ar.log().info("استئناف EN من %s", MODEL_EN_PATH.name)
        except Exception as exc:
            ar.log().warning("تعذر تحميل %s (%s) — بدء من الصفر", MODEL_EN_PATH.name, exc)

    opt = torch.optim.Adam(model.parameters(), lr=LR_EN)
    crit = torch.nn.CrossEntropyLoss()
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    best_acc = -1.0
    reached = False
    train_n = x_train.shape[0]
    for ep in range(1, MAX_EPOCHS_EN + 1):
        model.train()
        total = correct = 0
        perm = torch.randperm(train_n)
        ag_rng = np.random.default_rng(21 + ep)
        t0 = time.time()
        for i in range(0, train_n, BATCH_SIZE_EN):
            ids = perm[i:i + BATCH_SIZE_EN]
            xb = torch.from_numpy(ar._augment_batch(x_train[ids].numpy(), ag_rng)).float()
            yb = y_train[ids]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(ids)
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            vcorrect = 0
            for i in range(0, x_val.shape[0], BATCH_SIZE_EN):
                vcorrect += (model(x_val[i:i + BATCH_SIZE_EN]).argmax(1)
                             == y_val[i:i + BATCH_SIZE_EN]).sum().item()
        val_acc = vcorrect / y_val.shape[0]
        ar.log().info("epoch_en %02d | train %.4f | val %.4f | %.1fs",
                      ep, train_acc, val_acc, time.time() - t0)
        if val_acc > best_acc:
            best_acc = val_acc
            MODELS_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_EN_PATH)
            _write_class_map_en()
            ar.log().info("checkpoint أفضل: val %.4f → %s", best_acc, MODEL_EN_PATH.name)
        if val_acc >= TARGET_VAL_ACC_EN:
            reached = True
            break

    ar.log().info("تعليم EN انتهى: best val %.4f → %s", best_acc, MODEL_EN_PATH.name)
    print(f"[SELFTEST] train_en: {'PASS' if reached else 'FAIL'} best_val_acc={best_acc:.4f} "
          f"target={TARGET_VAL_ACC_EN}")


# ---------------------------------------------------------------- التشغيل (Step 3)

def _get_model_en():
    """يحمّل cnn_en.pt مرة واحدة (singleton، كأنماط العربي).
    مفقود/تالف → RuntimeError برسالة إجرائية واضحة (لا traceback خام)."""
    model = getattr(_get_model_en, "cache", None)
    if model is None:
        import torch
        if not MODEL_EN_PATH.exists():
            raise ar.RuntimeError(f"نموذج EN مفقود: {MODEL_EN_PATH.name} — شغّل: python engine_en.py --train-en")
        try:
            model = ar._build_cnn()
            if model.head[1].out_features != EN_CLASSES:
                model.head = torch.nn.Sequential(
                    model.head[0], torch.nn.Linear(64 * 8 * 8, EN_CLASSES))
            model = model.eval()
            model.load_state_dict(torch.load(MODEL_EN_PATH, weights_only=True))
        except Exception as exc:
            raise ar.RuntimeError(
                f"تعذر تحميل {MODEL_EN_PATH.name} ({exc}) — أعد التدريب: python engine_en.py --train-en") from exc
        _get_model_en.cache = model
    return model


def classify_en_softmax(patch: np.ndarray):
    """patch float32 (64,64)/(64,64,1) في [0,1] → (idx كثيف 0..23, conf top1, margin top1−top2)."""
    import torch
    p = patch if patch.ndim == 2 else patch[:, :, 0]
    x = torch.from_numpy(p.astype(np.float32)[None, None]).float()
    with torch.no_grad():
        probs = torch.softmax(_get_model_en()(x), dim=1)[0].numpy()
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    return idx, float(probs[idx]), float(probs[idx] - probs[order[1]])


def margin_accept(margin: float) -> bool:
    """margin-check العام: يقبل فقط التصنيفات التي تفوق فيها top1 توب2 بـ >= MARGIN_THRESHOLD_EN."""
    return float(margin) >= MARGIN_THRESHOLD_EN


def process_frame_en(frame, ts_ms: int, prof=None):
    """مثل process_frame العربي لكن بالنموذج/العلامات الإنجليزية + margin-check بدل SIGNPAT.
    result: {hand, unknown, idx, label, label_en, conf, bbox}.
    prof (dict اختياري): أزمنة المراحل (detect/crop/classify/geometry/draw).
    إطار None/فارغ → نتيجة فارغة (لا crash)."""
    result = {"hand": False, "unknown": False, "idx": None, "label": None,
              "label_en": None, "conf": 0.0, "bbox": None}
    if frame is None or getattr(frame, "size", 0) == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8), result
    _t0 = time.perf_counter()
    overlay = frame.copy()
    ar._profile_time(prof, "copy", _t0)
    _t0 = time.perf_counter()
    landmarks = ar.detect_hand(frame, ts_ms)  # HandLandmarker مشترك (فاصل متوافق بين اللغتين)
    ar._profile_time(prof, "detect", _t0)
    if landmarks:
        result["hand"] = True
        _t0 = time.perf_counter()
        patch, bbox = ar.crop_hand_patch(frame, landmarks, pad=ar.HAND_PAD)
        ar._profile_time(prof, "crop", _t0)
        if patch is not None:
            _t0 = time.perf_counter()
            idx, conf, margin = classify_en_softmax(patch)
            ar._profile_time(prof, "classify", _t0)
            _t0 = time.perf_counter()
            ok_geo = ar.validate_geometry(landmarks, idx, patterns=SIGNPAT_EN)
            ok_mar = margin_accept(margin)
            ar._trace("raw_conf=", f"{conf:.4f}", "class=", CLASS_EN_SYMS[idx] if 0 <= idx < EN_CLASSES else None,
                      "margin=", f"{margin:.4f}", "margin_ok=", ok_mar,
                      "geo=", "accept" if ok_geo else "REJECT")
            if not ok_mar or not ok_geo:
                idx, conf = None, 0.0
                result["unknown"] = True
            ar._profile_time(prof, "geometry", _t0)
            result.update(idx=idx, label=None if idx is None else CLASS_EN_SYMS[idx],
                          label_en=None if idx is None else CLASS_EN_SYMS[idx],
                          conf=conf, bbox=bbox)
            ar._trace("result=", {k: result[k] for k in ("hand", "unknown", "idx", "label", "conf")})
            _t0 = time.perf_counter()
            h, w = frame.shape[:2]
            for lm in landmarks:
                cv2.circle(overlay, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            x0, y0, x1, y1 = bbox
            if result["unknown"]:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 215, 255), 2)
                cv2.putText(overlay, "unknown / unclear", (x0, max(16, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            else:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(overlay, f"{result['label_en']} {conf:.2f}", (x0, max(16, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ar._profile_time(prof, "draw", _t0)
    return overlay, result


class SignSequencerEN:
    """نسخة من SignSequencer العربي برموز إنجليزية — التمييز الثلاثي (حرف/غير معروف/لا يد).
    P4: الالتزام بأغلبية نافذة انزلاقية (افتراضياً 5 من آخر 7) — يصمد أمام رمشة إطارٍ آخر؛
    لا تُحسب سوى الإطارات ذات الثقة ≥ العتبة، فلا التزامٌ تلقائيٌ بثقة أدنى."""

    def __init__(self, conf_threshold=CONF_THRESHOLD_EN, debounce=DEBOUNCE_FRAMES_EN,
                 silence_seconds=SILENCE_SECONDS_EN, max_word=ar.MAX_WORD_LEN,
                 vote_window=None, vote_majority=None):
        self.conf_threshold = conf_threshold
        self.debounce = debounce
        self.vote_window = vote_window if vote_window is not None else debounce
        self.vote_majority = vote_majority if vote_majority is not None else debounce
        self.silence_seconds = silence_seconds
        self.max_word = max_word
        self._wins = deque(maxlen=self.vote_window)
        self._last_idx = None
        self._last_conf = 0.0
        self._committed_idx = None
        self.word = []
        self.last_hand_time = 0.0
        self.finalized_word = None

    def feed(self, hand_seen: bool, idx=None, conf=0.0, ts=0.0):
        """إطار واحد: يصرّف التزام/كلمة/إنهاء، ويعيد events (committed/word/finalized)."""
        events = {"committed": None, "word": "".join(CLASS_EN_SYMS[i] for i in self.word),
                  "finalized": None}
        if hand_seen and idx is not None:
            self.last_hand_time = ts
            if conf >= self.conf_threshold:
                self._wins.append(idx)
                self._last_conf = conf
            else:
                self._wins.append(None)
            self._last_idx = idx
            n = self._wins.count(idx)
            if idx != self._committed_idx and n >= self.vote_majority:
                if len(self.word) >= self.max_word:
                    self._finalize()
                self.word.append(idx)
                events["committed"] = (idx, CLASS_EN_SYMS[idx], self._last_conf)
                self._committed_idx = idx
                self._wins.clear()
            events["word"] = "".join(CLASS_EN_SYMS[i] for i in self.word)
            ar._trace("feed_en", "conf=", f"{conf:.4f}", "th=", self.conf_threshold,
                      "commit=", "YES" if events["committed"] else "no",
                      "win=", list(self._wins), "word=", repr(events["word"]))
        elif hand_seen:
            ar._trace("feed_en", "hand-only(unknown)", "no_commit")
            self.last_hand_time = ts
            self._wins.clear()
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
        self.finalized_word = "".join(CLASS_EN_SYMS[i] for i in self.word)
        self.word = []

    def backspace(self):
        """تحكم يدوي: حذف آخر حرف مُلتزم (فوري، بلا انتظار صمت)."""
        if self.word:
            self.word.pop()
        self._committed_idx = None
        self._last_idx = None
        self._wins.clear()

    def clear(self):
        """تحكم يدوي: مسح كل الكلمة الحية (فوري، بلا انتظار صمت)."""
        self.word = []
        self._committed_idx = None
        self._last_idx = None
        self._wins.clear()

    def finalize_now(self):
        """تحكم يدوي: يغلق الكلمة الحالية ويصفّر المخزن — يعيدها نصاً أو None (فورية، بلا صمت)."""
        if not self.word:
            return None
        self._finalize()
        return self.finalized_word

    def force_commit(self, idx, conf=0.0):
        """تحكم يدوي: يُثبّت الحرف المعروض فوراً بلا debounce ولا اشتراط ثقة (P2).
        يحترم حد max_word ويُحدّث بوابة عدم-التكرار. يعيد events كبنية feed()."""
        events = {"committed": None, "word": "".join(CLASS_EN_SYMS[i] for i in self.word), "finalized": None}
        if self._committed_idx == idx:
            # بوابة ضد ازدواج «تثبيت»: نفس الحرف مُلتزم ولم يُصفَّر → إلغاء صامت (ضغطة = حرف واحد).
            return events
        if self.word and len(self.word) >= self.max_word:
            self._finalize()
            events["finalized"] = self.finalized_word
        self.word.append(idx)
        self._committed_idx = idx
        self._last_idx = idx
        self._wins.clear()
        events["committed"] = (idx, CLASS_EN_SYMS[idx], conf or self._last_conf)
        events["word"] = "".join(CLASS_EN_SYMS[i] for i in self.word)
        return events


class LivePipelineEN:
    """خط أنابيب إنجليزي كامل: detect+crop → CNN+margin → تسلسل → Groq(en) → gTTS(en)."""

    def __init__(self, conf_threshold=CONF_THRESHOLD_EN, debounce=DEBOUNCE_FRAMES_EN,
                 silence_seconds=SILENCE_SECONDS_EN, auto_correct=True, enable_profile=True,
                 vote_window=VOTE_WINDOW_EN, vote_majority=VOTE_MAJORITY_EN):
        self.seq = SignSequencerEN(conf_threshold, debounce, silence_seconds,
                                   vote_window=vote_window, vote_majority=vote_majority)
        self.t0 = time.monotonic()
        self.auto_correct = auto_correct
        self._profiler = ar.FrameProfiler(tag="en") if enable_profile else None
        self.words = []
        self.sentence = ""

    def push_word(self, raw: str):
        """P3: يسجّل كلمة مكتملة ويحقّق الجملة كاملة (تصحيح Groq لكل الكلمات + نطق بمسافات)."""
        if not raw:
            return None
        self.words.append(raw)
        self.sentence = " ".join(self.words)
        corrected = (ar.correct_word(self.sentence, language="en") if self.auto_correct
                     else {"text": self.sentence, "source": "raw", "api": False})
        audio = ar.synthesize_speech(corrected["text"], lang="en")
        return {"raw": raw, "sentence": self.sentence, "corrected": corrected, "audio": audio}

    def reset_words(self):
        """P3: إعادة تعيين كلمات/جملة مكتملة."""
        self.words = []
        self.sentence = ""

    def update(self, frame):
        """إطار BGR → result + events + word + (corrected/audio عند الإنهاء) + error (أو None).
        timestamp الكاشف عالمي متزايد (Video mode، يشترك مع العربي في HandLandmarker واحد)."""
        try:
            ts = time.monotonic() - self.t0
            prof = {}
            dts = ar._next_ts_ms()
            overlay, result = process_frame_en(frame, dts, prof)
            events = self.seq.feed(result["hand"], result["idx"], result["conf"], ts)
            if ar._TRACE:
                print("[LIVE-VOTE] EN feed(hand=%s idx=%s conf=%.4f) window=%d/%d "
                      "committed=%s unknown=%s ts_ms=%d" %
                      (result["hand"], result["idx"], result["conf"],
                       len(self.seq._wins), self.seq.vote_window,
                       events["committed"], result["unknown"], dts), flush=True)
            out = {**result, "events": events, "overlay": overlay, "timings": prof,
                   "word": events["word"], "corrected": None, "audio": None, "error": None,
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
            ar.log().error("LivePipelineEN.update تعثر: %s", exc)
            return {"hand": False, "unknown": False, "idx": None, "label": None,
                    "label_en": None, "conf": 0.0, "bbox": None,
                    "events": {"committed": None, "word": "", "finalized": None},
                    "overlay": np.zeros((240, 320, 3), dtype=np.uint8), "word": "",
                    "timings": {}, "corrected": None, "audio": None,
                    "words": list(self.words), "sentence": self.sentence,
                    "error": f"Failed to process frame ({exc}) — check camera and EN model."}


# ---------------------------------------------------------------- الفحص الذاتي (EN)

def en_checks(check) -> None:
    """فحوص en.* تُدمج في --selftest العربي عبر engine.run_selftest (دالة check تُمرَّر)."""
    check("en.artifacts.npz", ASL_MNIST_NPZ.exists())
    check("en.artifacts.model", MODEL_EN_PATH.exists())
    check("en.artifacts.class_map", CLASS_MAP_EN_PATH.exists())
    check("en.artifacts.dict_en", len(list(DICT_EN_DIR.glob("*.png"))) == EN_CLASSES)

    # ---- SIGNPAT_EN (تجميعات كنسية قياساً على البيانات — تعليق في TABLE) ----
    check("en.signpat.values",
          SIGNPAT_EN[1]["fingers"] == 4 and SIGNPAT_EN[0]["fingers"] == 0
          and SIGNPAT_EN[21]["fingers"] == 3 and SIGNPAT_EN[19]["fingers"] == 2
          and all(r["thumb"] == "any" for r in SIGNPAT_EN.values()))
    unanchored = {2, 6, 7, 13, 14, 15, 22}  # C, G, H, O, P, Q, X
    check("en.signpat.no_ambig", not (unanchored & set(SIGNPAT_EN))
          and set(SIGNPAT_EN) <= set(range(EN_CLASSES)))
    _fist = ar.synthetic_landmarks((0, 0, 0, 0), thumb=False)
    _one = ar.synthetic_landmarks((1, 0, 0, 0), thumb=True)
    _two = ar.synthetic_landmarks((1, 1, 0, 0), thumb=False)
    _three = ar.synthetic_landmarks((1, 1, 1, 0), thumb=False)
    _open = ar.synthetic_landmarks((1, 1, 1, 1), thumb=True)
    check("en.signpat.apply.open_B", ar.validate_geometry(_open, 1, SIGNPAT_EN) is True)
    check("en.signpat.apply.fist_on_B.reject", ar.validate_geometry(_fist, 1, SIGNPAT_EN) is False)
    check("en.signpat.apply.fist_on_S", ar.validate_geometry(_fist, 17, SIGNPAT_EN) is True)
    check("en.signpat.apply.two_on_V", ar.validate_geometry(_two, 20, SIGNPAT_EN) is True)
    check("en.signpat.apply.open_on_V.reject", ar.validate_geometry(_open, 20, SIGNPAT_EN) is False)
    check("en.signpat.apply.one_on_D", ar.validate_geometry(_one, 3, SIGNPAT_EN) is True)
    check("en.signpat.apply.unanchored.pass",
          ar.validate_geometry(_open, 2, SIGNPAT_EN) is True
          and ar.validate_geometry(_fist, 7, SIGNPAT_EN) is True)

    try:
        images, labels = load_en_dataset()
        check("en.data.shape", images.shape == (EN_IMAGES, IMG_SIZE, IMG_SIZE, 1)
              and len(np.unique(labels)) == EN_CLASSES)
        idx, conf, margin = classify_en_softmax(images[0][:, :, 0])
        check("en.classify.smoke", 0 <= idx < EN_CLASSES and 0.0 <= conf <= 1.0 and 0.0 <= margin <= 1.0)
        check("en.margin.gate", margin_accept(0.08) is True and margin_accept(0.01) is False)
    except Exception as exc:
        ar.log().error("en_checks data/classify تعثر: %s", exc)
        check("en.data.shape", False)
        check("en.classify.smoke", False)
        check("en.margin.gate", False)

    check("en.polarity.light", ar.normalize_live(np.full((64, 64), 0.8, np.float32)).mean() > 0.5)
    check("en.polarity.dark", ar.normalize_live(np.full((64, 64), 0.2, np.float32)).mean() > 0.5)

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    _, res = process_frame_en(black, 1)
    check("en.frame.no_hand", res["hand"] is False)

    seq = SignSequencerEN(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    seq.feed(True, 0, 1.0, 0.10); seq.feed(True, 0, 1.0, 0.20); seq.feed(True, 0, 1.0, 0.30)
    ev = seq.feed(True, 1, 1.0, 0.40); seq.feed(True, 1, 1.0, 0.50); seq.feed(True, 1, 1.0, 0.60)
    seq.feed(True, 2, 1.0, 0.70); seq.feed(True, 2, 1.0, 0.80); ev = seq.feed(True, 2, 1.0, 0.90)
    check("en.seq.word", ev["word"] == "ABC")
    ev = seq.feed(True, None, 0.0, 5.00)          # يد غير معروف وقتاً طويلاً → لا إنهاء مبكر
    check("en.seq.unknown.holds", ev["finalized"] is None and ev["word"] == "ABC")
    ev = seq.feed(False, None, 0.0, 7.00)         # اختفاء اليد → إنهاء بعد الصمت
    check("en.seq.finalize", ev["finalized"] == "ABC")

    seq2 = SignSequencerEN(conf_threshold=0.0, debounce=3, silence_seconds=1.0, max_word=2)
    ev = seq2.force_commit(0, 0.0)
    check("en.seq.force_commit.fast", ev["committed"] is not None and ev["word"] == "A")
    ev = seq2.force_commit(1, 0.0)
    check("en.seq.force_commit.full", ev["finalized"] is None and ev["word"] == "AB")
    ev = seq2.force_commit(2, 0.0)
    check("en.seq.force_commit.overflow", ev["finalized"] == "AB" and ev["word"] == "C")
    seq3 = SignSequencerEN(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    seq3.feed(True, 2, 1.0, 0.10); seq3.feed(True, 2, 1.0, 0.20); seq3.feed(True, 2, 1.0, 0.30)
    raw = seq3.finalize_now()
    check("en.seq.finalize_now", raw == "C" and seq3.word == [])
    check("en.seq.finalize_now.empty", seq3.finalize_now() is None)

    pl3 = LivePipelineEN(auto_correct=False)
    r = pl3.push_word("WORLD")
    check("en.p3.push.one", r and r["sentence"] == "WORLD" and r["corrected"]["text"] == "WORLD")
    r = pl3.push_word("PEACE")
    check("en.p3.push.sentence", r and r["sentence"] == "WORLD PEACE"
          and r["corrected"]["source"] == "raw")
    check("en.p3.words.list", pl3.words == ["WORLD", "PEACE"])
    pl3.reset_words()
    check("en.p3.reset", pl3.words == [] and pl3.sentence == "")

    _, res_en = process_frame_en(None, 1)
    check("en.frame.empty", res_en["hand"] is False)
    out_en0 = LivePipelineEN().update(None)
    check("en.pipeline.empty_frame", out_en0["error"] is None and out_en0["hand"] is False)

    seq_ar2 = ar.SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0)
    seq_ar2.feed(True, 30, 1.0, 0.10); seq_ar2.feed(True, 30, 1.0, 0.20)
    ev_ar = seq_ar2.feed(True, 30, 1.0, 0.30)
    check("switch.ar_active", ev_ar["word"] == ar.CLASS_SYMS[30])
    check("switch.no_crosstalk", ar.SignSequencer(conf_threshold=0.0, debounce=3, silence_seconds=1.0).word == []
          and SignSequencerEN(conf_threshold=0.0, debounce=3, silence_seconds=1.0).word == [])


# ---------------------------------------------------------------- CLI

def main() -> None:
    """CLI: --fetch-en | --train-en | --selftest-en (رسائل ودية عند الفشل)."""
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ar.setup_logging()
    if "--fetch-en" in sys.argv:
        try:
            fetch_en_data()
        except Exception as exc:
            print(f"[SELFTEST] fetch_en: FAIL — {exc}")
            sys.exit(1)
    elif "--train-en" in sys.argv:
        try:
            train_en()
        except Exception as exc:
            print(f"[SELFTEST] train_en: FAIL — {exc}")
            sys.exit(1)
    elif "--selftest-en" in sys.argv:
        ok = True
        def check(name, cond):
            nonlocal ok
            print(f"[SELFTEST] {name}: {'PASS' if cond else 'FAIL'}")
            ok = ok and bool(cond)
        en_checks(check)
        print(f"[SELFTEST] overall_en: {'ALL PASS' if ok else 'SOME FAIL'}")
        sys.exit(0 if ok else 1)
    else:
        print("استخدام (engine_en): --fetch-en | --train-en | --selftest-en")


if __name__ == "__main__":
    main()