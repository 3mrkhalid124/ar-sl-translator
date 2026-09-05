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
SILENCE_SECONDS_EN = 2.5
MARGIN_THRESHOLD_EN = 0.05  # margin-check عام: فرق احتمال top1−top2 المقبول

KAGGLE_DATASET = "datamunge/sign-language-mnist"
KAGGLE_TRAIN_REL = "sign_mnist_train/sign_mnist_train.csv"
KAGGLE_TEST_REL = "sign_mnist_test/sign_mnist_test.csv"

GITIGNORE_ENTRIES = ["asl_mnist.npz", "cnn_en.pt", "class_map_en.json", "dict_en/"]


# ---------------------------------------------------------------- البيانات (Step 1)

def load_en_dataset(force_rebuild: bool = False):
    """(images (N,64,64,1) float [0,1], labels (N,) كثيفة 0..23)."""
    if not ASL_MNIST_NPZ.exists():
        fetch_en_data(force_rebuild)
    z = np.load(ASL_MNIST_NPZ)
    return z["images"], z["labels"]


def _read_mnist_csv(path: Path):
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
        return
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub غير مثبَّت — شغّل: pip install kagglehub (يقرأ ~/.kaggle/access_token)") from exc

    base = Path(kagglehub.dataset_download(KAGGLE_DATASET))
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

    best_acc = 0.0
    reached = False
    train_n = x_train.shape[0]
    for ep in range(1, MAX_EPOCHS_EN + 1):
        model.train()
        total = correct = 0
        perm = torch.randperm(train_n)
        t0 = time.time()
        for i in range(0, train_n, BATCH_SIZE_EN):
            ids = perm[i:i + BATCH_SIZE_EN]
            opt.zero_grad()
            out = model(x_train[ids])
            loss = crit(out, y_train[ids])
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == y_train[ids]).sum().item()
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
        if val_acc >= TARGET_VAL_ACC_EN:
            reached = True
            break

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_EN_PATH)
    _write_class_map_en()
    ar.log().info("تعليم EN انتهى: best val %.4f → %s", best_acc, MODEL_EN_PATH.name)
    print(f"[SELFTEST] train_en: {'PASS' if reached else 'FAIL'} best_val_acc={best_acc:.4f} "
          f"target={TARGET_VAL_ACC_EN}")


# ---------------------------------------------------------------- التشغيل (Step 3)

def _get_model_en():
    """يحمّل cnn_en.pt مرة واحدة (singleton، كأنماط العربي)."""
    model = getattr(_get_model_en, "cache", None)
    if model is None:
        import torch
        model = ar._build_cnn()
        if model.head[1].out_features != EN_CLASSES:
            model.head = torch.nn.Sequential(
                model.head[0], torch.nn.Linear(64 * 8 * 8, EN_CLASSES))
        model = model.eval()
        model.load_state_dict(torch.load(MODEL_EN_PATH, weights_only=True))
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


def process_frame_en(frame, ts_ms: int):
    """مثل process_frame العربي لكن بالنموذج/العلامات الإنجليزية + margin-check بدل SIGNPAT.
    result: {hand, unknown, idx, label, label_en, conf, bbox}."""
    overlay = frame.copy()
    result = {"hand": False, "unknown": False, "idx": None, "label": None,
              "label_en": None, "conf": 0.0, "bbox": None}
    landmarks = ar.detect_hand(frame, ts_ms)  # HandLandmarker مشترك (فاصل متوافق بين اللغتين)
    if landmarks:
        result["hand"] = True
        patch, bbox = ar.crop_hand_patch(frame, landmarks, pad=ar.HAND_PAD)
        if patch is not None:
            idx, conf, margin = classify_en_softmax(patch)
            if not margin_accept(margin):
                idx, conf = None, 0.0
                result["unknown"] = True
            result.update(idx=idx, label=None if idx is None else CLASS_EN_SYMS[idx],
                          label_en=None if idx is None else CLASS_EN_SYMS[idx],
                          conf=conf, bbox=bbox)
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
    return overlay, result


class SignSequencerEN:
    """نسخة من SignSequencer العربي برموز إنجليزية — التمييز الثلاثي (حرف/غير معروف/لا يد)."""

    def __init__(self, conf_threshold=CONF_THRESHOLD_EN, debounce=DEBOUNCE_FRAMES_EN,
                 silence_seconds=SILENCE_SECONDS_EN, max_word=ar.MAX_WORD_LEN):
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
        events = {"committed": None, "word": "".join(CLASS_EN_SYMS[i] for i in self.word),
                  "finalized": None}
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
                self.word.append(idx)
                events["committed"] = (idx, CLASS_EN_SYMS[idx], self._last_conf)
                self._committed_idx = idx
                self._votes[idx] = 0
            events["word"] = "".join(CLASS_EN_SYMS[i] for i in self.word)
        elif hand_seen:
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
        self.finalized_word = "".join(CLASS_EN_SYMS[i] for i in self.word)
        self.word = []


class LivePipelineEN:
    """خط أنابيب إنجليزي كامل: detect+crop → CNN+margin → تسلسل → Groq(en) → gTTS(en)."""

    def __init__(self, conf_threshold=CONF_THRESHOLD_EN, debounce=DEBOUNCE_FRAMES_EN,
                 silence_seconds=SILENCE_SECONDS_EN, auto_correct=True):
        self.seq = SignSequencerEN(conf_threshold, debounce, silence_seconds)
        self.t0 = time.monotonic()
        self.auto_correct = auto_correct

    def update(self, frame):
        ts = time.monotonic() - self.t0
        overlay, result = process_frame_en(frame, int(ts * 1000))
        events = self.seq.feed(result["hand"], result["idx"], result["conf"], ts)
        out = {**result, "events": events, "overlay": overlay,
               "word": events["word"], "corrected": None, "audio": None}
        if events["finalized"]:
            out["corrected"] = (ar.correct_word(events["finalized"], language="en")
                                if self.auto_correct
                                else {"text": events["finalized"], "source": "raw", "api": False})
            out["audio"] = ar.synthesize_speech(out["corrected"]["text"], lang="en")
        return out


# ---------------------------------------------------------------- الفحص الذاتي (EN)

def en_checks(check) -> None:
    """فحوص en.* تُدمج في --selftest العربي عبر engine.run_selftest (دالة check تُمرَّر)."""
    from pathlib import Path as _P
    check("en.artifacts.npz", ASL_MNIST_NPZ.exists())
    check("en.artifacts.model", MODEL_EN_PATH.exists())
    check("en.artifacts.class_map", CLASS_MAP_EN_PATH.exists())
    check("en.artifacts.dict_en", len(list(DICT_EN_DIR.glob("*.png"))) == EN_CLASSES)

    images, labels = load_en_dataset()
    check("en.data.shape", images.shape == (EN_IMAGES, IMG_SIZE, IMG_SIZE, 1)
          and len(np.unique(labels)) == EN_CLASSES)

    idx, conf, margin = classify_en_softmax(images[0][:, :, 0])
    check("en.classify.smoke", 0 <= idx < EN_CLASSES and 0.0 <= conf <= 1.0 and 0.0 <= margin <= 1.0)
    check("en.margin.gate", margin_accept(0.08) is True and margin_accept(0.01) is False)

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


# ---------------------------------------------------------------- CLI

def main() -> None:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ar.setup_logging()
    if "--fetch-en" in sys.argv:
        fetch_en_data()
    elif "--train-en" in sys.argv:
        train_en()
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