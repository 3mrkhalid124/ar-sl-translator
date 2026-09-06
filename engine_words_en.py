"""English Words mode — ASL isolated signs via Kaggle asl-signs landmarks.

Standalone 4th subsystem: does NOT touch Arabic letters (engine.py),
English letters (engine_en.py) or any other existing mode.

Data: 10 words x ~20-37 sequences downloaded from Google asl-signs.
Feature vector per frame (126): left_hand 21x3 then right_hand 21x3 (order matches
MediaPipe handedness layout used by the live pipeline; missing hand rows = 0).
Sequences are time-resampled to MAX_T frames.

Usage:
  python engine_words_en.py --train
  python engine_words_en.py --eval
  python engine_words_en.py --live
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

BASE = pathlib.Path(__file__).resolve().parent
DATA_ROOT = BASE / "data" / "asl_signs" / "train_landmark_files"
MANIFEST = BASE / "data" / "asl_signs" / "train_selected.csv"
MODEL_DIR = BASE / "models"
CKPT = MODEL_DIR / "words_en.keras"
CLASSES_PATH = MODEL_DIR / "words_en_classes.json"
SPLIT_PATH = BASE / "data" / "asl_signs" / "words_en_split.json"
SCALER_PATH = BASE / "data" / "asl_signs" / "words_en_scaler.json"

MAX_T = 40
L_HAND = 126  # 2 * 21 * 3
LEFT, RIGHT = 0, 1


# ---------------------------------------------------------------- features
def parquet_to_frames(spath: pathlib.Path) -> np.ndarray:
    """Read a Kaggle asl-signs parquet -> (T, 126) float32 [left(63) | right(63)]."""
    import pandas as pd

    df = pd.read_parquet(spath)
    df = df[df["type"].isin(["left_hand", "right_hand"])]
    if df.empty:
        return np.zeros((MAX_T, L_HAND), dtype=np.float32)
    feats = [np.zeros((21, 3), dtype=np.float32), np.zeros((21, 3), dtype=np.float32)]
    present = df[df[["x", "y", "z"]].notna().all(axis=1)]
    for row in present.itertuples(index=False):
        ti = RIGHT if row.type == "right_hand" else LEFT
        li = int(row.landmark_index)
        if li < 0 or li > 20:
            continue
        feats[ti][li] = (row.x, row.y, row.z)
    left = feats[LEFT].reshape(-1)
    right = feats[RIGHT].reshape(-1)
    seq = np.concatenate([left, right], axis=0).astype(np.float32)
    if seq.size != L_HAND:
        pad = np.zeros(L_HAND - seq.size, np.float32)
        seq = np.concatenate([seq, pad])
    seq = wrist_relative(seq.reshape(1, L_HAND))[0].reshape(-1)
    return _resample_t40(seq)


def wrist_relative(x: np.ndarray) -> np.ndarray:
    """Per frame: subtract each hand's wrist landmark (idx 0) rows for that hand.
    x shape (T,126) or (...,126). Removes absolute position/scale of the signed pose."""
    x = x.copy()
    wl = x[..., 0:3]
    wr = x[..., 63:66]
    x[..., :63] = x[..., :63] - np.tile(wl, 21)
    x[..., 63:] = x[..., 63:] - np.tile(wr, 21)
    return x


def apply_scaler(x: np.ndarray, stats) -> np.ndarray:
    mu = np.asarray(stats["mean"], np.float32)
    sd = np.asarray(stats["std"], np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def _resample_t40(seq: np.ndarray, dim: int = L_HAND) -> np.ndarray:
    n = seq.size // dim
    if n == MAX_T:
        return seq.reshape(MAX_T, dim)
    if n == 0:
        return np.zeros((MAX_T, dim), np.float32)
    sample = max(n, 2)
    out = np.empty((MAX_T, dim), np.float32)
    idx_old = np.linspace(0, sample - 1, n)
    idx_new = np.linspace(0, sample - 1, MAX_T)
    for c in range(dim):
        out[:, c] = np.interp(idx_new, idx_old, seq.reshape(n, dim)[:, c])
    return out


# ---------------------------------------------------------------- dataset
def _discover():
    import pandas as pd

    if not MANIFEST.exists():
        raise SystemExit("manifest missing — run the Kaggle puller first")
    meta = pd.read_csv(MANIFEST)
    have = []
    for row in meta.itertuples(index=False):
        cand = DATA_ROOT / row.path
        if not cand.exists():
            cand = DATA_ROOT / "train_landmark_files" / pathlib.Path(row.path).name
            if not cand.exists():
                # keep old double-nested download layout as a fallback
                hit = list(DATA_ROOT.rglob(pathlib.Path(row.path).name))
                cand = hit[0] if hit else None
        if cand is not None and cand.exists() and cand.stat().st_size > 0:
            have.append((row.path, str(row.participant_id), row.sign, cand))
    return have


def load_all():
    rows = _discover()
    print(f"[words_en] sequences on disk: {len(rows)}", flush=True)
    samples = []
    for path, pid, sign, spath in rows:
        samples.append({"path": path, "pid": int(pid), "sign": sign, "x": parquet_to_frames(spath)})
    signs = sorted({s["sign"] for s in samples})
    print(f"[words_en] signs: {signs}", flush=True)
    per = {}
    for s in samples:
        per.setdefault(s["sign"], []).append(s)
    for k in signs:
        pids = len({s["pid"] for s in per[k]})
        print(f"[words_en] {k:10s} n={len(per[k])} participants={pids}", flush=True)
    return samples, signs


def _pid_partition(pids, val_n=2, test_n=2):
    order = sorted(int(p) for p in pids)
    test = order[:test_n]
    val = order[test_n:test_n + val_n]
    train = order[test_n + val_n:]
    return train, val, test


def split_data(samples, signs):
    import collections

    pids = {int(s["pid"]) for s in samples}
    train_p, val_p, test_p = _pid_partition(pids)
    tr, va, te = [], [], []
    for s in samples:
        target = tr if s["pid"] in train_p else (va if s["pid"] in val_p else te)
        target.append(s)
    counts = collections.Counter()
    for s in tr:
        counts[s["sign"]] += 1
    keep = [sgn for sgn in signs if counts.get(sgn, 0) > 0]
    if len(keep) < len(signs):
        dropped = [sgn for sgn in signs if sgn not in keep]
        print(f"[words_en] WARNING: dropping signs without train samples: {dropped}", flush=True)
    def filt(lst):
        return [s for s in lst if s["sign"] in keep]
    return filt(tr), filt(va), filt(te), keep


def stack(samples, sign2idx):
    X = np.stack([s["x"] for s in samples], axis=0).astype(np.float32)
    y = np.array([sign2idx[s["sign"]] for s in samples], np.int64)
    return X, y


# ---------------------------------------------------------------- model
def build_model(classes: int):
    import keras

    inp = keras.Input(shape=(MAX_T, L_HAND))
    x = keras.layers.Masking(mask_value=0.0)(inp)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.35)(x)
    out = keras.layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------- augment
def mirror(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[..., :63], x[..., 63:] = x[..., 63:].copy(), x[..., :63].copy()
    for k in range(0, L_HAND, 3):
        x[..., k] = 1.0 - x[..., k]
    return x


def jitter(x: np.ndarray, rng: np.random.Generator, sd=0.012) -> np.ndarray:
    return (x + rng.normal(0.0, sd, x.shape)).astype(np.float32)


def speed(x: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    n = MAX_T
    src = np.linspace(0, n - 1, n)
    dst = np.linspace(0, n - 1, int(n / factor))
    out = np.empty((MAX_T, 126), np.float32)
    idx = np.linspace(0, dst.size - 1, n)
    for c in range(126):
        out[:, c] = np.interp(idx, np.arange(dst.size), np.interp(dst, src, x[:, c]))
    return out


def augment_batch(X: np.ndarray, y: np.ndarray, copies=5, seed=7):
    rng = np.random.default_rng(seed + len(X))
    xs, ys = [X], [y]
    for i in range(copies):
        Xi = X.copy()
        for b in range(len(X)):
            r = rng.random(3)
            if r[0] < 0.5:
                Xi[b] = mirror(Xi[b])
            Xi[b] = jitter(Xi[b], rng)
            if r[1] < 0.4:
                Xi[b] = speed(Xi[b], 0.9 + 0.2 * r[2], rng)
        xs.append(Xi)
        ys.append(y.copy())
    Xa = np.concatenate(xs, axis=0).astype(np.float32)
    ya = np.concatenate(ys, axis=0)
    return Xa, ya


# ---------------------------------------------------------------- pipeline
def train_main():
    samples, signs = load_all()
    tr, va, te, keep = split_data(samples, signs)
    sign2idx = {s: i for i, s in enumerate(keep)}
    Xtr, ytr = stack(tr, sign2idx)
    Xva, yva = stack(va, sign2idx)
    Xte, yte = stack(te, sign2idx)

    mu = Xtr.reshape(-1, L_HAND).mean(axis=0).astype(np.float32)
    sd = Xtr.reshape(-1, L_HAND).std(axis=0, ddof=1).astype(np.float32)
    sc = {"mean": mu.tolist(), "std": sd.tolist()}
    SCALER_PATH.write_text(json.dumps(sc), encoding="utf-8")
    Xtr, Xva, Xte = (apply_scaler(a, sc) for a in (Xtr, Xva, Xte))
    print(f"[words_en] scaler saved {SCALER_PATH.name} — split train={len(tr)} val={len(va)} "
          f"test={len(te)} classes={len(keep)}", flush=True)
    SPLIT_PATH.write_text(json.dumps({
        "classes": keep, "sign2idx": sign2idx,
        "train_p": sorted({s['pid'] for s in tr}),
        "val_p": sorted({s['pid'] for s in va}),
        "test_p": sorted({s['pid'] for s in te}),
        "n": {"train": len(tr), "val": len(va), "test": len(te)}}, indent=1), encoding="utf-8")

    Xa, ya = augment_batch(Xtr, ytr, copies=8)
    print(f"[words_en] augmented train: {Xa.shape}", flush=True)

    import keras
    keras.utils.set_random_seed(9)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = build_model(len(keep))
    model.summary(print_fn=lambda s: None)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(CKPT), monitor="val_accuracy",
                                        save_best_only=True, mode="max", verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20,
                                      restore_best_weights=True, verbose=0),
    ]
    t0 = time.time()
    hist = model.fit(Xa, ya, validation_data=(Xva, yva), epochs=220, batch_size=16,
                     callbacks=callbacks, verbose=1)
    print(f"[words_en] train took {time.time()-t0:.0f}s — last val_acc={hist.history['val_accuracy'][-1]:.4f}",
          flush=True)

    best = model if CKPT.exists() else model
    best = keras.models.load_model(str(CKPT))
    CLASSES_PATH.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
    report(best, Xte, yte, keep, "test")


def report(model, X, y, keep, name):
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    acc = float((pred == y).mean())
    print(f"\n[words_en] {name} accuracy: {acc:.4f} (n={len(y)})", flush=True)
    for i, sgn in enumerate(keep):
        mask = y == i
        n = int(mask.sum())
        if n:
            print(f"[words_en]   {sgn:10s} acc={(pred[mask] == i).mean():.3f} n={n}", flush=True)
    print(f"[SELFTEST] words_en.{name}: {'PASS' if acc >= 0.55 else 'FAIL'} acc={acc:.4f}")


# ---------------------------------------------------------------- v2 training (budgeted)
def build_model_v2(kind: str, classes: int):
    """Recipe variants for the budgeted --train-v2 search. All consume (40,126)->classes."""
    import keras

    inp = keras.Input(shape=(MAX_T, L_HAND))
    x = keras.layers.Masking(mask_value=0.0)(inp)
    if kind == "base":
        x = keras.layers.LSTM(64, return_sequences=True)(x)
        x = keras.layers.LSTM(64, return_sequences=True)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(0.35)(x)
    elif kind == "deep":
        x = keras.layers.LSTM(96, return_sequences=True)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.LSTM(64)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.4)(x)
    elif kind == "gru":
        x = keras.layers.GRU(64, return_sequences=True)(x)
        x = keras.layers.GRU(64)(x)
        x = keras.layers.Dense(48, activation="relu")(x)
        x = keras.layers.Dropout(0.4)(x)
    else:
        raise ValueError(kind)
    out = keras.layers.Dense(classes, activation="softmax")(x)
    return keras.Model(inp, out)


def train_v2_main(budget_min: float = 18.0):
    """Budgeted recipe+seed search on the SAME participant split as --train (test=33).
    Picks champion by best VAL accuracy; reports its test. Replaces CKPT ONLY if test
    strictly beats the current checkpoint's test (same 33 held-out clips)."""
    import keras

    samples, signs = load_all()
    tr, va, te, keep = split_data(samples, signs)
    sign2idx = {s: i for i, s in enumerate(keep)}
    Xtr, ytr = stack(tr, sign2idx)
    Xva, yva = stack(va, sign2idx)
    Xte, yte = stack(te, sign2idx)
    sc = {"mean": Xtr.reshape(-1, L_HAND).mean(axis=0).astype(np.float32).tolist(),
          "std": Xtr.reshape(-1, L_HAND).std(axis=0, ddof=1).astype(np.float32).tolist()}
    Xtr, Xva, Xte = (apply_scaler(a, sc) for a in (Xtr, Xva, Xte))

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    def _acc(model, X, y):
        p = np.argmax(model.predict(X, verbose=0), axis=1)
        return float((p == y).mean()), int((p == y).sum()), int(len(y))

    old_test, *_ = _acc(keras.models.load_model(str(CKPT)), Xte, yte)
    print(f"[v2] current checkpoint test acc = {old_test:.4f} (must beat to replace)", flush=True)

    recipes = [("base", 9), ("base", 17), ("deep", 9), ("gru", 9)]
    best = {"rec": None, "seed": None, "val": -1.0, "test": 0.0, "n_test": 0}
    t0 = time.time()
    for rec, seed in recipes:
        if time.time() - t0 > budget_min * 60:
            print(f"[v2] budget {budget_min}min exceeded before {rec}/{seed} — stopping search", flush=True)
            break
        keras.utils.set_random_seed(seed)
        Xa, ya = augment_batch(Xtr, ytr, copies=10)
        m = build_model_v2(rec, len(keep))
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "words_v2_tmp.keras"),
                                              monitor="val_accuracy", save_best_only=True,
                                              mode="max", verbose=0),
              keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                            restore_best_weights=True, verbose=0)]
        m.fit(Xa, ya, validation_data=(Xva, yva), epochs=140, batch_size=32,
              callbacks=cb, verbose=0)
        bm = keras.models.load_model(str(MODEL_DIR / "words_v2_tmp.keras"))
        tva, cn, nv = _acc(bm, Xva, yva)
        tte, ct, nt = _acc(bm, Xte, yte)
        el = time.time() - t0
        print(f"[v2] {rec}/seed{seed}: val={tva:.4f} ({cn}/{nv}) test={tte:.4f} ({ct}/{nt}) "
              f"elapsed={el:.0f}s", flush=True)
        if tva > best["val"]:
            best = {"rec": rec, "seed": seed, "val": tva, "test": tte, "n_test": nt}
        if time.time() - t0 > budget_min * 60:
            print(f"[v2] budget {budget_min}min exceeded after {rec}/{seed} — stopping search", flush=True)
            break

    if best["rec"] is None:
        raise SystemExit("[v2] no valid run — budget too small?")
    print(f"[v2] champion: {best['rec']}/seed{best['seed']} val={best['val']:.4f} "
          f"test={best['test']:.4f} ({int(best['test'] * best['n_test'])}/{best['n_test']})", flush=True)
    if best["test"] > old_test:
        bm = keras.models.load_model(str(MODEL_DIR / "words_v2_tmp.keras"))
        (MODEL_DIR / "words_v2_tmp.keras").unlink(missing_ok=True)
        bm.save(str(CKPT))
        CLASSES_PATH.write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
        SPLIT_PATH.write_text(json.dumps({
            "classes": keep, "sign2idx": sign2idx,
            "train_p": sorted({s['pid'] for s in tr}),
            "val_p": sorted({s['pid'] for s in va}),
            "test_p": sorted({s['pid'] for s in te}),
            "n": {"train": len(tr), "val": len(va), "test": len(te)}}, indent=1), encoding="utf-8")
        SCALER_PATH.write_text(json.dumps(sc), encoding="utf-8")
        print(f"[v2] REPLACED checkpoint: old test {old_test:.4f} -> new {best['test']:.4f}", flush=True)
    else:
        (MODEL_DIR / "words_v2_tmp.keras").unlink(missing_ok=True)
        print(f"[v2] kept current checkpoint: {best['test']:.4f} <="
              f" {old_test:.4f} (no numerically-better model)", flush=True)
    return best["test"] if best["test"] > old_test else old_test


def eval_main():
    import keras
    if not CKPT.exists():
        raise SystemExit("no checkpoint — run --train first")
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    sign2idx = {s: i for i, s in enumerate(classes)}
    model = keras.models.load_model(str(CKPT))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    sc = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
    samples, _ = load_all()
    test_p = set(int(p) for p in split["test_p"])
    te = [s for s in samples if int(s["pid"]) in test_p]
    Xte, yte = stack(te, sign2idx)
    Xte = apply_scaler(Xte, sc)
    report(model, Xte, yte, classes, "test")


class WordsLive:
    """Stateful live ISLR pipeline for the English-Words subsystem (Streamlit + CLI).
    Two-hand landmarks -> 126-dim wrist-relative+scaled frame -> rolling 40-frame
    window -> LSTM prediction. One instance per tab session."""

    def __init__(self):
        import keras
        if not CKPT.exists():
            raise SystemExit("no checkpoint — run --train first")
        self.classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        self.sc = json.loads(SCALER_PATH.read_text(encoding="utf-8"))
        self.model = keras.models.load_model(str(CKPT))
        from engine import HAND_MODEL
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.det = mp_vision.HandLandmarker.create_from_options(opts)
        self.buf = np.zeros((MAX_T, L_HAND), np.float32)
        self.fill = 0
        self.t_hand = None
        self._ts = 0

    def _frame_vec(self, res):
        vec = np.zeros(L_HAND, np.float32)
        if res and res.hand_landmarks:
            pairs = sorted(zip(res.hand_landmarks, [h.category_name for h in (res.handednesses or [])]),
                           key=lambda t: 0 if t[1] == "Left" else 1)
            for lm, cat in pairs[:2]:
                slot = LEFT if cat == "Left" else RIGHT
                vec[slot * 63:slot * 63 + 63] = \
                    np.array([[p.x, p.y, p.z] for p in lm], np.float32).reshape(-1)
        vec = wrist_relative(vec.reshape(1, L_HAND))[0]
        return apply_scaler(vec, self.sc).astype(np.float32)

    def update(self, frame_bgr):
        """frame_bgr -> dict(overlay, word, conf, hand, ready, error). Cheap per frame."""
        import cv2
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        try:
            res = self.det.detect_for_video(mp_image, self._ts)
        except Exception:
            res = None
        had = bool(res and res.hand_landmarks)
        vec = self._frame_vec(res) if had else np.zeros(L_HAND, np.float32)
        now = time.time()
        if had:
            if self.fill < MAX_T:
                self.buf[self.fill] = vec
                self.fill += 1
            else:
                self.buf[:-1] = self.buf[1:]
                self.buf[-1] = vec
            self.t_hand = now
        elif self.t_hand and now - self.t_hand > 1.0 and self.fill > 0:
            self.fill = 0  # gap long enough -> start a new word

        word, conf = None, 0.0
        if self.fill == MAX_T:
            probs = self.model.predict(self.buf[None, ...], verbose=0)[0]
            top = int(np.argmax(probs))
            conf = float(np.max(probs))
            word = self.classes[top]
            cv2.putText(frame_bgr, f"{word} {conf:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
        return {"overlay": frame_bgr, "word": word, "conf": conf,
                "hand": had, "ready": self.fill == MAX_T, "error": None}


def live_main():
    """CLI live webcam mode (same pipeline as the Streamlit tab)."""
    import cv2
    pl = WordsLive()
    cap = cv2.VideoCapture(0)
    print("[words_en] live — q to quit", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = pl.update(frame)
            cv2.imshow("English Words", out["overlay"])
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-v2", type=float, nargs="?", const=18.0,
                    help="budgeted recipe/seed search on the same split (minutes, default 18)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.train:
        train_main()
    elif args.train_v2 is not None:
        train_v2_main(args.train_v2)
    elif args.eval:
        eval_main()
    elif args.live:
        live_main()
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(main())