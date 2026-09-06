"""بند 6: يولّد صورة تمثيلية (سكيلتون يدين) لكل كلمة من بيانات ISLR الإنجليزية.
يقرأ parquet اليدين (Kaggle asl_signs) من manifest، يختار إطاراً ممثلاً (الأغنى
باللاندماركات، الأقرب لمنتصف التسلسل عند التعادل)، يرسم الهيكل 21 نقطة × يدين
ويحفظ PNG في data/asl_signs/dict_words/{sign}.png بلا أي نص مضاف."""
import pathlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = pathlib.Path(__file__).resolve().parent
MANIFEST = BASE / "data" / "asl_signs" / "train_selected.csv"
DATA_ROOT = BASE / "data" / "asl_signs" / "train_landmark_files"
OUT_DIR = BASE / "data" / "asl_signs" / "dict_words"

WORDS = ["bye", "drink", "happy", "hello", "no", "please", "sleep", "thankyou", "water", "yes"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

IMG = 240
MARGIN = 18
FILL = (26, 26, 24, 255)
STROKE = (180, 70, 33, 255)
BG = (250, 249, 246, 255)
PLACEHOLDER_FILL = (240, 238, 232, 255)
PLACEHOLDER_DIM = (110, 108, 100, 255)


def find_data(spath: pathlib.Path):
    cand = DATA_ROOT / spath
    if not cand.exists():
        cand = DATA_ROOT / "train_landmark_files" / spath.name
        if not cand.exists():
            hits = list(DATA_ROOT.rglob(spath.name))
            cand = hits[0] if hits else None
    return cand if cand is not None and cand.exists() and cand.stat().st_size > 0 else None


def pick_sequence(sign: str):
    meta = pd.read_csv(MANIFEST)
    rows = meta[meta["sign"] == sign]
    for row in rows.itertuples(index=False):
        spath = find_data(pathlib.Path(row.path))
        if spath is not None:
            return spath
    return None


def pick_frame(spath: pathlib.Path):
    df = pd.read_parquet(spath)
    df = df[df["type"].isin(["left_hand", "right_hand"])]
    df = df[df[["x", "y", "z"]].notna().all(axis=1)]
    if df.empty:
        return None
    per = {int(f): [] for f in df["frame"].unique()}
    for row in df.itertuples(index=False):
        per[int(row.frame)].append(
            (row.type, int(row.landmark_index), float(row.x), float(row.y)))
    best = None
    for frame, pts in per.items():
        n = len(pts)
        if best is None or n > best[0] or (n == best[0] and abs(frame - len(per) / 2) < abs(best[1] - len(per) / 2)):
            best = (n, frame, pts)
    return best


def draw(opts):
    img = Image.new("RGBA", (IMG, IMG), BG)
    if opts is None:
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([8, 8, IMG - 8, IMG - 8], radius=16, outline=PLACEHOLDER_DIM, width=2)
        d.text((IMG / 2, IMG / 2), "no data", fill=PLACEHOLDER_DIM, anchor="mm")
        return np.asarray(img.convert("RGB"))
    scale = (IMG - 2 * MARGIN)
    c = lambda v: MARGIN + v * scale
    hands = {"left_hand": {}, "right_hand": {}}
    for typ, li, x, y in opts:
        hands[typ][li] = (c(x), c(y))
    d = ImageDraw.Draw(img)
    for pts in hands.values():
        for a, b in HAND_CONNECTIONS:
            if a in pts and b in pts:
                d.line([pts[a], pts[b]], fill=STROKE, width=4, joint="curve")
        for li, (x, y) in pts.items():
            r = 5 if li in (0, 5, 9, 13, 17) else 4
            d.ellipse([x - r, y - r, x + r, y + r], fill=FILL, outline=STROKE, width=2)
    return np.asarray(img.convert("RGB"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done, missing = [], []
    for sign in WORDS:
        spath = pick_sequence(sign)
        opts = None
        if spath is not None:
            frame = pick_frame(spath)
            opts = frame[2] if frame is not None else None
        arr = draw(opts)
        Image.fromarray(arr).save(OUT_DIR / f"{sign}.png")
        if opts is None:
            missing.append(sign)
        done.append(sign)
        print(f"[dict_words] {sign:10s} -> {OUT_DIR / (sign + '.png')}", flush=True)
    print(f"[dict_words] done={len(done)} no-data={len(missing)}", flush=True)
    if missing:
        print("[dict_words] WARNING (no landmarks):", ", ".join(missing), flush=True)


if __name__ == "__main__":
    main()