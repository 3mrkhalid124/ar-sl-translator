"""واجهة Streamlit لمترجم لغة الإشارة العربية — تبويبان فقط (ترجمة لحظية + قاموس).
كل المنطق في engine.py؛ هنا الاستهلاك/العرض فقط (RTL، خط عربي، بطاقات)."""

import sys
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine
try:
    import engine_en
except Exception:
    engine_en = None  # لو غابت الوحدة الإنجليزية — العربي يعمل كما كان بلا كسر

st.set_page_config(page_title="Sign Language Translator", page_icon="\U0001F91F", layout="wide")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
:root, .stApp { direction: rtl; }
html, body, .stApp, [class*="css"] {
  font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif;
  background: #0e1117; color: #e8eaed;
}
h1, h2, h3 { color: #f5c518 !important; text-align: right; }
.big-letter { font-size: 72px; font-weight: 700; color: #1db954; text-align: center; min-height: 78px; line-height: 78px; }
.big-letter-unknown { font-size: 54px; font-weight: 700; color: #ff9f43; text-align: center; min-height: 78px; line-height: 78px; }
.word-line { font-size: 30px; color: #ffffff; text-align: center; direction: rtl; }
.word-line .raw { color: #9aa4b5; font-size: 22px; }
.meta { color: #9aa4b5; text-align: center; min-height: 20px; }

.dict-card {
  background: linear-gradient(180deg, #1c2230, #161b25);
  border: 2px solid #2a3040; border-radius: 14px;
  padding: 10px; margin: 8px 0; text-align: center;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  cursor: default;
}
.dict-card:hover {
  transform: translateY(-4px);
  border-color: #f5c518;
  box-shadow: 0 8px 20px rgba(0, 0, 0, .5);
}
.dict-card img { border-radius: 10px; width: 100%; border: 2px solid #2a3040; }
.dict-name { font-weight: 700; color: #f5c518; margin-top: 6px; font-size: 18px; }
.dict-sym { color: #9aa4b5; font-size: 15px; }
.cat-shamsi { border-color: #9c7c1e; }
.cat-shamsi:hover { border-color: #ffd34d; box-shadow: 0 8px 20px rgba(245, 197, 24, .18); }
.cat-qamari { border-color: #14683a; }
.cat-qamari:hover { border-color: #1db954; box-shadow: 0 8px 20px rgba(29, 185, 84, .18); }
.cat-markab { border-color: #5b3fa8; }
.cat-markab:hover { border-color: #a78bfa; box-shadow: 0 8px 20px rgba(167, 139, 250, .18); }

.badge {
  display: inline-block; font-size: 11px; font-weight: 700; border-radius: 999px;
  padding: 2px 10px; margin: 2px; color: #0e1117; letter-spacing: .3px;
}
.badge-shamsi { background: linear-gradient(135deg, #ffd34d, #ff9f43); }
.badge-qamari { background: linear-gradient(135deg, #2ec4b6, #1db954); }
.badge-markab { background: linear-gradient(135deg, #a78bfa, #7c3aed); color: #ffffff; }
.badge-index { background: #2a3040; color: #9aa4b5; }
.badge-hand-ok { background: linear-gradient(135deg, #1db954, #2ec4b6); font-size: 14px; }
.badge-hand-unknown { background: linear-gradient(135deg, #ff9f43, #f5c518); font-size: 14px; }
.badge-hand-wait { background: #2a3040; color: #9aa4b5; font-size: 14px; }

.conf-wrap { background: #2a3040; border-radius: 999px; height: 10px; width: 100%; margin: 8px auto 0; max-width: 340px; position: relative; }
.conf-fill { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #1db954, #f5c518); transition: width .12s ease-out, opacity .12s ease-out; }
.conf-wrap::after { content: ""; position: absolute; left: 90%; top: -3px; bottom: -3px; width: 2px; background: rgba(255, 255, 255, .35); }
.badge, .big-letter, .big-letter-unknown { transition: opacity .12s ease-out; }
.meta { color: #9aa4b5; text-align: center; min-height: 20px; }

/* ---- البند 1: أنيميشن تكوين الحرف/الكلمة (بلاطات + كشف Groq + فقاعات) ---- */
@keyframes pop-in { 0% { opacity: 0; transform: scale(.78) translateY(6px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
@keyframes tile-in { 0% { opacity: 0; transform: translateY(10px) scale(.7) rotate(-2deg); } 70% { transform: translateY(-2px) scale(1.06); } 100% { opacity: 1; transform: translateY(0) scale(1) rotate(0); } }
@keyframes bubble-in { 0% { opacity: 0; transform: translateY(14px) scale(.92); } 100% { opacity: 1; transform: translateY(0) scale(1); } }

.big-slot { min-height: 86px; text-align: center; }
.stImage img { max-height: 380px; width: auto !important; margin: 0 auto; display: block; object-fit: contain; }
.big-letter, .big-letter-unknown { animation: pop-in .2s ease 1; }
.big-letter-unknown { color: #ff9f43; }

/* تراكم الكلمة: 40 خانة ثابتة — البلاطة تُملأ عند كل حرف مُلتزم (transition على نفس الخانة) */
.word-stage { display: flex; flex-direction: column; align-items: center; gap: 4px; min-height: 74px; margin-top: 10px; }
.tilerow { display: flex; gap: 4px; min-height: 56px; align-items: center; justify-content: center; }
.tile { width: 40px; height: 54px; flex: none; display: inline-flex; align-items: center; justify-content: center;
        border-radius: 9px; border: 1.5px solid #1db954; background: linear-gradient(160deg, #123524, #0e2016);
        color: #e8f7ee; font-weight: 700; font-size: 27px; box-shadow: 0 4px 12px rgba(29, 185, 84, .22);
        animation: tile-in .22s ease 1; }
.tile-spacer { width: 0; flex: none; }
.tile-count { color: #5f6b7d; font-size: 12px; }

/* كشف فاعلية AI عند إغلاق الكلمة: يخفت الخام ويدخل المصحح بصورة أكبر */
.reveal-stage { display: flex; flex-direction: column; align-items: center; gap: 4px; min-height: 66px; margin-top: 8px; }
.reveal-stage .rev-raw { display: flex; gap: 3px; opacity: 1; transition: opacity .3s ease-out; }
.reveal-stage .rev-raw .tile { width: 26px; height: 34px; font-size: 18px; border-color: #9aa4b5; box-shadow: none;
                               background: linear-gradient(160deg, #1c2230, #141922); }
.reveal-stage .rev-cor { opacity: 0; transform: translateY(8px) scale(.9); transition: opacity .3s ease-out, transform .3s ease-out;
                          font-size: 32px; font-weight: 700; color: #f5c518; }
.reveal-stage.done .rev-raw { opacity: 0; }
.reveal-stage.done .rev-cor { opacity: 1; transform: translateY(0) scale(1); }

/* سجل الجمل كفقاعات شات */
.bub-wrap { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.bub { align-self: flex-start; max-width: 82%; padding: 8px 14px; border-radius: 14px 14px 14px 4px;
       background: linear-gradient(160deg, #1c2230, #151a25); border: 1px solid #2a3040; font-size: 21px;
       color: #e8eaed; box-shadow: 0 3px 10px rgba(0, 0, 0, .35); animation: bubble-in .28s ease 1; }
.bub-ai { align-self: flex-end; border-radius: 14px 14px 4px 14px; background: linear-gradient(160deg, #332d14, #221b08);
          border-color: #f5c518; color: #ffedb0; }
.bub .raw-mini { display: block; font-size: 13px; color: #9aa4b5; padding-bottom: 2px; }
.bub .badge { vertical-align: middle; }
.bub-slot { display: none; }

/* ---- البند 2: مؤشرات أوضح — لون الثقة حسب قيمتها، موجة صوت CSS، fade بين اللغتين ---- */
@keyframes page-fade { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
@keyframes wave { 0%, 100% { transform: scaleY(.35); } 50% { transform: scaleY(1); } }
.stMainBlockContainer { animation: page-fade .3s ease 1; }
.wave-row { display: flex; gap: 4px; align-items: flex-end; justify-content: center; height: 26px; margin: 6px 0 2px; }
.wave-row .bar { width: 5px; height: 24px; border-radius: 3px; background: #2ec4b6; transform-origin: bottom;
                animation: wave 1s ease-in-out infinite; animation-play-state: paused; opacity: .85; }
.wave-row.playing .bar { animation-play-state: running; }

/* ---- البند 3: خلفية متدرجة + نفس الشكر (pattern) حسب اللغة + skeleton القاموس ---- */
html, body, .stApp { background-color: #0d1117; }
body.stApp, .stMain, .stApp {
  background-image:
    radial-gradient(1100px 750px at 88% -8%, rgba(245, 197, 24, .08), transparent 62%),
    radial-gradient(1000px 700px at -8% 112%, rgba(29, 185, 84, .07), transparent 60%);
  background-attachment: fixed;
}
.stMainBlockContainer { background: rgba(13, 17, 23, .62); border-radius: 18px; padding: 14px 20px; }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes card-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.dict-card { animation: card-in .3s ease 1; }
.dict-card img {
  min-height: 92px; object-fit: contain;
  background: linear-gradient(110deg, #1a2030 25%, #232b3d 45%, #1a2030 65%);
  background-size: 200% 100%; animation: shimmer 1.3s infinite;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _load_class_map():
    try:
        return json.loads(engine.CLASS_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {str(i): {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]}
                for i in range(engine.EXPECTED_CLASSES)}


# فئات حروف عربية موضوعية (شمسية/قمرية) + مركّبات — تلوين البطاقات بلا افتراضات في شكل اليد
SHAMSI_LETTERS = {24, 25, 4, 26, 19, 31, 21, 22, 20, 6, 23, 5, 16, 18}
QAMARI_LETTERS = {2, 3, 12, 11, 14, 0, 9, 7, 8, 13, 17, 10, 28, 30}
CAT_LABEL = {"shamsi": "شمسية", "qamari": "قمرية", "markab": "خاصة/مركبة"}


def _cat(idx):
    if idx in SHAMSI_LETTERS:
        return "shamsi"
    if idx in QAMARI_LETTERS:
        return "qamari"
    return "markab"


def _img_uri(idx):
    p = engine.DICT_DIR / f"class_{idx:02d}.png"
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


# ---- بند 1: عرض «تكوين» الحروف/الكلمات — HTML فقط، من بيانات الحالة نفسها (بلا لمس منطق) ----
_MAX_TILES = 40
_HIST_SLOTS = 8


def _conf_bar(pct):
    """شريط الثقة: تدرّج لوني حي — أحمر تحت العتبة، أصفر قريب، أخضر/سماوي فوقها (CSS inline فقط)."""
    TH = int(round(engine.CONF_THRESHOLD * 100))
    if pct >= TH:
        k = min(1.0, (pct - TH) / max(1, 100 - TH))
        col = f"rgb({29 + round(82 * k)},{185 + round(20 * k)},{84 + round(124 * k)})"
    elif pct >= 60:
        k = (pct - 60) / max(1, TH - 60)
        col = f"rgb({round(229 - 20 * k)},{round(82 + 57 * k)},{round(80 + 117 * k)})"
    else:
        col = "rgb(229,82,80)"
    return (f"<div class='conf-wrap'><div class='conf-fill' "
            f"style='width:{pct}%; background:{col}; box-shadow:0 0 12px {col}66'></div></div>")


def _wave_md():
    """موجة صوت خفيفة (CSS keyframes فقط) تُعرض أثناء تشغيل mp3."""
    bars = "".join(f"<span class='bar' style='animation-delay:{i * .12:.2f}s'></span>" for i in range(7))
    return "<div class='wave-row playing'>" + bars + "</div>"


def _tile_md(word, lang):
    """خط الكلمة الحي: 40 خانة ثابتة، البلاطات تُملأ مع كل حرف مُلتزم (تلوين مرجعي فقط)."""
    slots = []
    for i in range(_MAX_TILES):
        if i < len(word):
            slots.append(f"<span class='tile'>{word[i]}</span>")
        else:
            slots.append("<span class='tile-spacer'></span>")
    return f"<div class='word-stage'><div class='tilerow'>{''.join(slots)}</div>" \
           f"<div class='tile-count'>{len(word)}/{_MAX_TILES}</div></div>"


def _reveal_md(rev, lang):
    """أنيميشن كشف فاعلية AI: التسلسل الخام يخفت ← المصحح يدخل أكبر/بلون مميز."""
    raw, text, src = rev
    ai = src == "groq" and raw != text
    badge = ("<span class='badge badge-qamari'>AI ✨</span>" if ai
             else "<span class='badge badge-index'>" + tr("raw", "خام") + "</span>")
    raw_tiles = "".join(f"<span class='tile'>{c}</span>" for c in raw)
    return ("<div class='reveal-stage done' dir='" + _DIRN + "'>"
            f"<div class='rev-raw'>{raw_tiles}</div>"
            f"<div class='rev-cor'>{text} {badge}</div></div>")


def _history_md(items, lang):
    """رسوم آخر الكلمات كفقاعات شات (آخرها أسفل) — عرض فقط، بلا تعديل لبيانات الحالة."""
    if not items:
        return ("<div class='meta'>History empty — e.g. «H E L L O»</div>"
                if IS_EN else "<div class='meta'>سجلّ الكلمات فارغ — مثل: «س ← ل ← ا ← م»</div>")
    items = items[-_HIST_SLOTS:]
    out = []
    for i in range(_HIST_SLOTS):
        idx = len(items) - _HIST_SLOTS + i
        if 0 <= idx < len(items):
            raw, text, src = items[idx]
            ai = src == "groq" and raw != text
            tag = ("<span class='badge badge-qamari'>AI ✨</span>" if ai
                   else "<span class='badge badge-index'>"
                        + tr("raw", "خام") + "</span>")
            mini = f"<span class='raw-mini'>{raw}</span>" if ai else ""
            out.append(f"<div class='bub {'bub-ai' if ai else ''}' dir='{_DIRN}'>{mini}{text}{tag}</div>")
        else:
            out.append("<div class='bub-slot'></div>")
    return "<div class='bub-wrap'>" + "".join(out) + "</div>"


# ---- بند 3: خلفية وهوية بصرية — نقش هندسي إسلامي خفيف (عربي) مقابل شبكة بسيطة (إنجليزي) ----
def _svg_pattern(simple):
    if simple:
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'>"
               "<defs><pattern id='p' width='120' height='120' patternUnits='userSpaceOnUse'>"
               "<g fill='none' stroke='rgba(255,255,255,0.05)' stroke-width='1'>"
               "<path d='M0 60h120M60 0v120'/><circle cx='60' cy='60' r='4' fill='rgba(255,255,255,0.08)'/>"
               "</g></pattern></defs><rect width='100%25' height='100%25' fill='url(%23p)'/></svg>")
    else:
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'>"
               "<defs><pattern id='p' width='120' height='120' patternUnits='userSpaceOnUse'>"
               "<g fill='none' stroke='rgba(245,197,24,0.05)' stroke-width='1'>"
               "<rect x='36' y='36' width='48' height='48'/>"
               "<polygon points='60,10 70,50 110,60 70,70 60,110 50,70 10,60 50,50'/>"
               "<circle cx='60' cy='60' r='10'/>"
               "</g></pattern></defs><rect width='100%25' height='100%25' fill='url(%23p)'/></svg>")
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# --- مبدّل اللغة: يحوّل النموذج النشط/خريطة الفئات/الاتجاه/اللغة الصوتية/صور القاموس — بلا لمس العربي
lang = st.radio("اللغة / Language", ["عربي", "English"], horizontal=True, key="lang_toggle")
IS_EN = lang == "English"
if IS_EN and engine_en is None:
    st.error("English module unavailable — run: python engine_en.py --fetch-en then --train-en")


def tr(en, ar_):
    """ترجمة نص حسب اللغة النشطة — إنجليزية كاملة في الوضع الإنجليزي، عربية كاملة في العربي."""
    return en if IS_EN else ar_


# خلفية لغة-مخصوصة + توجيه خالص (LTR في الإنجليزي، RTL في العربي — بلا مزيج)
_WALL = _svg_pattern(IS_EN)
_DIRCSS = (":root, .stApp { direction: rtl; } h1, h2, h3 { text-align: right; }"
           if not IS_EN else
           ":root, .stApp { direction: ltr; } h1, h2, h3 { text-align: left; }")
st.markdown(
    f"<style>{_DIRCSS} html, body, .stApp {{ background-image: url('{_WALL}'), "
    "radial-gradient(1100px 750px at 88% -8%, rgba(245,197,24,.08), transparent 62%), "
    "radial-gradient(1000px 700px at -8% 112%, rgba(29,185,84,.07), transparent 60%); "
    "background-attachment: fixed; background-size: auto, auto, auto; }}</style>",
    unsafe_allow_html=True)

tab_live, tab_dict = (st.tabs(["\U0001F3A5 Live Translation", "\U0001F4D6 Dictionary"])
                      if IS_EN else
                      st.tabs(["\U0001F3A5 ترجمة لحظية", "\U0001F4D6 القاموس"]))

if IS_EN:
    _PCMODEL = engine_en.MODEL_EN_PATH
    _PCLASS = engine_en.LivePipelineEN
    _DIRN = "ltr"
else:
    _PCMODEL = engine.MODEL_PATH
    _PCLASS = engine.LivePipeline
    _DIRN = "rtl"

with tab_live:
    st.title(tr("Sign Language Translator", "مترجم لغة الإشارة العربية"))
    if not _PCMODEL.exists():
        st.error("Model not trained yet — run: " + (tr("python engine_en.py --train-en", "python engine_en.py --train-en")
                 if IS_EN else "python engine.py --train"))
        st.stop()
    if not engine.HAND_MODEL.exists():
        st.error(tr("Hand landmarker model missing: features/hand_landmarker.task",
                    "أصل MediaPipe مفقود: features/hand_landmarker.task"))
        st.stop()

    pkey = f"pipe_{lang}"
    hkey = f"hist_{lang}"
    lkey = f"last_{lang}"
    if pkey not in st.session_state:
        st.session_state[pkey] = _PCLASS()
    if hkey not in st.session_state:
        st.session_state[hkey] = []
    if lkey not in st.session_state:
        st.session_state[lkey] = None
    if "live" not in st.session_state:
        st.session_state.live = False
    if "cap" not in st.session_state:
        st.session_state.cap = None

    has_key = bool(engine._load_groq_key())
    st.caption("Groq: " + (
        tr("Active — auto-correcting text", "مفعّل — يُصحَّح النص تلقائياً") if has_key
        else tr("Inactive — raw text used until GROQ_API_KEY is set in .env",
                "غير مفعّل — يُستخدم النص الخام حتى إضافة GROQ_API_KEY في .env")))

    col_ctrl = st.columns([1, 3])[0]
    with col_ctrl:
        if st.button(tr("Start camera", "بدء التقاط الكاميرا"), type="primary", width="stretch"):
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error(tr("Could not open camera (source=0)", "تعذر فتح الكاميرا (source=0)"))
                else:
                    st.session_state.cap = cap
            st.session_state.live = True
        if st.button(tr("Stop", "إيقاف"), width="stretch"):
            st.session_state.live = False
        if has_key is False:
            st.info(tr("Set GROQ_API_KEY in .env to auto-correct text",
                       "أضف مفتاح Groq في .env ليُصحَّح النص تلقائياً"), icon="\U0001F511")

    cam_col, res_col = st.columns([3, 4])
    with cam_col:
        frame_ph = st.empty()
        hand_ph = st.empty()
    with res_col:
        letter_ph = st.empty()
        reveal_ph = st.empty()
        tiles_ph = st.empty()
        status_ph = st.empty()
        audio_ph = st.empty()
        wave_ph = st.empty()
        history_ph = st.empty()


    @st.fragment(run_every=0.1)
    def live_loop():
        if not st.session_state.get("live", False):
            return
        cap = st.session_state.cap
        if cap is None or not cap.isOpened():
            st.session_state.live = False
            status_ph.error(tr("Camera unavailable", "الكاميرا غير متاحة"))
            return
        ok, frame = cap.read()
        if not ok:
            return
        frame = engine.downscale_live(frame)
        out = st.session_state[pkey].update(frame)
        if out.get("error"):
            st.session_state.live = False
            status_ph.error(out["error"])
            return
        frame_ph.image(out["overlay"], channels="BGR", width="stretch", output_format="JPEG")

        if not out["hand"]:
            hand_md = ("<span class='badge badge-hand-wait'>No hand — show a sign</span>"
                       if IS_EN else "<span class='badge badge-hand-wait'>لا يد — اعرض إشارتك</span>")
        elif out["idx"] is not None:
            hand_md = ("<span class='badge badge-hand-ok'>🖐 Hand detected — letter clear</span>"
                       if IS_EN else "<span class='badge badge-hand-ok'>🖐 يد مكتشفة — الحرف واضح</span>")
        elif out.get("unknown"):
            hand_md = ("<span class='badge badge-hand-unknown'>🖐 Unclear — reposition your hand</span>"
                       if IS_EN else "<span class='badge badge-hand-unknown'>🖐 غير معروف — مش واضح، رجّع وضع اليد</span>")
        else:
            hand_md = ("<span class='badge badge-hand-wait'>Hand visible — waiting...</span>"
                       if IS_EN else "<span class='badge badge-hand-wait'>يد مرئية — بانتظار الوضوح...</span>")
        hand_ph.markdown(f"<div style='text-align:center; margin-bottom:10px'>{hand_md}</div>",
                         unsafe_allow_html=True)

        if out["hand"] and out["idx"] is not None:
            pct = int(out["conf"] * 100)
            if IS_EN:
                letter_ph.markdown(
                    f"<div class='big-slot'><div class='big-letter'>{out['label']}</div>"
                    f"<div class='meta'>confidence {out['conf']:.2f}</div>"
                    f"{_conf_bar(pct)}</div>",
                    unsafe_allow_html=True)
            else:
                letter_ph.markdown(
                    f"<div class='big-slot'><div class='big-letter'>{out['label']}</div>"
                    f"<div class='meta'>{out['label_en']} — الثقة {out['conf']:.2f}</div>"
                    f"{_conf_bar(pct)}</div>",
                    unsafe_allow_html=True)
        elif out.get("unknown"):
            letter_ph.markdown(
                ("<div class='big-slot'><div class='big-letter-unknown'>Unclear</div>"
                 "<div class='meta'>Not confident — change your handshape</div></div>"
                 if IS_EN else "<div class='big-slot'><div class='big-letter-unknown'>غير معروف</div>"
                 "<div class='meta'>مش واضح — غيّر وضع يدك ليُحسم الحرف</div></div>"),
                unsafe_allow_html=True)
        elif out["hand"]:
            letter_ph.markdown(
                ("<div class='big-slot'><div class='meta'>Hand visible, not committed yet</div></div>"
                 if IS_EN else "<div class='big-slot'><div class='meta'>يد مرئية بلا التزام بعد</div></div>"),
                unsafe_allow_html=True)
        else:
            letter_ph.markdown(
                ("<div class='big-slot'><div class='meta'>Waiting for a hand...</div></div>"
                 if IS_EN else "<div class='big-slot'><div class='meta'>بانتظار اليد...</div></div>"),
                unsafe_allow_html=True)

        # تكوين بلاطات الكلمة الحية + كشف AI (reveal) عند الإغلاق
        if out["word"]:
            st.session_state.pop(f"rev_{lang}", None)
            tiles_ph.markdown(_tile_md(out["word"], lang), unsafe_allow_html=True)
            reveal_ph.markdown("")
        else:
            tiles_ph.markdown(_tile_md("", lang), unsafe_allow_html=True)
            rev = st.session_state.get(f"rev_{lang}")
            if rev:
                reveal_ph.markdown(_reveal_md(rev, lang), unsafe_allow_html=True)
            else:
                reveal_ph.markdown("")

        if out["events"]["finalized"] and out["events"]["finalized"] != st.session_state[lkey]:
            st.session_state[lkey] = out["events"]["finalized"]
            st.session_state[hkey].append(
                (out["events"]["finalized"], out["corrected"]["text"], out["corrected"]["source"]))
            st.session_state[f"rev_{lang}"] = (out["events"]["finalized"],
                                               out["corrected"]["text"], out["corrected"]["source"])
            if out["audio"]:
                audio_ph.audio(out["audio"], format="audio/mp3", autoplay=True)
                st.session_state[f"aud_{lang}"] = out["audio"]
                st.session_state[f"wav_{lang}"] = time.perf_counter()

        wts = st.session_state.get(f"wav_{lang}")
        if wts and time.perf_counter() - wts < 4.5:
            wave_ph.markdown(_wave_md(), unsafe_allow_html=True)
        else:
            wave_ph.markdown("")

        history_ph.markdown(_history_md(st.session_state[hkey], lang), unsafe_allow_html=True)


    live_loop()

    if f"aud_{lang}" in st.session_state:
        col_aud = st.columns([1, 4])[0]
        with col_aud:
            st.download_button(
                tr("⬇ Download last audio (mp3)", "⬇ تحميل آخر تسجيل صوتي (mp3)"),
                data=st.session_state[f"aud_{lang}"],
                file_name=f"sign_word_{lang}.mp3",
                mime="audio/mpeg",
                width="stretch")

    st.divider()
    st.markdown(tr(
        "**How it works:** hold a sign for ≥ 5 consecutive frames with confidence ≥ 0.90 and a "
        "top1−top2 margin ≥ 0.05 to commit a letter (unclear → shown). Then hold no hand for 2.5 s "
        "to lock the word and speak it (auto Groq correction when a key is set).",
        "**كيف تعمل:** اعرض إشارة أمام الكاميرا ≥ 5 إطارات متتالية بثقة ≥ 0.90 مع "
        "تحقّق هندسي (ألف/سين/فاء/ثاء معتمدة عند التحقق) ليُلتزم الحرف. إن عارضت الهندسة "
        "تصنيف CNN يُعرض «غير معروف» بدل حرف مخطئ. ثم انتظر بلا يد 2.5 ثانية لتُقفل "
        "الكلمة وتُنطق (مع تصحيح Groq التلقائي إن توفّر)."))

with tab_dict:
    st.title("Sign Language Dictionary" if IS_EN else "قاموس الإشارات")
    if IS_EN:
        n = engine_en.EN_CLASSES
        try:
            _cmap = json.loads(engine_en.CLASS_MAP_EN_PATH.read_text(encoding="utf-8"))
        except Exception:
            _cmap = {str(i): {"sym": s, "name": s} for i, s in enumerate(engine_en.CLASS_EN_SYMS)}

        def _dict_info(i):
            return _cmap.get(str(i), {"sym": engine_en.CLASS_EN_SYMS[i],
                                      "name": engine_en.CLASS_EN_SYMS[i]})

        def _dict_cat(_i):
            return "markab"

        def _dict_uri(i):
            p = engine_en.DICT_EN_DIR / f"class_{i:02d}.png"
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else None

        def _dict_badge(i, cat):
            return f"<span class='badge badge-{cat}'>EN</span><span class='badge badge-index'>#{i:02d}</span>"
    else:
        n = engine.EXPECTED_CLASSES
        cmap_ar = _load_class_map()

        def _dict_info(i):
            return cmap_ar.get(str(i), {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]})

        def _dict_cat(i):
            return _cat(i)

        def _dict_uri(i):
            return _img_uri(i)

        def _dict_badge(i, cat):
            return (f"<span class='badge badge-{cat}'>{CAT_LABEL[cat]}</span>"
                    f"<span class='badge badge-index'>#{i:02d}</span>")

    cols = st.columns(4)
    for i in range(n):
        info = _dict_info(i)
        cat = _dict_cat(i)
        uri = _dict_uri(i)
        card = f"<div class='dict-card cat-{cat}' style='margin-top:14px'>"
        if uri:
            card += f"<img src='{uri}' alt='{info['name']}' loading='lazy'/>"
        card += f"<div style='margin-top:8px'>{_dict_badge(i, cat)}</div>"
        card += (f"<div class='dict-sym'>{info['sym']}</div>"
                 f"<div class='dict-name'>{info['name']}</div></div>")
        with cols[i % 4]:
            st.markdown(card, unsafe_allow_html=True)

st.markdown(tr(
    "<div class='meta' style='margin-top:24px'>Arabic: val 95.18% · 32 ArASL signs · "
    "English: val 100% · 24 ASL signs — local Streamlit interface (M3–M11 + P1 + EN)</div>",
    "<div class='meta' style='margin-top:24px'>العربي: val 95.18% · 32 إشارة ArASL · "
    "English: val 100% · 24 ASL signs — واجهة محلية Streamlit (M3–M11 + P1 + EN)</div>"),
    unsafe_allow_html=True)