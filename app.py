"""واجهة Streamlit لمترجم لغة الإشارة العربية — تبويبان فقط (ترجمة لحظية + قاموس).
كل المنطق في engine.py؛ هنا الاستهلاك/العرض فقط (RTL، خط عربي، بطاقات)."""

import sys
import base64
import json
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

st.set_page_config(page_title="مترجم لغة الإشارة العربية", page_icon="\U0001F91F", layout="wide")

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

.conf-wrap { background: #2a3040; border-radius: 999px; height: 10px; width: 100%; margin: 8px auto 0; max-width: 340px; }
.conf-fill { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #1db954, #f5c518); transition: width .12s ease-out, opacity .12s ease-out; }
.badge, .big-letter, .big-letter-unknown { transition: opacity .12s ease-out; }
.meta { color: #9aa4b5; text-align: center; }
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


# --- مبدّل اللغة: يحوّل النموذج النشط/خريطة الفئات/الاتجاه/اللغة الصوتية/صور القاموس — بلا لمس العربي
lang = st.radio("اللغة / Language", ["عربي", "English"], horizontal=True, key="lang_toggle")
IS_EN = lang == "English"
if IS_EN and engine_en is None:
    st.error("الوحدة الإنجليزية غير متاحة — شغّل: python engine_en.py --fetch-en ثم --train-en")

tab_live, tab_dict = st.tabs(["\U0001F3A5 ترجمة لحظية", "\U0001F4D6 القاموس"])

if IS_EN:
    _PCMODEL = engine_en.MODEL_EN_PATH
    _PCLASS = engine_en.LivePipelineEN
    _DIRN = "ltr"
else:
    _PCMODEL = engine.MODEL_PATH
    _PCLASS = engine.LivePipeline
    _DIRN = "rtl"

with tab_live:
    st.title("مترجم لغة الإشارة" if IS_EN else "مترجم لغة الإشارة العربية")
    if not _PCMODEL.exists():
        st.error("النموذج غير مدرب بعد — شغّل: " + ("python engine_en.py --train-en" if IS_EN
                 else "python engine.py --train"))
        st.stop()
    if not engine.HAND_MODEL.exists():
        st.error("أصل MediaPipe مفقود: features/hand_landmarker.task")
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
        "مفعّل — يُصحَّح النص تلقائياً" if has_key
        else "غير مفعّل — يُستخدم النص الخام حتى إضافة GROQ_API_KEY في .env"))

    col_ctrl = st.columns([1, 3])[0]
    with col_ctrl:
        if st.button("بدء التقاط الكاميرا", type="primary", width="stretch"):
            if st.session_state.cap is None or not st.session_state.cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("تعذر فتح الكاميرا (source=0)")
                else:
                    st.session_state.cap = cap
            st.session_state.live = True
        if st.button("إيقاف", width="stretch"):
            st.session_state.live = False
        if has_key is False:
            st.info("أضف مفتاح Groq في .env ليُصحَّح النص تلقائياً", icon="\U0001F511")

    frame_ph = st.empty()
    hand_ph = st.empty()
    letter_ph = st.empty()
    word_ph = st.empty()
    status_ph = st.empty()
    audio_ph = st.empty()
    history_ph = st.empty()


    @st.fragment(run_every=0.1)
    def live_loop():
        if not st.session_state.get("live", False):
            return
        cap = st.session_state.cap
        if cap is None or not cap.isOpened():
            st.session_state.live = False
            status_ph.error("الكاميرا غير متاحة")
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
                    f"<div class='big-letter'>{out['label']}</div>"
                    f"<div class='meta'>confidence {out['conf']:.2f}</div>"
                    f"<div class='conf-wrap'><div class='conf-fill' style='width:{pct}%'></div></div>",
                    unsafe_allow_html=True)
            else:
                letter_ph.markdown(
                    f"<div class='big-letter'>{out['label']}</div>"
                    f"<div class='meta'>{out['label_en']} — الثقة {out['conf']:.2f}</div>"
                    f"<div class='conf-wrap'><div class='conf-fill' style='width:{pct}%'></div></div>",
                    unsafe_allow_html=True)
        elif out.get("unknown"):
            letter_ph.markdown(
                ("<div class='big-letter-unknown'>Unclear</div>"
                 "<div class='meta'>Not confident — change your handshape</div>"
                 if IS_EN else "<div class='big-letter-unknown'>غير معروف</div>"
                 "<div class='meta'>مش واضح — غيّر وضع يدك ليُحسم الحرف</div>"),
                unsafe_allow_html=True)
        elif out["hand"]:
            letter_ph.markdown(
                ("<div class='meta'>Hand visible, not committed yet</div>"
                 if IS_EN else "<div class='meta'>يد مرئية بلا التزام بعد</div>"),
                unsafe_allow_html=True)
        else:
            letter_ph.markdown(
                ("<div class='meta'>Waiting for a hand...</div>"
                 if IS_EN else "<div class='meta'>بانتظار اليد...</div>"),
                unsafe_allow_html=True)

        if out["word"]:
            word_ph.markdown(f"<div class='word-line' style='direction:{_DIRN}'>{out['word']}</div>",
                             unsafe_allow_html=True)
        else:
            word_ph.markdown("")

        if out["events"]["finalized"] and out["events"]["finalized"] != st.session_state[lkey]:
            st.session_state[lkey] = out["events"]["finalized"]
            st.session_state[hkey].append(
                (out["events"]["finalized"], out["corrected"]["text"], out["corrected"]["source"]))
            if out["audio"]:
                audio_ph.audio(out["audio"], format="audio/mp3", autoplay=True)
                st.session_state[f"aud_{lang}"] = out["audio"]

        history_items = []
        for raw, text, src in st.session_state[hkey]:
            if src == "groq" and raw != text:
                history_items.append(f"<span class='raw'>{raw} ← </span>{text}"
                                     "<span class='badge badge-qamari'>Groq</span>")
            else:
                tag = "raw" if (src == "raw" and IS_EN) else ("خام" if src == "raw" else "Groq")
                history_items.append(f"{text}<span class='badge badge-index'>{tag}</span>")
        history_ph.markdown(
            f"<div class='word-line' style='direction:{_DIRN}'>{'&nbsp;&nbsp;'.join(history_items)}</div>"
            if history_items else ("<div class='meta'>History empty — e.g. «H E L L O»</div>"
                                   if IS_EN else "<div class='meta'>سجلّ الكلمات فارغ — مثل: «س ← ل ← ا ← م»</div>"),
            unsafe_allow_html=True)


    live_loop()

    if f"aud_{lang}" in st.session_state:
        col_aud = st.columns([1, 4])[0]
        with col_aud:
            st.download_button(
                "⬇ تحميل آخر تسجيل صوتي (mp3)",
                data=st.session_state[f"aud_{lang}"],
                file_name=f"sign_word_{lang}.mp3",
                mime="audio/mpeg",
                width="stretch")

    st.divider()
    if IS_EN:
        st.markdown("**How it works:** show a sign ≥ 5 consecutive frames with conf ≥ 0.90 and a "
                    "top1−top2 margin ≥ 0.05 to commit the letter (unclear → shown). Then hold "
                    "no hand for 2.5 s to lock the word and speak it (auto Groq correction if key set).")
    else:
        st.markdown("**كيف تعمل:** اعرض إشارة أمام الكاميرا ≥ 5 إطارات متتالية بثقة ≥ 0.90 مع "
                    "تحقّق هندسي (ألف/سين/فاء/ثاء معتمدة عند التحقق) ليُلتزم الحرف. إن عارضت الهندسة "
                    "تصنيف CNN يُعرض «غير معروف» بدل حرف مخطئ. ثم انتظر بلا يد 2.5 ثانية لتُقفل "
                    "الكلمة وتُنطق (مع تصحيح Groq التلقائي إن توفّر).")

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
            card += f"<img src='{uri}' alt='{info['name']}'/>"
        card += f"<div style='margin-top:8px'>{_dict_badge(i, cat)}</div>"
        card += (f"<div class='dict-sym'>{info['sym']}</div>"
                 f"<div class='dict-name'>{info['name']}</div></div>")
        with cols[i % 4]:
            st.markdown(card, unsafe_allow_html=True)

st.markdown("<div class='meta' style='margin-top:24px'>العربي: val 95.18% · 32 إشارة ArASL · "
            "English: val 100% · 24 ASL signs — واجهة محلية Streamlit (M3–M11 + P1 + EN)</div>",
            unsafe_allow_html=True)