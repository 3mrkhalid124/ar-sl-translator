"""واجهة Streamlit لمترجم لغة الإشارة العربية — تبويبان فقط (ترجمة لحظية + قاموس).
كل المنطق في engine.py؛ هنا الاستهلاك/العرض فقط (RTL، خط عربي، بطاقات)."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine

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
.big-letter { font-size: 72px; font-weight: 700; color: #1db954; text-align: center; }
.word-line { font-size: 30px; color: #ffffff; text-align: center; direction: rtl; }
.dict-card {
  background: #1b1f27; border: 1px solid #2a3040; border-radius: 12px;
  padding: 10px; margin: 4px; text-align: center;
}
.dict-card img { border-radius: 8px; width: 100%; }
.dict-name { font-weight: 700; color: #f5c518; margin-top: 6px; }
.dict-sym { color: #9aa4b5; }
.meta { color: #9aa4b5; text-align: center; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _load_class_map():
    import json
    try:
        return json.loads(engine.CLASS_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {str(i): {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]}
                for i in range(engine.EXPECTED_CLASSES)}


tab_live, tab_dict = st.tabs(["\U0001F3A5 ترجمة لحظية", "\U0001F4D6 القاموس"])

with tab_live:
    st.title("مترجم لغة الإشارة العربية")
    if not engine.MODEL_PATH.exists():
        st.error("النموذج غير مدرب بعد — شغّل: python engine.py --train")
        st.stop()
    if not engine.HAND_MODEL.exists():
        st.error("أصل MediaPipe مفقود: features/hand_landmarker.task")
        st.stop()

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = engine.LivePipeline()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_final" not in st.session_state:
        st.session_state.last_final = None
    if "live" not in st.session_state:
        st.session_state.live = False
    if "cap" not in st.session_state:
        st.session_state.cap = None

    has_key = bool(engine._load_groq_key())
    st.caption("Groq: " + (
        "مفعّل — يُصحَّح النص تلقائياً" if has_key
        else "غير مفعّل — يُستخدم النص الخام حتى إضافة GROQ_API_KEY في .env"))

    col_ctrl, col_state = st.columns([1, 3])
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
        out = st.session_state.pipeline.update(frame)
        frame_ph.image(out["overlay"], channels="BGR", width="stretch")

        if out["hand"] and out["idx"] is not None:
            letter_ph.markdown(
                f"<div class='big-letter'>{engine.CLASS_NAMES[out['idx']]}</div>"
                f"<div class='meta'>({engine.ENG_LABELS[out['idx']]} — الثقة "
                f"{out['conf']:.2f})</div>", unsafe_allow_html=True)
        elif out["hand"]:
            letter_ph.markdown("<div class='meta'>يد مرئية بلا التزام بعد</div>",
                               unsafe_allow_html=True)
        else:
            letter_ph.markdown("<div class='meta'>لا يد — رجاءً اعرض إشارتك</div>",
                               unsafe_allow_html=True)

        if out["word"]:
            word_ph.markdown(f"<div class='word-line'>{out['word']}</div>",
                             unsafe_allow_html=True)
        else:
            word_ph.markdown("")

        if out["events"]["finalized"] and out["events"]["finalized"] != st.session_state.last_final:
            st.session_state.last_final = out["events"]["finalized"]
            st.session_state.history.append(
                (out["corrected"]["text"], out["corrected"]["source"]))
            if out["audio"]:
                audio_ph.audio(out["audio"], format="audio/mp3", autoplay=True)

        history_md = "".join(
            f"{t} <span class='meta'>({'Groq' if s == 'groq' else 'raw'}) ·</span> "
            for t, s in st.session_state.history)
        history_ph.markdown(f"<div class='word-line'>{history_md}</div>",
                            unsafe_allow_html=True)


    live_loop()

    st.divider()
    st.markdown("**كيف تعمل:** اعرض إشارة أمام الكاميرا ≥ 3 إطارات متتالية بثقة ≥ 0.85 "
                "ليُلتزم الحرف، ثم انتظر بلا يد 2.5 ثانية لتُقفل الكلمة وتُنطق.")

with tab_dict:
    st.title("قاموس الإشارات")
    cmap = _load_class_map()
    cols = st.columns(4)
    for i in range(engine.EXPECTED_CLASSES):
        info = cmap.get(str(i), {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]})
        img = engine.DICT_DIR / f"class_{i:02d}.png"
        with cols[i % 4]:
            if img.exists():
                st.image(str(img), width="stretch")
            st.markdown(
                f"<div class='dict-card'><div class='dict-sym'>{info['sym']}</div>"
                f"<div class='dict-name'>{info['name']}</div>"
                f"<div class='meta'>#{i:02d}</div></div>",
                unsafe_allow_html=True)

st.markdown("<div class='meta' style='margin-top:24px'>M3–M10 مكتملة — واجهة محلية "
            "Streamlit · نموذج val 95.18% · 32 إشارة ArASL</div>", unsafe_allow_html=True)