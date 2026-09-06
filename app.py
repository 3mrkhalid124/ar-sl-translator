"""واجهة Streamlit لمترجم لغة الإشارة العربية — تبويبان فقط (ترجمة لحظية + قاموس).
كل المنطق في engine.py؛ هنا الاستهلاك/العرض فقط (RTL، خط عربي، بطاقات)."""

import sys
import base64
import json
import time
from pathlib import Path

import cv2
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
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500&display=swap');
:root {
  --bg-page: #FAF9F6; --surface-card: #FFFFFF; --border: #E5E2D9;
  --text-primary: #1A1A18; --text-secondary: #6B6960;
  --accent: #B44621;
  --success-bg: #EAF3DE; --success-text: #3B6D11;
  --warn-bg: #FBF3E0; --warn-text: #8A6D1E;
  --danger-bg: #FCEBEB; --danger-text: #A32D2D;
}
html, body, .stApp { background: var(--bg-page); color: var(--text-primary);
  font-family: 'Cairo', 'Segoe UI', Tahoma, sans-serif; }
h1, h2, h3 { color: var(--text-primary) !important; font-weight: 500; }
h1 { font-size: 24px !important; line-height: 1.2; margin: 0 0 4px !important; }
.stMainBlockContainer { background: transparent; border-radius: 12px; padding: 8px 16px; }
[data-testid="stCaptionContainer"] { margin: 0 0 2px !important; }
[data-testid="stCaptionContainer"] p { font-size: 13px; margin: 0 !important; }
[data-testid="stRadio"] { margin-bottom: 2px !important; }
[data-testid="stTabs"] { margin-top: 4px !important; }
[data-testid="stExpander"] summary { padding: 4px 0 !important; }

/* مبدّل اللغة: مؤشرات التابل — المختار accent، غير المختار رمادي شفاف */
.stRadio [role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
.stRadio [role="radiogroup"] label {
  background: transparent; border: 1px solid var(--border); border-radius: 8px;
  padding: 8px 20px; color: var(--text-secondary); font-weight: 400; cursor: pointer;
  font-family: 'Cairo', sans-serif;
}
.stRadio [role="radiogroup"] label:has(input:checked) {
  border-color: var(--accent); color: var(--accent); font-weight: 500;
}

/* أزرار: زر أساسي واحد accent فقط، الباقي أبيض/رمادي بحدود خفيفة */
.stButton button {
  background: var(--surface-card); border: 1px solid var(--border); border-radius: 8px;
  color: var(--text-primary); font-family: 'Cairo', sans-serif; font-weight: 400;
}
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--accent); border-color: var(--accent); color: #FFFFFF; font-weight: 500;
}

/* تبويبات Streamlit: النشط = accent بخط سفلي (v1.63: [data-testid=stTab] + نسخة قديمة [data-baseweb=tab]) */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {
  background: transparent; color: var(--text-secondary) !important; font-weight: 400;
  border-radius: 8px; padding: 8px 16px; font-family: 'Cairo', sans-serif;
}
.stTabs [data-baseweb="tab"]:hover, .stTabs [data-testid="stTab"]:hover { color: var(--accent) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-testid="stTab"][aria-selected="true"] {
  color: var(--accent) !important; font-weight: 500;
  border-bottom: 2px solid var(--accent) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== P1: كسر النصوص المخفية في الثيم الداكن — كل نص له تباين واضح مع خلفيته الفعلية ===== */
:root, .stApp, [data-testid="stAppViewContainer"] { color-scheme: light; }

/* تسميات العناصر (widget label للـradio/checkbox) — كانت #fafafa (غير مرئية) في الثيم الداكن */
[data-testid="stWidgetLabel"] { color: var(--text-primary) !important; }

/* captions (st.caption) — v1.63 تسميتهم stCaptionContainer لا .stCaption */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p { color: var(--text-secondary) !important; }

/* أزرار معطلة — نص داكن مقروء على خلفية رمادية فاتحة (كان نصاً فاتحاً شفافاً 40%) */
.stButton button:disabled, .stButton button[disabled=""] {
  color: var(--text-secondary) !important; background: #EEEDE8 !important;
  border-color: #DCD9CE !important; opacity: 1 !important; cursor: not-allowed; }

/* عناصر اختيار إضافية (selectbox/multiselect إن أُضيفت لاحقاً) — نص داكن/خلفية بيضاء دائماً */
[data-baseweb="select"], [data-baseweb="select"] * { color: var(--text-primary) !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] [role="listbox"] {
  background: var(--surface-card) !important; }
[data-baseweb="menu"] li { color: var(--text-primary) !important; }
[data-baseweb="menu"] li:hover { background: var(--success-bg) !important; color: var(--text-primary) !important; }

/* حلقة التركيز — بلون accent داكن بدل الأحمر الافتراضي (أوضح على زر الـaccent) */
.stButton button:focus-visible { box-shadow: 0 0 0 3px rgba(180, 70, 33, 0.55) !important;
  outline-color: var(--accent) !important; }
.stRadio [role="radiogroup"] label:focus-within { box-shadow: 0 0 0 2px rgba(180, 70, 33, 0.55); border-radius: 8px; }
.stTabs [data-testid="stTab"]:focus-visible, .stTabs [data-baseweb="tab"]:focus-visible {
  box-shadow: 0 0 0 2px rgba(180, 70, 33, 0.5); border-radius: 8px; }

/* كروت موحّدة: سطح أبيض + حدود 1px + r=12 */
.card { background: var(--surface-card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid var(--border) !important; border-radius: 12px !important; }

/* شارات الحالة: ألوان وظيفية فقط (أخضر/كهرماني/أحمر) */
.badge { display: inline-block; font-size: 12px; font-weight: 500; border-radius: 8px;
  padding: 3px 12px; border: 1px solid transparent; }
.badge-hand-ok { background: var(--success-bg); color: var(--success-text); border-color: var(--success-text); }
.badge-ai { background: var(--success-bg); color: var(--success-text); border-color: var(--success-text); }
.badge-hand-unknown { background: var(--warn-bg); color: var(--warn-text); border-color: var(--warn-text); }
.badge-hand-wait, .badge-index { background: transparent; color: var(--text-secondary); border-color: var(--border); }

/* الحرف الكبير — نص أساسي، غير واضح = كهرماني */
.big-slot { min-height: 86px; text-align: center; }
.big-letter { font-size: 72px; font-weight: 500; color: var(--text-primary); text-align: center; min-height: 78px; line-height: 78px; }
.big-letter-unknown { font-size: 54px; font-weight: 500; color: var(--warn-text); text-align: center; min-height: 78px; line-height: 78px; }
.big-letter-weak { font-size: 54px; font-weight: 400; color: var(--text-secondary); text-align: center; min-height: 78px; line-height: 78px; }

/* شريط الثقة: لون وظيفي حسب القيمة (أحمر <0.5، كهرماني 0.5–0.85، أخضر >0.85) */
.conf-wrap { background: var(--border); border-radius: 8px; height: 8px; width: 100%; margin: 8px auto 0; max-width: 340px; }
.conf-fill { height: 8px; border-radius: 8px; transition: width .12s ease-out; }

/* بلاطات الكلمة: مربعات 36×36 بحدود، لا نص عادي */
.word-stage { display: flex; flex-direction: column; align-items: center; gap: 8px; margin-top: 4px;
  max-height: 64px; overflow: hidden; }
.tilerow { display: flex; gap: 8px; min-height: 36px; align-items: center; justify-content: center;
  flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden; max-height: 40px; }
.tile { width: 36px; height: 36px; flex: none; display: inline-flex; align-items: center; justify-content: center;
        border: 1px solid var(--border); background: var(--surface-card); color: var(--text-primary);
        font-weight: 500; font-size: 20px; border-radius: 8px; }
.tile-spacer { width: 0; flex: none; }
.tile-count { color: var(--text-secondary); font-size: 12px; }

/* صف جملة الكلمات المنفصلة (P3): شرائح كلمات + مسافات واضحة عند العرض */
.sent-wrap { margin-top: 8px; max-height: 50px; overflow: hidden; }
.sent-row { display: flex; gap: 6px; align-items: center; justify-content: center; flex-wrap: wrap; }
.sent-row .wordchip { background: var(--surface-card); border: 1px solid var(--border); border-radius: 8px;
        padding: 2px 12px; font-size: 20px; font-weight: 500; color: var(--text-primary); }
.sent-sep { color: var(--text-secondary); }

/* فقاعات كشف AI والتراكم */
.reveal-stage { overflow: hidden; max-height: 72px; }
.reveal-stage .rev-raw { display: flex; gap: 3px; }
.reveal-stage .rev-raw .tile { width: 26px; height: 34px; font-size: 18px; border-color: var(--border); }
.rev-cor { font-size: 30px; font-weight: 500; color: var(--text-primary); text-align: center; }
.bub { align-self: flex-start; max-width: 82%; padding: 10px 14px; border-radius: 12px;
       background: var(--surface-card); border: 1px solid var(--border); font-size: 20px; color: var(--text-primary); }
.bub-ai { align-self: flex-end; background: var(--success-bg); border-color: var(--success-text); }
.bub .raw-mini { display: block; font-size: 12px; color: var(--text-secondary); padding-bottom: 2px; }
.bub-slot { display: none; }
.bub-wrap { max-height: 72px; overflow-y: auto; overflow-x: hidden; padding-right: 4px; scrollbar-width: thin; }
.meta { color: var(--text-secondary); text-align: center; }

/* القاموس: كروت بيضاء نظيفة — صورة + اسم + فئة بصرية واحدة، بلا badges نصية */
.dict-card { background: var(--surface-card); border: 1px solid var(--border); border-radius: 12px;
  padding: 12px 14px; margin: 8px 0; text-align: center; }
.dict-card img { border-radius: 8px; width: 100%; border: 1px solid var(--border);
  object-fit: contain; min-height: 92px; }
.dict-name { font-weight: 500; color: var(--text-primary); margin-top: 6px; font-size: 18px; }
.dict-cat { color: var(--text-secondary); font-size: 12px; margin-top: 4px; }

.stImage img { max-height: 320px; width: auto !important; margin: 0 auto; display: block; object-fit: contain; }
.stImageContainer { text-align: center; }

/* أزرار الحذف داخل كارت البلاطات فقط = خطر أحمر (بلا accent) */
[data-testid="stVerticalBlockBorderWrapper"] button[data-testid="stBaseButton-secondary"] {
  background: var(--danger-bg); border-color: var(--danger-text); color: var(--danger-text); font-weight: 500;
}

/* موجة الصوت أثناء التشغيل */
@keyframes wave { 0%,100% { transform: scaleY(.35); } 50% { transform: scaleY(1); } }
.wave-row { display: flex; gap: 4px; align-items: flex-end; justify-content: center; height: 26px; margin: 6px 0 2px; }
.wave-row .bar { width: 5px; height: 24px; border-radius: 3px; background: var(--accent); transform-origin: bottom;
                animation: wave 1s ease-in-out infinite; animation-play-state: paused; opacity: .85; }
.wave-row.playing .bar { animation-play-state: running; }

/* تفاصيل نصية متناسقة مع السمة */
.stMarkdown, .stCaption, .stText { font-family: 'Cairo', sans-serif; color: var(--text-primary); }
.stCheckbox label { color: var(--text-primary); }

/* حذف كل التدرجات والظلال مضمون أعلى: لا raw-gradient / لا box-shadow في الملف */
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def _load_class_map():
    try:
        return json.loads(engine.CLASS_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {str(i): {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]}
                for i in range(engine.EXPECTED_CLASSES)}


# تصنيف بصري لشكل اليد الفعلي في كل صورة قاموس عربي — مصدره: تعبئة المستخدم المرئية في dict_classes.tsv.
# (أُلغيت شمسية/قمرية نهائياً). الفئات: أصابع مفرودة / قبضة مغلقة / إبهام بارز.
_VIS_LABEL = {"extended": "أصابع مفرودة", "fist": "قبضة مغلقة", "thumb": "إبهام بارز"}
_VIS_LABEL_EN = {"extended": "Fingers extended", "fist": "Closed fist", "thumb": "Thumb prominent"}
DICT_CLASSES = {0: "extended", 1: "extended", 2: "thumb", 3: "extended", 4: "extended", 5: "extended",
                6: "thumb", 7: "fist", 8: "extended", 9: "extended", 10: "fist", 11: "extended",
                12: "extended", 13: "extended", 14: "extended", 15: "extended", 16: "extended",
                17: "extended", 18: "extended", 19: "extended", 20: "fist", 21: "extended",
                22: "extended", 23: "extended", 24: "extended", 25: "extended", 26: "extended",
                27: "extended", 28: "thumb", 29: "thumb", 30: "extended", 31: "extended"}

# P4: الكلمات العشر الفعلية المُدرَّبة لنموذج «كلمات إنجليزية» (engine_words_en) — مرجع تعليمي


def _cat(idx):
    return DICT_CLASSES.get(idx, "extended")


def _img_uri(idx):
    p = engine.DICT_DIR / f"class_{idx:02d}.png"
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


# ---- بند 1: عرض «تكوين» الحروف/الكلمات — HTML فقط، من بيانات الحالة نفسها (بلا لمس منطق) ----
_MAX_TILES = 40
_HIST_SLOTS = 8


def _conf_bar(pct):
    """شريط الثقة في كارت الكاميرا — لون وظيفي فقط حسب القيمة (أحمر <0.5، كهرماني 0.5–0.85، أخضر >0.85)."""
    if pct < 50:
        col = "var(--danger-text)"
    elif pct <= 85:
        col = "var(--warn-text)"
    else:
        col = "var(--success-text)"
    return (f"<div class='conf-wrap'><div class='conf-fill' "
            f"style='width:{pct}%; background:{col}'></div></div>")


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


def _sentence_md(words, lang):
    """الجملة كقائمة كلمات منفصلة (P3): شرائح بمسافات واضحة + سطر النص المتصل تحتها."""
    if not words:
        return ("<div class='meta'>No words yet — sign, then ␣ / silence</div>"
                if IS_EN else "<div class='meta'>لا كلمات بعد — أشر، ثم ␣ / الصمت</div>")
    chips = []
    for i, w in enumerate(words):
        if i:
            chips.append("<span class='sent-sep'>␣</span>")
        chips.append(f"<span class='wordchip'>{w}</span>")
    return ("<div class='sent-wrap'><div class='sent-row'>" + "".join(chips) + "</div>"
            f"<div class='meta'>{' '.join(words)}</div></div>")


def _reveal_md(rev, lang):
    """أنيميشن كشف فاعلية AI: التسلسل الخام يخفت ← المصحح يدخل أكبر/بلون مميز."""
    raw, text, src = rev
    ai = src == "groq" and raw != text
    badge = ("<span class='badge badge-ai'>AI ✨</span>" if ai
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
            tag = ("<span class='badge badge-ai'>AI ✨</span>" if ai
                   else "<span class='badge badge-index'>"
                        + tr("raw", "خام") + "</span>")
            if raw:
                seq = " · ".join(list(raw))
                mini = ("<span class='raw-mini'>" + tr("signed: ", "مُسجَّل: ") + seq + "</span>")
            else:
                mini = ""
            out.append(f"<div class='bub {'bub-ai' if ai else ''}' dir='{_DIRN}'>{mini}{text}{tag}</div>")
        else:
            out.append("<div class='bub-slot'></div>")
    return "<div class='bub-wrap'>" + "".join(out) + "</div>"


# --- مبدّل اللغة: يحوّل النموذج النشط/خريطة الفئات/الاتجاه/اللغة الصوتية/صور القاموس — بلا لمس العربي
lang = st.radio("اللغة / Language", ["عربي", "English"], horizontal=True, key="lang_toggle")
IS_EN = lang == "English"
if IS_EN and engine_en is None:
    st.error("English module unavailable — run: python engine_en.py --fetch-en then --train-en")


def tr(en, ar_):
    """ترجمة نص حسب اللغة النشطة — إنجليزية كاملة في الوضع الإنجليزي، عربية كاملة في العربي."""
    return en if IS_EN else ar_


# خلفية لغة-مخصوصة + توجيه خالص (LTR في الإنجليزي، RTL في العربي — بلا مزيج)
_DIRCSS = (":root, .stApp { direction: rtl; } h1, h2, h3 { text-align: right; }"
           if not IS_EN else
           ":root, .stApp { direction: ltr; } h1, h2, h3 { text-align: left; }")
st.markdown(f"<style>{_DIRCSS}</style>", unsafe_allow_html=True)

tab_live, tab_dict, tab_wlive = (st.tabs(["\U0001F3A5 Live Translation", "\U0001F4D6 Dictionary",
                                          "\U0001F44B English Words"])
                                 if IS_EN else
                                 st.tabs(["\U0001F3A5 ترجمة لحظية", "\U0001F4D6 القاموس",
                                          "\U0001F44B كلمات إنجليزية"]))

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
        cb1, cb2, cb3 = st.columns([2, 2, 3])
        with cb1:
            if st.button(tr("Start camera", "بدء التقاط الكاميرا"), type="primary", width="stretch",
                         key="btn_start_live"):
                if st.session_state.cap is None or not st.session_state.cap.isOpened():
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error(tr("Could not open camera (source=0)", "تعذر فتح الكاميرا (source=0)"))
                    else:
                        st.session_state.cap = cap
                st.session_state.live = True
        with cb2:
            if st.button(tr("Stop", "إيقاف"), width="stretch", key="btn_stop_live"):
                st.session_state.live = False
        with cb3:
            st.checkbox(tr("🔧 Debug diagnostics", "🔧 معلومات تشخيصية"), key="dbg_show")
        if has_key is False:
            st.info(tr("Set GROQ_API_KEY in .env to auto-correct text",
                       "أضف مفتاح Groq في .env ليُصحَّح النص تلقائياً"), icon="\U0001F511")

    if st.session_state.get("dbg_show"):
        _dl = "English" if IS_EN else "عربي"
        st.markdown(
            f"<div class='meta' style='text-align:center'>"
            f"🔧 {tr('language', 'اللغة')}: <b>{_dl}</b> · "
            f"{tr('checkpoint', 'النموذج')}: <b>{_PCMODEL.name}</b></div>",
            unsafe_allow_html=True)

    def _back_last():
        pl = st.session_state.get(pkey)
        if pl is not None and getattr(pl, "seq", None) is not None:
            pl.seq.backspace()
        st.session_state.pop(f"rev_{lang}", None)

    def _clear_word():
        pl = st.session_state.get(pkey)
        if pl is not None and getattr(pl, "seq", None) is not None:
            pl.seq.clear()
            if hasattr(pl, "reset_words"):
                pl.reset_words()
        st.session_state.pop(f"rev_{lang}", None)

    # P2+P3: مسار إغلاق موحّد (تلقائي عبر الصمت أو يدوي عبر الأزرار) — يُصحَّح ويُنطق ويُسجَّل في التاريخ.
    def _correct_synth(raw):
        if not raw:
            return None, None
        corr = engine.correct_word(raw, language="en") if IS_EN else engine.correct_word(raw)
        aud = engine.synthesize_speech(corr["text"], lang="en" if IS_EN else "ar")
        return corr, aud

    def _on_finalized(raw, corrected, audio):
        st.session_state[lkey] = raw
        entry = (raw, corrected["text"], corrected["source"])
        st.session_state[hkey].append(entry)
        st.session_state[f"rev_{lang}"] = entry
        if audio:
            audio_ph.audio(audio, format="audio/mp3", autoplay=True)
            st.session_state[f"aud_{lang}"] = audio
            st.session_state[f"wav_{lang}"] = time.perf_counter()

    # P2: عمودان ثابتان على شاشة قياسية — شمال: كاميرا فقط (مقيدة الارتفاع)،
    # يمين: الحرف + شريط الثقة + صف أزرار (✓/␣/✕/🗑) + البلاطات + السجل — معاً بلا سكرول.
    cam_col, res_col = st.columns([3, 4])
    with cam_col:
        cam_card = st.container(border=True)
        with cam_card:
            frame_ph = st.empty()
            hand_ph = st.empty()
    with res_col:
        letter_ph = st.empty()
        conf_ph = st.empty()
        act_row = st.columns(4)
        with act_row[0]:
            if st.button(tr("✓ Pin letter", "✓ تثبيت الحرف"),
                         key=f"btn_ok_commit_{lang}", width="stretch"):
                st.session_state[f"commit_{lang}"] = True
        with act_row[1]:
            if st.button(tr("␣ New word", "␣ كلمة جديدة"),
                         key=f"btn_new_word_{lang}", width="stretch"):
                st.session_state[f"new_word_{lang}"] = True
        with act_row[2]:
            if st.button(tr("✕ Delete last", "✕ حذف آخر حرف"),
                         key=f"wb_back_{lang}", width="stretch"):
                _back_last()
        with act_row[3]:
            if st.button(tr("🗑 Clear", "🗑 مسح الكل"),
                         key=f"wb_clr_{lang}", width="stretch"):
                _clear_word()
        tiles_ph = st.empty()
        sentence_ph = st.empty()
        reveal_ph = st.empty()
        status_ph = st.empty()
        wave_ph = st.empty()
        history_ph = st.empty()
        audio_ph = st.empty()


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

        # P2: ثبّت يدوياً — يُلتزم الحرف المعروض فوراً بلا debounce/ثقة، ويفتح كلمة عند امتلائها.
        if st.session_state.pop(f"commit_{lang}", False):
            pl = st.session_state.get(pkey)
            cand = st.session_state.get(f"cand_{lang}")
            if pl is not None and cand and cand[0] is not None:
                ev = pl.seq.force_commit(cand[0], cand[1])
                if ev["finalized"]:
                    corr, aud = _correct_synth(ev["finalized"])
                    _on_finalized(ev["finalized"], corr, aud)
            else:
                status_ph.warning(tr("No letter to pin yet",
                                     "لا حرف واضح للتثبيت بعد"))

        # P3: كلمة جديدة يدوياً — يغلق الكلمة الحالية فوراً ويبدأ غيرها (بجانب الصمت التلقائي).
        if st.session_state.pop(f"new_word_{lang}", False):
            pl = st.session_state.get(pkey)
            if pl is not None:
                raw = pl.seq.finalize_now()
                if raw:
                    r = pl.push_word(raw)
                    if r:
                        _on_finalized(raw, r["corrected"], r["audio"])
            else:
                st.session_state[f"new_word_{lang}"] = True  # أعدها — الأنبوب ليس جاهزاً

        out = st.session_state[pkey].update(frame)
        if out.get("error"):
            st.session_state.live = False
            status_ph.error(out["error"])
            return
        st.session_state[f"cand_{lang}"] = (
            (int(out["idx"]), float(out["conf"]))
            if out["hand"] and out["idx"] is not None and not out.get("unknown") else None)
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

        if out["hand"]:
            conf_ph.markdown(_conf_bar(int(out["conf"] * 100)), unsafe_allow_html=True)
        else:
            conf_ph.markdown("")

        if out["hand"] and out["idx"] is not None and not out.get("unknown"):
            below = out["conf"] < engine.CONF_THRESHOLD
            if engine._TRACE:
                print("[TRACE UI] big-slot shows candidate", repr(out["label"]),
                      f"conf={out['conf']:.4f}", "th=", engine.CONF_THRESHOLD,
                      "BELOW_TH_DISPLAYED=", below, flush=True)
            cls = "big-letter-weak" if below else "big-letter"
            if IS_EN:
                meta = (f"proposed · confidence {out['conf']:.2f}" if below
                        else f"confidence {out['conf']:.2f}")
            else:
                meta = (f"مقترح · الثقة {out['conf']:.2f}" if below
                        else f"{out['label_en']} — الثقة {out['conf']:.2f}")
            letter_ph.markdown(
                f"<div class='big-slot'><div class='{cls}'>{out['label']}</div>"
                f"<div class='meta'>{meta}</div></div>",
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
        if engine._TRACE:
            print("[TRACE UI] tiles_word=", repr(out["word"]), "commit_count=", len(out["word"] or ""), flush=True)
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
            _on_finalized(out["events"]["finalized"], out["corrected"], out["audio"])

        sentence_ph.markdown(_sentence_md(out.get("words", []), lang), unsafe_allow_html=True)

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
                width="stretch",
                key="btn_dl_audio")

    st.divider()
    with st.expander(tr("How it works (click to expand)", "كيف تعمل؟ (اضغط للتفاصيل)")):
        st.markdown(tr(
            "**How it works:** hold a sign for 5 of the last 7 frames, each with confidence ≥ 0.90 "
            "and a top1−top2 margin ≥ 0.05, to commit a letter (majority vote over a sliding "
            "window — one stray frame no longer resets progress) — or press **✓ Pin** to commit "
            "the shown letter instantly (bypasses the window/confidence). **␣ New word** closes the "
            "current word immediately (same as the 2.5 s silence auto-close — both work). Completed "
            "words form a separate list shown with clear spaces; Groq corrects the full sentence "
            "and the audio speaks it with word pauses.",
            "**كيف تعمل:** اعرض إشارةً لـ **5 من آخر 7 إطارات** بثقة ≥ 0.90 مع تحقّق هندسي "
            "ليُلتزم الحرف — تصويت أغلبية على نافذة انزلاقية، فإطار واحد مختلف لا يصفّر التقدّم "
            "(على عكس «إطارات متتالية» سابقاً). أو اضغط **✓ تثبيت** لتُثبّت الحرف المعروض فوراً "
            "(يتجاوز النافذة/الثقة). زر **␣ كلمة جديدة** يغلق الكلمة الحالية فوراً (مثل آلية "
            "الصمت 2.5 ثانية — كلاهما يعمل). الكلمات المكتملة تتراكم كقائمة منفصلة تُعرض بمسافات "
            "واضحة؛ Groq يصحّح الجملة كاملة ويُنطقها الصوت بفواصل كلمات."))

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
            return ""

        def _dict_uri(i):
            p = engine_en.DICT_EN_DIR / f"class_{i:02d}.png"
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else None

    else:
        n = engine.EXPECTED_CLASSES
        cmap_ar = _load_class_map()

        def _dict_info(i):
            return cmap_ar.get(str(i), {"sym": engine.CLASS_SYMS[i], "name": engine.CLASS_NAMES[i]})

        def _dict_cat(i):
            return _cat(i)

        def _dict_uri(i):
            return _img_uri(i)

    cols = st.columns(4)
    for i in range(n):
        info = _dict_info(i)
        cat = _dict_cat(i)
        uri = _dict_uri(i)
        card = f"<div class='dict-card'>"
        if uri:
            card += f"<img src='{uri}' alt='{info['name']}' loading='lazy'/>"
        card += f"<div class='dict-name'>{info['name']}</div>"
        if cat:
            card += f"<div class='dict-cat'>{tr(_VIS_LABEL_EN[cat], _VIS_LABEL[cat])}</div>"
        card += "</div>"
        with cols[i % 4]:
            st.markdown(card, unsafe_allow_html=True)


# ---- بند 4: «كلمات إنجليزية» — نظام رابع منفصل (engine_words_en.py) — تحميل كسول، لا يمس الثلاثة العليا ----
@st.cache_resource
def _words_live_pipe():
    import engine_words_en as ew
    return ew.WordsLive(), ew


with tab_wlive:
    st.title(tr("English Words — Live", "كلمات إنجليزية — حيّ"))
    if IS_EN:
        st.caption("LSTM over 40 landmark frames · 10 words (hello, thankyou, please, yes, no, bye, "
                   "drink, water, happy, sleep) · held-out signers test = 63.6%")
    else:
        st.caption("LSTM على 40 إطاراً من اليدين · 10 كلمات (hello, thankyou, please, yes, no, bye, "
                   "drink, water, happy, sleep) · اختبار على مشاركين جدد = 63.6%")

    wlive = "wlive_on"
    wcap = "wcap_on"
    if wlive not in st.session_state:
        st.session_state[wlive] = False
        st.session_state[wcap] = None

    _wt = st.session_state.get("words_en_ckpt")
    _st = st.caption(tr("Model: " + ("trained ✓" if _wt else "not trained yet"),
                        "النموذج: " + ("مُدرَّب ✓" if _wt else "ليس مُدرَّباً بعد")))
    try:
        from pathlib import Path
        _ck = next(Path("models").glob("words_en.keras"), None)
        if not _ck and not _wt:
            st.error(tr("No checkpoint — run: python engine_words_en.py --train",
                        "لا يوجد نموذج — شغّل: python engine_words_en.py --train"))
    except Exception:
        pass

    wcol_ctrl = st.columns([1, 3])[0]
    with wcol_ctrl:
        if st.button(tr("Start words camera", "بدء كاميرا الكلمات"), type="primary", width="stretch",
                     key="btn_start_words"):
            if st.session_state[wcap] is None or not st.session_state[wcap].isOpened():
                _cap = cv2.VideoCapture(0)
                if not _cap.isOpened():
                    st.error(tr("Could not open camera (source=0)", "تعذر فتح الكاميرا (source=0)"))
                else:
                    st.session_state[wcap] = _cap
                    st.session_state[wlive] = True
            else:
                st.session_state[wlive] = True
        if st.button(tr("Stop", "إيقاف"), width="stretch", key="btn_stop_words"):
            st.session_state[wlive] = False

    wcam_col, wres_col = st.columns([3, 4])
    with wcam_col:
        _wc = st.container(border=True)
        with _wc:
            ws_frame = st.empty()
    with wres_col:
        ws_word = st.empty()
        ws_conf = st.empty()
        ws_status = st.empty()

    @st.fragment(run_every=0.1)
    def words_loop():
        if not st.session_state.get(wlive, False):
            return
        cap = st.session_state.get(wcap)
        if cap is None or not cap.isOpened():
            st.session_state[wlive] = False
            return
        try:
            pl, _ew = _words_live_pipe()
        except Exception as _e:
            st.session_state[wlive] = False
            ws_status.error(f"{_e}")
            return
        ok, frame = cap.read()
        if not ok:
            return
        out = pl.update(engine.downscale_live(frame))
        if out.get("error"):
            st.session_state[wlive] = False
            ws_status.error(out["error"])
            return
        ws_frame.image(out["overlay"], channels="BGR", width="stretch", output_format="JPEG")
        if out["ready"] and out["word"]:
            ws_word.markdown(f"<div class='big-slot'><div class='big-letter'>{out['word']}</div>"
                             f"<div class='meta'>confidence {out['conf']:.2f}</div></div>",
                             unsafe_allow_html=True)
            ws_conf.markdown(_conf_bar(int(out["conf"] * 100)), unsafe_allow_html=True)
        elif out["hand"]:
            ws_word.markdown("<div class='big-slot'><div class='meta'>Collecting frames…</div></div>",
                             unsafe_allow_html=True)
            ws_conf.markdown("")
        else:
            ws_word.markdown("<div class='big-slot'><div class='meta'>Waiting for a hand…</div></div>",
                             unsafe_allow_html=True)
            ws_conf.markdown("")

    words_loop()

    st.divider()
    with st.expander(tr("English-Words how it works (click to expand)", "كيف تعمل الكلمات؟ (اضغط للتفاصيل)")):
        st.markdown(tr(
            "**How it works:** English ISLR — MediaPipe two-hand landmarks are re-sampled to a fixed "
            "40-frame window and classified by an LSTM. A hold a hand (~4 s) to fill the window, drop "
            "it for 1 s to start a new word.",
            "**كيف تعمل:** نظام ISLR إنجليزي مستقل — لاندماركات اليدين تُعاد عيناتها إلى نافذة 40 إطاراً "
            "ثابتة ويصنّفها LSTM. أبقِ يدك (~4 ثوانٍ) لتمتلئ النافذة، ثم فارق يد 1 ثانية لبدء كلمة جديدة."))

st.markdown(tr(
    "<div class='meta' style='margin-top:24px'>Arabic: val 95.18% · 32 ArASL signs · "
    "English: val 100% · 24 ASL signs · Words (EN): held-out test 63.6% · 10 words — "
    "local Streamlit interface (M3–M11 + P1 + EN + W)</div>",
    "<div class='meta' style='margin-top:24px'>العربي: val 95.18% · 32 إشارة ArASL · "
    "English: val 100% · 24 ASL signs · الكلمات (إنج): اختبار 63.6% · 10 كلمات — "
    "واجهة محلية Streamlit (M3–M11 + P1 + EN + W)</div>"),
    unsafe_allow_html=True)