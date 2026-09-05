# PROJECT_MAP

> معرّف البنية الحية. يُحدَّث مع كل تغيير هيكلي، وأي جزء غير مكتمل يُسجَّل فوراً في [ORPHANS & PENDING].
> Last verified: 2026-09-05 · OS: Windows · Root: `C:\Users\Amr\Documents\ar-sl-translator`

## [TECH_STACK]

- Python **3.11** (venv `.venv`، `py -3.11`) — نظام الجهاز 3.14 لكن MediaPipe تُصنَّف حتى 3.12 فقط؛ 3.11 يغطي كل الأطر.
- واجهة: **streamlit 1.63.0** (تبويبان فقط) + CSS مضمّن (خط Cairo/Tajawal، RTL).
- كاميرا/قصّ يد: **opencv 4.14.0.94** (مثبَّت مزدوجاً `opencv-python` + `opencv-contrib-python` بنفس النسخة لمنع تعارض `cv2`) + **mediapipe 1.0.1** عبر **Tasks API فقط** (`mediapipe.tasks.python.vision.HandLandmarker`، لأن `mediapipe.solutions` أُزيل نهائياً) + أصل `assets/hand_landmarker.task` (7.8MB).
- مصنّف CNN: **torch 2.14.0+cpu** (عجل CPU؛ TensorFlow مستبعد أثقل).
- بيانات/علم: **pandas 3.0.5 · numpy 2.2.6 · scikit-learn 1.9.0**.
- LLM: **Groq** عبر **requests 2.34.2** (`https://api.groq.com/openai/v1/chat/completions`، موديل `llama-3.3-70b-versatile` — مُختبَر مسبقاً على هذا الجهاز). المفتاح من `.env`/env فقط، يُطلب من المستخدم عند الحاجة الفعلية فقط.
- صوت: **gTTS 2.5.4** (عربي، يحتاج إنترنت).
- Env: **python-dotenv 1.2.2**.
- الداتا: **ArASL2018** (Mendeley، DOI 10.17632/y7pckrw6z2.1، CC BY 4.0) — 54,049 صورة رمادي 64×64، 32 class، CSV labels. رابط مباشر بلا تسجيل (sha256 متحقَّق في M2). بديل احتياطي: HF parquet `pain/ArASL_Database_Grayscale`.

## [SYSTEM_FLOW]

```
تبويب 1 (ترجمة لحظية):
  كاميرا cv2.VideoCapture(0) → HandLandmarker (VIDEO mode) → bbox اليد → قصّ → رمادي 64×64 مقسّم [0,1]
  → CNN classify → gate (N إطارات متتالية same argmax + ثقة ≥0.85) → حرف ملتزم → خط الكلمة
  → صمت 2.5ث (لا يد) → finalize → Groq تصحيح (مفتاح عند الحاجة) → gTTS ar → mp3 → تشغيل تلقائي st.audio

تبويب 2 (قاموس): اختيار من 32 إشارة → صورة مرجعية من الداتاسيت (assets/dict/*.png) + الاسم + لوحة Signs_32_New.png

Logging لا-حظري: queue.Queue + Listener thread → logs/app.log (تصنيف/حرف/ثقة/t، LLM in/out/latency، أخطاء). صفر مفاتيح في السجلات.
```

## [ARCHITECTURE]

```
ar-sl-translator/
├── app.py        # واجهة Streamlit فقط: CSS مضمّن، تبويب ترجمة لحظية، تبويب قاموس. بلا منطق تعليمي/كشف.
├── engine.py     # كل المنطق: داتا، CNN (torch)، HandLandmarker cropper، LivePipeline، Groq REST، gTTS، logging.
├── requirements.txt   # مثبَّت (انظر TECH_STACK). torch من pytorch CPU index فقط.
├── .env               # GROQ_API_KEY محلي (يبقى خارج git). .gitignore يستثنيه.
├── PROJECT_MAP.md
├── data/   (الداتا المصدر/parquet) · models/ (cnn.pt + class_map.json) · assets/ (hand_landmarker.task + dict/*.png) · logs/
└── .venv/ (معزول)
```

قرارات ملزمة:
- ملفا كود فقط (app.py + engine.py). صفر Placeholders / TODO. كل مسار أخطاء مسجَّل ويُعرض للمستخدم.
- CNN في engine.py فقط، تُدرَّب عبر `python engine.py --train` (لا تدريب من الواجهة — النطاق مقفول).
- الإنتاجية: عتبة ثقة 0.85 + debounce 3 إطارات + صمت 2.5ث — متغيرات قابلة للضبط في engine.py.
- Groq: لا SDK، requests مباشر، نص عربي صريح، JSON صارم اختياري → استخراج نص.

## [ORPHANS & PENDING]

- [M1 ✔ verified] بيئة: كل الحزم تعمل تحت py3.11؛ HandLandmarker 1.0.1 يُحمَّل ويستجيب؛ hand_landmarker.task (7,819,105 bytes) يشتغل. numpy=2.2.6 (وليس 2.5.1 لأن numpy 2.5 يشترط ≥3.12).
- [PENDING M2] تنزيل ArASL2018 من Mendeley + تحقق sha256 + فك ضغط + تحقق 54,049/32 class/64×64 + استخراج صور القاموس. (المسار المعتمد للتنزيل المباشر: `https://data.mendeley.com/public-files/datasets/y7pckrw6z2/files/<file_id>/file_downloaded`).
- [PENDING M3] تدريب CNN (target val ≥ 95%).
- [PENDING M4–M8] كاميرا/سلسلة/كلمة/Groq/TTS.
- [PENDING M9] تبويب القاموس.
- [PENDING M10] التصميم البصري.
- [NOTE] كاميرا الجهاز لم تُختبر بعد (أول تشغيل M4).
- [NOTE] gTTS وGroq يحتاجان إنترنت؛ الفشل يمرّ بلا كسر (تسجيل + رسالة).