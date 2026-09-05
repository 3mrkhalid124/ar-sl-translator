# PROJECT_MAP

> معرّف البنية الحية. يُحدَّث مع كل تغيير هيكلي، وأي جزء غير مكتمل يُسجَّل فوراً في [ORPHANS & PENDING].
> Last verified: 2026-09-05 · OS: Windows · Root: `C:\Users\Amr\Documents\ar-sl-translator`

## [TECH_STACK]

- Python **3.11** (venv `.venv`، `py -3.11`) — نظام الجهاز 3.14 لكن MediaPipe تُصنَّف حتى 3.12 فقط؛ 3.11 يغطي كل الأطر.
- واجهة: **streamlit 1.63.0** (تبويبان فقط) + CSS مضمّن (خط Cairo/Tajawal، RTL).
- كاميرا/قصّ يد: **opencv 4.14.0.94** (مثبَّت مزدوجاً `opencv-python` + `opencv-contrib-python` بنفس النسخة لمنع تعارض `cv2`) + **mediapipe 1.0.1** عبر **Tasks API فقط** (`mediapipe.tasks.python.vision.HandLandmarker`، لأن `mediapipe.solutions` أُزيل نهائياً) + أصل `assets/hand_landmarker.task` (7.8MB).
- مصنّف CNN: **torch 2.14.0+cpu** (عجل CPU؛ TensorFlow مستبعد أثقل).
- بيانات/علم: **pandas 3.0.5 · numpy 2.2.6 · scikit-learn 1.9.0**.
- LLM: **Groq** عبر **requests 2.34.2** (`https://api.groq.com/openai/v1/chat/completions`، موديل **`allam-2-7b`** عربي-مختص — بديل `llama-3.3-70b-versatile` الذي أُزيل من الكتالوج 2026). المفتاح من `.env`/env فقط، يُطلب من المستخدم عند الحاجة الفعلية فقط.
- صوت: **gTTS 2.5.4** (عربي، يحتاج إنترنت).
- Env: **python-dotenv 1.2.2**.
- الداتا: **ArASL2018** (Mendeley، DOI 10.17632/y7pckrw6z2.1، CC BY 4.0) — 54,049 صورة رمادي 64×64، 32 class، CSV labels. رابط مباشر بلا تسجيل (sha256 متحقَّق في M2). بديل احتياطي: HF parquet `pain/ArASL_Database_Grayscale`.

## [SYSTEM_FLOW]

```
تبويب 1 (ترجمة لحظية):
  كاميرا cv2.VideoCapture(0) → HandLandmarker (VIDEO mode) → bbox اليد → قصّ → رمادي 64×64 مقسّم [0,1]
  → CNN classify → تحقق هندسي (عدّ الأصابع + إبهام من 21 نقطة؛ 4 مراسي مؤكدة) → رفض → «غير معروف» (idx=None)
  → gate (N إطارات متتالية same argmax + ثقة ≥0.90) → حرف ملتزم → خط الكلمة
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
- الإنتاجية: عتبة ثقة 0.90 + debounce 5 إطارات + طبقة تحقق هندسي (P1) + صمت 2.5ث — متغيرات قابلة للضبط في engine.py.
- Groq: لا SDK، requests مباشر، نص عربي صريح، JSON صارم اختياري → استخراج نص.

## [ORPHANS & PENDING]

- [M1 ✔ verified] بيئة: كل الحزم تعمل تحت py3.11؛ HandLandmarker 1.0.1 يُحمَّل ويستجيب؛ hand_landmarker.task (7,819,105 bytes) يشتغل. numpy=2.2.6 (وليس 2.5.1 لأن numpy 2.5 يشترط ≥3.12).
- [M2 ✔ verified] الداتا: `--fetch-data` نزّل 54,049 صورة (تشمل 638 صورة 256×256 و10 صور 1024×768 ← أُعيد تنسيقها لـ64×64)، 32 صنفاً، class 0=عين/2102 والأكبر، 30=ياء/1293 الأصغر (يطابق ورقة الداتا)، 32 صورة قاموس في assets/dict، تخزين `data/arasl.npz`. **ملاحظة مصدر:** روابط Mendeley المباشرة محجوبة بـCloudflare لطلبات غير-متصفح (403/turnstile) — المصدر المعتمد مرآة HF `pain/ArASL_Database_Grayscale` (نفس الداتا، CC BY 4.0). `Signs_32_New.png` المرجعية أيضاً خلف Cloudflare → تبقى اختيارية.
- [M3 ✔ verified] التدريب: `--train` استأنف من `models/cnn.pt` (معمارية slim، contiguous، CPU_THREADS=6، BATCH_SIZE=512) → epoch1 val 0.8997 · epoch2 0.9406 · **epoch3 val 0.9518 ≥ 0.95 → استيفاء الهدف وتوقف مبكر** (القاعدة الجديدة: حد أقصى 3 epochs لكل جلسة). `models/cnn.pt` (730KB) + `models/class_map.json` (32 صنفاً UTF-8) محفوظان؛ `[SELFTEST] train: PASS best_val_acc=0.9518`. أضعف 3 أصناف: قاف 0.816، ثاء 0.894، زاي 0.908.
- [M4 ✔ رمزياً] الكاميرا: `detect_hand` (HandLandmarker Tasks VIDEO، singleton)، `crop_hand_patch` (مربع ±هادي 1.4 + تطبيع قطبية)، `process_frame`، `run_camera` (`--camera`). الاختبار الاصطناعي للحلقة نجح؛ **كاميرا الجهاز الفعلية لم تُختبر في البيئة** (لا جهاز مرفقloadable).
- [M5 ✔ verified] `SignSequencer`: التزام بحرف بعد DEBOUNCE_FRAMES إطارات بثقة ≥ CONF_THRESHOLD، بلا تكرار صناعي لنفس الحرف حتى انقطاع/تغيّر.
- [M6 ✔ verified] إغلاق الكلمة عند صمت SILENCE_SECONDS بلا يد + حد MAX_WORD_LEN؛ `LivePipeline.update(frame)` يستهلك كل M4–M6 في result واحد.
- [M7 ✔ verified] `correct_word`: Groq عبر requests. **مفتاح المستخدم أُضيف `.env` (مستثنى من git).** لاحظنا 404 على `llama-3.3-70b-versatile` (أُزيل من الكتالوج 2026) → اعتُمد **`allam-2-7b`** (عربي-مختص) بعد فحص `/v1/models` (14 موديلاً متاحاً). اختبار حي: `سلم→سَلِمْتَ`، `بسمله→بسم الله`، زمن ~0.8s.
- [M8 ✔ verified] `synthesize_speech`: gTTS ar → mp3 (اختبار حي أنتج 10752 بايت بإنترنت)؛ تشغيل عبر `st.audio` في الواجهة.
- [M9 ✔ verified] تبويب القاموس: شبكة 8×4 بطاقات (صورة مرجعية + رمز + اسم) من `assets/dict/*.png` + `class_map.json`.
- [M10 ✔ verified] الواجهة: `app.py` Streamlit (نسخة 1.63) بتبويبين، RTL/خط Cairo CSS مضمّن، حلقة حية `st.fragment(run_every=0.1)` مع إيقاف تشغيل آمن، تاريخ الكلمات، `st.audio` تشغيل تلقائي. فحص `AppTest`: تبويبان بلا استثناءات؛ الخادم أقلع headless بخدمة 8599.
- [M11 ✔ committed] إغلاق: `--selftest` شامل (16 فحصاً) + تحديث هذا الملف.
- [P1 ✔ committed] تحسين الدقة (التحقق الهندسي): `_LM`/`_FINGER_LINKS`/`finger_features`/`validate_geometry` من نقاط HandLandmarker الـ21، مع جدول **SIGNPAT بمراسي أربعة مؤكدة من الصور المرجعية في `assets/dict/`** (فاء=قبضة+إبهام جانبي، سين=كف مفتوح، ثاء=W بثلاث أصابع، ياء=سبابة لأعلى — تأكيد المستخدم، بلا أصناف إضافية مفترضة). قواعد الرفض: فرق عدد أصابع ≥2 يُرفض؛ تعارض إبهام وحده لا يُرفض؛ الصنف بلا مرساة لا يُقيَّد. عند الرفض → `process_frame` يُرجع حالة `unknown` (idx=None, conf=0) ويظهر «غير معروف / مش واضح» في overlay، و`SignSequencer` يعامل «يد بلا تصنيف» كحالة ثالثة (توقيت اليد فقط، صفر التزام، بلا إنهاء كلمة مبكر). الثوابت: **CONF_THRESHOLD 0.90 + DEBOUNCE_FRAMES 5**. سلبيات معروفة مفتوحة: لا "مسح قياسي" ممكن على الداتا (HandLandmarker لا يلتقط صور ArASL المرسومة — مسح أعطى 0–9/80)؛ الشكل الهندسي الثابت لـ ألف(2)/ياء(30) متطابق تقريباً فيُحسم بالثقة/الـCNN لا بالهندسة؛ `geo.*` فحوص selftest 10 إضافية → `--selftest` **ALL PASS** (32 فحصاً).
- [P2 🔜 PENDING] تحسينات واجهة app.py (كروت قاموس ملونة/badges/hover، شارة حالة اليد، شريط ثقة حي، عرض قبل/بعد تصحيح Groq، مع الحفاظ على Cairo/RTL) — بانتظار اكتمال بعد P1 per أمر المستخدم.
- [NOTE] الاختبار اليدوي للأوضاع الأربعة (سبابة/كف/قبضة/W) على كاميرا المستخدم مطلوب — لا كاميرا في بيئة التطوير.
- [NOTE] Groq مفعّل بمفتاح المستخدم في `.env` (M7). عرض «قبل/بعد» التصحيح على الواجهة ضمن P2.
- [PENDING] قوالب إشارات لكلمات كاملة (توقيع حرف بحرف يُجمع في كلمات) — **تطوير مستقبلي فقط، لا يُنفَّذ الآن**؛ تُستخدم الحروف المفردة مع تصحيح Groq.
- [NOTE] `models/` مستبعد من git (قابل لإعادة الإنتاج عبر `--train`); `logs/` و`data/*.parquet` أيضاً. `.env` و`.venv` مستبعدان.