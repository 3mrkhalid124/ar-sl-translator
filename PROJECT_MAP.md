# خريطة مشروع مترجم لغة الإشارة — الحالة النهائية (06-09-2026)

## البنية (4 أنظمة مستقلة)

```
C:\Users\Amr\Documents\ar-sl-translator
├── app.py                  # واجهة Streamlit — 3 تبويبات (حروف حي، قاموس، كلمات حي) + مبدّل لغة
├── engine.py               # النظام العربي: CNN على ArASL، SignSequencer، SIGNPAT، live، Groq/gTTS، selftest (AR+EN)
├── engine_en.py            # النظام الإنجليزي حروف: CNN على ASL-MNIST، SIGNPAT_EN، LivePipelineEN (يتكامل مع selftest)
├── engine_words_en.py      # النظام الإنجليزي كلمات ISLR: LSTM على landmarks، WordsLive، --train/--train-v2/--eval
├── selftest_all.py         # بوابة فحص موحّدة للأربعة (عربي+إنجليزي حروف+كلمات+واجهة) → 0 = صفر تراجع
├── apptest_sl.py           # AppTest لواجهة Streamlit (كل الأزرار/اللغات، صفر DuplicateElementId)
├── features/
│   └── hand_landmarker.task    # نموذج MediaPipe (يُحمَّل عبر engine.fetch)
├── models/                 # cnn.pt · cnn_en.pt · words_en.keras + words_en_classes.json  (gitignored)
└── data/                   # arasl.npz · asl_mnist.npz · asl_signs/(manifest+parquet+scaler/split)  (gitignored)
```

## النماذج والدقة المعتمدة (أحدث قيمة على القرص)

| النظام | النموذج | المعيار | القيمة |
|---|---|---|---|
| عربي | `models/cnn.pt` | val (85/15) | **97.05%** (جولة oversampling — commit 1fa9905) |
| إنجليزي حروف | `models/cnn_en.pt` | val | **≈100%** |
| كلمات إنجليزية | `models/words_en.keras` | test held-out (مشاركون 2044/4718) | **63.64%** (21/33) |

## سجلّ الجلسات والعمل المنجز (أحدث أولاً)

1. **جلسة التشخيص والتصميم** — خط الأساس القديم (`PROJECT_MAP` §1):
   - لا تجاوز للعتبة فيلم حي (raw_conf 0.3955 + geo accept) → فشل الإرسال القديم؛ المستخدم تخلى عن التصوير الحي → الفحص عبر الصور الموثقة (canonical أول exemplar npz + DICT_EN).
   - فحص النموذج offline: AR in-domain 0.966/معكوس 0.156/perturbed 0.281؛ EN in-domain 1.0/معكوس 0.208/perturbed 0.167.
   - **Step A**: `normalize_live` (قلب الداكن فقط) استرداد الداكن 0.156→0.688 عربي و0.208→1.000 إنجليزي؛ margin_accept عربي MARGIN 0.05 + بوابة `ok_mar` في process_frame.
   - **Step B**: `_augment_batch` (flip/rotate/scale/brightness/blur) موصّلة في train_cnn وtrain_en؛ رفض calibration حراري بالدليل (softmax حاد).
   - القطبية السليمة للحي: الـ12 crop في المسار الجديد كلها ≥0.97–0.99 (ثبات بدل تذبذب).
   - **commit 42df59c**: إعادة بناء الواجهة بالضوء الصارم (متغيرات CSS فاتحة، كروت قاموس = صورة+اسم+فئة واحدة، big-slot = التزام+مقترح).

2. **Goal1 – أداء الكاميرا (commit 0392807)**: `downscale_live(max 640)` + JPEG + تصحيح timestamp المشترك (التجمّد عند تبديل اللغة) → `LivePipeline.update ≈ 28ms/p90 31ms/0 أخطاء`؛ FrameProfiler لكل 60 إطاراً.
3. **Goal2 – إنجليزي SIGNPAT (ea7138e)**: تجميعات عدد الأصابع الكنسية (18 حرفاً، تسامح ±1، thumb=any) بقياس الداتا؛ `validate_geometry(patterns=)` معمّم؛ 8 selftests.
4. **Goal2.2–2.3 (UI smoothness + mp3)** و**Goal2.4 وثائق**.
5. **سلسلة UI-1..UI-6** (`تسميع البلاطات/الحروف + فقاعات محادثة`، `مؤشرات أدق`، `هوية إسلامية-هندسية للأوضاع`، `L10N كامل عربي/إنجليزي بتوجيه LTR/RTL`، `تخطيط عمودَين حي + capped 380px`، `فئات بصرية للقاموس`، `تاريخ كلمات لا badges`، `Common Words tab`، `حذف آخر حرف + مسح`). — انتهى بإعادة ضبط التصميم النهائي في 42df59c.
6. **Polish (جولة نقدية · 4c3150c → 842edaf)**: حذف كود ميت (import time/MOD/col_state)، رفع محوّل خريطة عربي للموديول، safe fallback map_en، إعادة بناء أرتيفاكتات EN عند الغياب، `train_en` يحفظ أفضل checkpoint فقط، تحديث وثائق.

7. **P1 (حرج · 615a05b): مفاتيح فريدة لكل زر** — StreamlitDuplicateElementId صفر (AppTest 8 حالات × لغتين).
8. **P2 (8299df2): زر «✓ تثبيت الحرف» اليدوي** — `SignSequencer.force_commit/idx,conf` (يحتترم max_word + بوابة لا-تكرار) + `finalize_now()`؛ path موحّد `_on_finalized` للتلقائي واليدوي؛ selftests force_commit (fast/full/overflow/gate) + finalize_now AR/EN.
9. **P3 (a5429a1): كلمات منفصلة بمسافات** — `words[]/sentence` في pipeline؛ `push_word(raw)` تصحيح Groq للجملة كاملة + TTS بفواصل؛ زر «␣ كلمة جديدة» (`btn_new_word_{lang}`) يغلق الكلمة فوراً (مع الصمت 2.5ث)؛ عرض `_sentence_md` بشرائح + سطر نص بمسافات؛ selftests p3.* AR/EN.
10. **P4a (28ffbbd): تنظيف التبويبات** — حذف "Common Words" الثابت (العربي+الإنجليزي)؛ الإبقاء على "English Words" الحي؛ في القاموس الإنجليزي مبدّل **Letters/Words** يعرض العشر كلمات المُدرَّبة (hello..sleep) بطاقات مرجعية.
11. **P4.1 (c0e5656): تدريب إضافي للكلمات — لا تحسين** — قياس الحقيقة: 297 ملف parquet الإضافية على القرص هي إشارات **أخرى** (dog/cat/food…) وليست كلمات؛ لذا أُجري بحث وصفات+seeds (`--train-v2`) على نفس الـ320 (نفس تقسيم المشاركين، سقف 20 دقيقة): أفضل = 63.64% (تعادل) → **لا استبدال**؛ فحص شامل للداتا موثّق في رسالة الالتزام.
12. **P4.2 (8a8df4e): بوابة صفر-تراجع موحّدة** — `selftest_all.py` يتحقق من الأربعة معاً؛ `apptest_sl.py` نُقل إلى الريبو (مسار نسبي). التشغيل: `engine.py --selftest` ALL PASS، `words_en --eval` PASS-0.6364، `AppTest` PASS.
13. **P4.3 (4e60530): Polish ثانية** — حذف `_augment_train` اليتيم (engine.py)، `import numpy as np` غير المستخدم (app.py)، CSS `.dict-sym` اليتيم (من كروت Common Words)، `Path as _P` غير المستخدم (engine_en.py). إعادة تشغيل selftest_all: **zero regression**.

## الحالة (Checklist نهائي)

- [x] عربي 97.05% · إنجليزي حروف ~100% · كلمات استثمار group-held-out 63.64%
- [x] أزرار القابلة للنقر كلها بمفاتيح فريدة (P1) — AppTest 0/Duplicate
- [x] تثبيت يدوي + كلمة جديدة يدوية (P2+P3) كمسار واحد موحّد
- [x] كلمات بمسافات واضحة + تصحيح Groq للجملة كاملة + نطق بفواصل (P3)
- [x] تبويب Common Words الثابت محذوف؛ English Words الحي باقٍ؛ فلتر Words في قاموس EN (P4)
- [x] التدريب الإضافي للكلمات: لا نموذج يتفوق عددياً → checkpoint مُبقي (P4.1)
- [x] selftest_all.py = ALL PASS zero regression (P4.2)
- [x] كود ميت مُنقّى (P4.3)
- [x] وثائق جاهزة للعرض: REPORT.md + PROJECT_MAP.md (P4.4)

## بند معلّق (بانتظار قرار/كاميرا حقيقية، خارج نطاق هذه الجلسة)

- قال المستخدم: التوقف عند أول بند يحتاج قراره أو كاميرا حقيقية. لا توجد كاميرا متاحة الآن؛ فحص المعاينة الحية (EN-6) وضبط عتبات SIGNPAT_EN الحرف-بحرف مؤجّلان لحين بيئة كاميرا حقيقية.