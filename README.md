# مترجم لغة الإشارة العربية / العربية-الإنجليزية (AR→Sign, Sign→Text)

واجهة Streamlit للترجمة اللحظية من لغة الإشارة (الحروف العربية/الإنجليزية والكلمات الإنجليزية) إلى نص وصوت.

> ⚠️ **الكاميرا تعمل فقط عند التشغيل المحلي على جهازك** — رابط Streamlit Cloud لعرض الواجهة فقط بدون كاميرا فعلية (لا حاجة للتدريب بأي حال).

## التشغيل بدون تدريب (النماذج المدرّبة مرفوعة مع الريبو)

```bash
git clone https://github.com/3mrkhalid124/ar-sl-translator.git
cd ar-sl-translator

python -m venv .venv
.venv\Scripts\activate          # ويندوز
# source .venv/bin/activate     # لينكس / ماك

pip install -r requirements.txt
streamlit run app.py
```

يفتح التطبيق في المتصفح: يقوم تبويب «الترجمة اللحظية» بـ MediaPipe لتتبع اليد وكشف الحرف، وتبويب «القاموس» يعرض الحروف والمراجع — النماذج جاهزة (لا حاجة لتشغيل أي أمر تدريب).

### ملاحظات

- ملفات الاستدلال المرفوعة: `models/cnn.pt`، `models/cnn_en.pt`، `models/words_en.keras`، خرائط الأصناف، و`data/asl_signs/words_en_scaler.json`.
- عناصر تدريب/فحص اختيارية: `python engine.py --selftest` ، `python engine_words_en.py --eval`.