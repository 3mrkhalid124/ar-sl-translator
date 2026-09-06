"""AppTest للتحقق من app.py — كل الأقسام بلا DuplicateElementId ولا استثناءات.
أنماط: تشغيل افتراضي، تبديل اللغة، النقر على الأزرار، فحص at.exception (والأخطاء الجديدة)."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

DUPE = "Duplicate"
ROOT = Path(__file__).resolve().parent
APP_PATH = str(ROOT / "app.py")


def snapshot(at, note):
    errs = list(at.exception)
    dupes = [str(e) for e in errs if DUPE in str(e)]
    others = [str(e) for e in errs if DUPE not in str(e)]
    return (len(dupes), len(others), dupes[:2], others[:2], note, len(at.button), len(at.radio))


def main():
    results = []
    # 1) إقلاع افتراضي (عربي)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    results.append(snapshot(at, "boot_ar_default"))

    # 2) تبديل للإنجليزي
    at.radio(key="lang_toggle").set_value("English").run()
    results.append(snapshot(at, "switch_en"))

    # 3) عودة للعربي
    at.radio(key="lang_toggle").set_value("عربي").run()
    results.append(snapshot(at, "switch_ar_back"))

    # 4) قاموس إنجليزي: التبديل بين Letters و Words
    at.radio(key="lang_toggle").set_value("English").run()
    try:
        at.radio(key="dict_mode").set_value("Words").run()
        results.append(snapshot(at, "dict_words"))
        at.radio(key="dict_mode").set_value("Letters").run()
        results.append(snapshot(at, "dict_letters"))
    except Exception as e:
        results.append((0, 1, [], [f"EXC dict_mode: {e}"], "dict_filter", 0, 0))

    # 4) إنجليزي + نقرة على أزرار الإدارة (Clear/Delete/Commit — ليست كاميرا)
    at.radio(key="lang_toggle").set_value("English").run()
    for key in ("wb_clr_English", "wb_back_English", "btn_ok_commit_English", "btn_new_word_English"):
        cur = AppTest.from_file(APP_PATH, default_timeout=60)
        cur.run()
        cur.radio(key="lang_toggle").set_value("English").run()
        try:
            cur.button(key=key).click().run()
            results.append(snapshot(cur, f"click_{key}"))
        except Exception as e:
            results.append((0, 1, [], [f"CLICK_EXC {key}: {e}"], f"click_{key}", 0, 0))

    # زر التثبيت + كلمة جديدة في الوضع العربي الأصلي (المفتاح يتغير باللغة)
    for key in ("btn_ok_commit_عربي", "btn_new_word_عربي"):
        cur = AppTest.from_file(APP_PATH, default_timeout=60)
        cur.run()
        try:
            cur.button(key=key).click().run()
            results.append(snapshot(cur, f"click_{key}"))
        except Exception as e:
            results.append((0, 1, [], [f"CLICK_EXC {key}: {e}"], f"click_{key}", 0, 0))

    # 5) النقر Start يفتح كاميرا — نتوقعه يظهر رسالة خطأ حائط بلا فرود — نتأكد أنه لا يفرط في العناصر
    at2 = AppTest.from_file(APP_PATH, default_timeout=60)
    at2.run()
    try:
        at2.button(key="btn_start_live").click().run()
        errs = list(at2.exception)
        results.append((len([e for e in errs if DUPE in str(e)]),
                        len([e for e in errs if DUPE not in str(e)]),
                        [str(e) for e in errs if DUPE in str(e)][:2],
                        [str(e) for e in errs if DUPE not in str(e)][:2],
                        "click_start_live", len(at2.button), len(at2.radio)))
    except Exception as e:
        results.append((0, 1, [], [f"EXC start_live: {e}"], "click_start_live", 0, 0))

    fail = [r for r in results if r[0] > 0 or r[1] > 0]
    for r in results:
        dupes, others = r[0], r[1]
        mark = "FAIL" if (dupes or others) else "ok"
        print(f"[AppTest] {mark:4} {r[4]:24} dupes={dupes} other_exc={others} buttons={r[5]} radio={r[6]}")
        for d in r[2] + r[3]:
            print("         -", d[:300].replace("\n", " "))
    print("[AppTest] RESULT:", "PASS (no exceptions/duplicates)" if not fail else f"FAIL ({len(fail)} cases)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())