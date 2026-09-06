"""Comprehensive selftest across all four subsystems.
Returns exit 0 only if ALL pass; prints any FAIL clearly.

Systems:
  1. Arabic letters (engine.py --selftest)
  2. English letters (engine_en.py --selftest-en)
  3. English Words held-out test (engine_words_en.py --eval, threshold ≥ 0.55)
  4. Streamlit UI (apptest_sl.py — no DuplicateElementId / no exceptions)
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable
results: dict[str, tuple[bool, str]] = {}


def _run(label: str, cmd: list[str], expect: str):
    print(f"\n{'='*60}\n[TEST] {label}  cmd: {' '.join(cmd)}\n{'='*60}", flush=True)
    # engine.py selftest prints Arabic/Turkish on stdout and may emit TF logs to stderr;
    # decode both with UTF-8 (not the console codepage) to avoid 'charmap' decoding errors.
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, timeout=300)
        out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
        for line in out.splitlines():
            print("  ", line)
        ok = expect in out
        results[label] = (ok, out if not ok else "PASS")
    except Exception as e:
        results[label] = (False, f"EXCEPTION: {e}")
        print(f"  EXCEPTION: {e}")


def main():
    # 1+2: combined AR+EN engine selftest (engine.py --selftest covers both)
    _run("engine (AR+EN letters)", [PYTHON, "engine.py", "--selftest"], "overall: ALL PASS")

    # 3: English Words held-out test accuracy (≥ 0.55 threshold)
    _run("engine_words_en (--eval)", [PYTHON, "engine_words_en.py", "--eval"],
         "[SELFTEST] words_en.test: PASS")

    # 4: Streamlit AppTest
    _run("UI (apptest_sl.py)", [PYTHON, str(ROOT / "apptest_sl.py")],
         "[AppTest] RESULT: PASS")

    # summary
    print(f"\n{'='*60}")
    fails = {k: v for k, v in results.items() if not v[0]}
    for label, (ok, detail) in results.items():
        tag = "PASS" if ok else "FAIL"
        print(f"[SELFTEST-FINAL] {tag:4} {label}")
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for label, (_, detail) in fails.items():
            print(f"  {label}:")
            for line in detail.splitlines()[-10:]:
                print(f"    {line}")
        return 1
    print("\n[SELFTEST-FINAL] ALL PASS — zero regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())