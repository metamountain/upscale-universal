#!/usr/bin/env python
"""Standalone runner -- the portable ComfyUI python has no pytest.

    python tests/run_tests.py

Every `test_*` function in the modules below runs. A name listed in a
module's XFAIL set is expected to fail; if it starts passing the runner
says XPASS, which is the signal to delete the entry.
"""

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = ["test_sizing", "test_crop", "test_io", "test_exact_box", "test_preview"]


def main():
    passed = failed = xfailed = xpassed = 0
    failures = []

    for name in MODULES:
        mod = importlib.import_module(name)
        expect_fail = getattr(mod, "XFAIL", set())
        print(f"\n{name}")

        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue

            try:
                fn()
            except Exception:
                if attr in expect_fail:
                    xfailed += 1
                    print(f"  xfail  {attr}")
                else:
                    failed += 1
                    print(f"  FAIL   {attr}")
                    failures.append((name, attr, traceback.format_exc()))
            else:
                if attr in expect_fail:
                    xpassed += 1
                    print(f"  XPASS  {attr}  <- passing now, drop it from XFAIL")
                else:
                    passed += 1
                    print(f"  ok     {attr}")

    for name, attr, tb in failures:
        print(f"\n{'=' * 60}\n{name}.{attr}\n{'=' * 60}\n{tb}")

    bits = [f"{passed} passed"]
    if failed:
        bits.append(f"{failed} failed")
    if xfailed:
        bits.append(f"{xfailed} xfailed")
    if xpassed:
        bits.append(f"{xpassed} xpassed")
    print("\n" + ", ".join(bits))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
