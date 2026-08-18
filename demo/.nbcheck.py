"""Quick structural check of the demo notebooks (crash-recovery triage)."""

import glob
import json

for p in sorted(glob.glob("demo/*.ipynb")):
    try:
        d = json.load(open(p))
    except Exception as e:  # noqa: BLE001
        print(f"{p}: INVALID JSON: {e}")
        continue
    cells = d["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    unrun = [
        i
        for i, c in enumerate(code)
        if c.get("execution_count") is None and "".join(c["source"]).strip()
    ]
    errs = [
        i
        for i, c in enumerate(code)
        if any(o.get("output_type") == "error" for o in c.get("outputs", []))
    ]
    empty = [i for i, c in enumerate(cells) if not "".join(c["source"]).strip()]
    counts = [c.get("execution_count") for c in code if c.get("execution_count")]
    monotonic = counts == sorted(counts)
    print(
        f"{p}: {len(cells)} cells ({len(code)} code) | unrun={len(unrun)} "
        f"err={len(errs)} empty={len(empty)} exec_monotonic={monotonic}"
    )
    if errs:
        print(f"   ERROR at code idx {errs[:6]}")
    if unrun:
        print(f"   UNRUN at code idx {unrun[:8]}")
