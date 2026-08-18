"""Dump the failing cell + traceback from a notebook (crash-recovery triage)."""

import json
import sys

p = sys.argv[1]
d = json.load(open(p))
code = [c for c in d["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"=== code cell {i} SOURCE ===")
            print("".join(c["source"])[:1500])
            print(f"=== {o.get('ename')}: {o.get('evalue')} ===")
            tb = "\n".join(o.get("traceback", []))
            import re

            print(re.sub(r"\x1b\[[0-9;]*m", "", tb)[-1800:])
