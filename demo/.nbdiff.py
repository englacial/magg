"""Summarize which cells changed vs HEAD for a notebook (crash-recovery triage)."""

import json
import subprocess
import sys

p = sys.argv[1]
head = json.loads(subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True).stdout)
cur = json.load(open(p))


def srcs(d):
    return ["".join(c["source"]) for c in d["cells"]]


a, b = srcs(head), srcs(cur)
print(f"{p}: HEAD {len(a)} cells -> now {len(b)} cells")
import difflib

sm = difflib.SequenceMatcher(None, a, b)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print(f"--- {tag}: HEAD[{i1}:{i2}] -> now[{j1}:{j2}]")
    for s in b[j1:j2][:3]:
        first = [ln for ln in s.splitlines() if ln.strip()][:3]
        print("    NEW: " + " | ".join(first)[:160])
    for s in a[i1:i2][:2]:
        first = [ln for ln in s.splitlines() if ln.strip()][:2]
        print("    OLD: " + " | ".join(first)[:160])
