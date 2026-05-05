"""
Build the tournament submission ZIP.

Usage (from repo root):
    python build_submission.py

Output: ../AI-Poker-Agent-submission.zip

Structure produced:
  submission/
    custom_player.py
    mcts_player.py
    submission_value_model.json
    mcts/           <- portal sees this as a local package
    heuristics/     <- portal sees this as a local package
  pypokerengine/    <- whitelisted by portal
  randomplayer.py
  raise_player.py
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT  = ROOT.parent / "AI-Poker-Agent-submission.zip"

SKIP = {"__pycache__", ".git", ".claude", ".DS_Store", ".mypy_cache", "__pycache__"}


def add_tree(zf: zipfile.ZipFile, src: Path, zip_prefix: str) -> int:
    count = 0
    for path in sorted(src.rglob("*")):
        if any(part in SKIP for part in path.parts):
            continue
        if path.is_file():
            arcname = zip_prefix + "/" + path.relative_to(src).as_posix()
            zf.write(path, arcname)
            count += 1
    return count


def build():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # submission/ — portal only whitelists packages found here
        n  = add_tree(zf, ROOT / "submission",   "submission")
        n += add_tree(zf, ROOT / "mcts",         "submission/mcts")
        n += add_tree(zf, ROOT / "heuristics",   "submission/heuristics")
        zf.write(ROOT / "mcts_player.py", "submission/mcts_player.py")
        n += 1

        # pypokerengine at zip root (portal whitelists it by name)
        n += add_tree(zf, ROOT / "pypokerengine", "pypokerengine")

        # Optional: baseline players (used in benchmarks, not the submission itself)
        for fname in ("randomplayer.py", "raise_player.py"):
            p = ROOT / fname
            if p.exists():
                zf.write(p, fname)
                n += 1

    # Verify
    with zipfile.ZipFile(OUT) as z:
        names = sorted(z.namelist())
    tops = sorted({n.split("/")[0] for n in names})
    sub_pkgs = sorted({
        "/".join(n.split("/")[:2])
        for n in names
        if n.startswith("submission/") and len(n.split("/")) > 2
    })
    print(f"Written {n} files to {OUT}")
    print(f"Top-level entries : {tops}")
    print(f"submission/ packages: {sub_pkgs}")

    missing = []
    for required in ["submission/custom_player.py",
                     "submission/mcts_player.py",
                     "submission/submission_value_model.json",
                     "submission/mcts/__init__.py",
                     "submission/heuristics/__init__.py"]:
        if required not in names:
            missing.append(required)
    if missing:
        print(f"\nWARNING: missing expected entries: {missing}")
    else:
        print("All required entries present.")


if __name__ == "__main__":
    build()
