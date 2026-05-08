"""
Build the tournament submission ZIP.

Usage (from repo root):
    python build_submission.py

Output: ../AI-Poker-Agent-submission.zip

Structure produced:
  submission/
    custom_player.py
    mcts_player.py
    submission_value_model.json   <- always taken from BEST_MODEL below
    mcts/           <- portal sees this as a local package
    heuristics/     <- portal sees this as a local package
  pypokerengine/    <- whitelisted by portal
  randomplayer.py
  raise_player.py
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT  = ROOT.parent / "AI-Poker-Agent-submission.zip"

# Best trained model — update this path if a better checkpoint is produced.
BEST_MODEL = (
    ROOT / "offline_learning" / "models" / "transformer" / "rollout"
    / "submission_value_model.json"
)

SKIP = {"__pycache__", ".git", ".DS_Store", ".mypy_cache"}

# Files inside submission/ that are managed explicitly (not copied from the dir).
_SUBMISSION_MANAGED = {"submission_value_model.json"}


def add_tree(zf: zipfile.ZipFile, src: Path, zip_prefix: str,
             skip_names: set[str] | None = None) -> int:
    count = 0
    for path in sorted(src.rglob("*")):
        if any(part in SKIP for part in path.parts):
            continue
        if path.is_file():
            if skip_names and path.name in skip_names:
                continue
            arcname = zip_prefix + "/" + path.relative_to(src).as_posix()
            zf.write(path, arcname)
            count += 1
    return count


def verify_model(path: Path) -> str:
    """Load the model and return its type string. Raises on failure."""
    sys.path.insert(0, str(ROOT))
    from mcts.value_model import load_value_model
    m = load_value_model(str(path))
    with open(path) as f:
        raw = json.load(f)
    return raw.get("model_type", type(m).__name__)


def build():
    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    if not BEST_MODEL.exists():
        print(f"ERROR: best model not found at:\n  {BEST_MODEL}")
        print("Run training first:  python offline_learning/train.py --model transformer --target rollout ...")
        sys.exit(1)

    print(f"Best model : {BEST_MODEL.relative_to(ROOT)}")
    model_type = verify_model(BEST_MODEL)
    print(f"Model type : {model_type}")
    print(f"Output     : {OUT}\n")

    # ------------------------------------------------------------------
    # Build zip
    # ------------------------------------------------------------------
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:

        # submission/ — skip any stale submission_value_model.json living there
        n  = add_tree(zf, ROOT / "submission", "submission",
                      skip_names=_SUBMISSION_MANAGED)

        # Best model goes in explicitly so we always use the right checkpoint
        zf.write(BEST_MODEL, "submission/submission_value_model.json")
        n += 1

        # MCTS engine and heuristics live under submission/ so the portal
        # recognises them as local packages relative to custom_player.py
        n += add_tree(zf, ROOT / "mcts",       "submission/mcts")
        n += add_tree(zf, ROOT / "heuristics", "submission/heuristics")

        # mcts_player.py at submission root (imported by custom_player.py)
        zf.write(ROOT / "mcts_player.py", "submission/mcts_player.py")
        n += 1

        # pypokerengine at zip root (portal whitelists it by name)
        n += add_tree(zf, ROOT / "pypokerengine", "pypokerengine")

        # Baseline players at zip root (used in arena, not the submission itself)
        for fname in ("randomplayer.py", "raise_player.py"):
            p = ROOT / fname
            if p.exists():
                zf.write(p, fname)
                n += 1

    # ------------------------------------------------------------------
    # Verify zip contents
    # ------------------------------------------------------------------
    with zipfile.ZipFile(OUT) as z:
        names = set(z.namelist())

    required = [
        "submission/custom_player.py",
        "submission/mcts_player.py",
        "submission/submission_value_model.json",
        "submission/mcts/__init__.py",
        "submission/heuristics/__init__.py",
        "pypokerengine/__init__.py",
    ]
    missing = [r for r in required if r not in names]

    tops    = sorted({n.split("/")[0] for n in names})
    sub_pkgs = sorted({
        "/".join(n.split("/")[:2])
        for n in names
        if n.startswith("submission/") and len(n.split("/")) > 2
    })

    print(f"Files written       : {n}")
    print(f"Top-level entries   : {tops}")
    print(f"submission/ packages: {sub_pkgs}")

    if missing:
        print(f"\nERROR: missing required entries:\n  " + "\n  ".join(missing))
        sys.exit(1)

    # Confirm the bundled model is the one we intended
    with zipfile.ZipFile(OUT) as z:
        bundled = json.loads(z.read("submission/submission_value_model.json"))
    with open(BEST_MODEL) as f:
        source  = json.load(f)
    if bundled != source:
        print("\nERROR: bundled model does not match BEST_MODEL — zip may be corrupt.")
        sys.exit(1)

    print(f"\nAll checks passed.")
    print(f"Submission ready: {OUT}")


if __name__ == "__main__":
    build()
