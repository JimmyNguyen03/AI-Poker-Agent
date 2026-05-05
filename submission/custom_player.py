"""
Course submission entry: the portal expects submission/custom_player.py in the zip.

Zip layout:
  submission/
    custom_player.py          <-- this file
    submission_value_model.json
    mcts_player.py
    mcts/
    heuristics/
  pypokerengine/
  randomplayer.py
  raise_player.py

All custom packages live inside submission/ so the portal's import checker
recognises them as local helper modules. We add submission/ to sys.path so
existing import statements (e.g. "from mcts.search import ...") keep working.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SUBMISSION = Path(__file__).resolve().parent
# submission/ contains mcts/, heuristics/, mcts_player.py
if str(_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION))
# Also add the zip root so pypokerengine is importable
_ROOT = _SUBMISSION.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcts_player import MCTSPlayer, setup_ai as setup_ai_from_project


def setup_ai():
    """Delegate to main agent factory (loads submission_value_model.json when present)."""
    return setup_ai_from_project()


class CustomPlayer(MCTSPlayer):
    """Some harnesses construct CustomPlayer() with no arguments."""

    def __init__(self, **kwargs):
        if not kwargs:
            # Model lives next to this file in submission/
            submission = Path(__file__).resolve().parent / "submission_value_model.json"
            if not submission.is_file():
                # Fallback: legacy path from repo root
                submission = _ROOT / "offline_learning" / "models" / "submission_value_model.json"
            if submission.is_file():
                kwargs = {"value_model_path": str(submission)}
        super().__init__(**kwargs)


__all__ = ["setup_ai", "CustomPlayer", "MCTSPlayer"]
