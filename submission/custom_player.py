"""
Course submission entry: the portal expects submission/custom_player.py in the zip.

Zip layout (example):
  submission/custom_player.py   <-- this file
  mcts_player.py
  offline_learning/...
  mcts/...
  heuristics/...
  pypokerengine/...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root is parent of submission/
_ROOT = Path(__file__).resolve().parent.parent
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
            submission = _ROOT / "offline_learning" / "models" / "submission_value_model.json"
            if submission.is_file():
                kwargs = {"value_model_path": str(submission.resolve())}
        super().__init__(**kwargs)


__all__ = ["setup_ai", "CustomPlayer", "MCTSPlayer"]
