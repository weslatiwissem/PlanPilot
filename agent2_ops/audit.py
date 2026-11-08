"""Audit log: one JSON line per write (who / what / when / agent / model version).

Proposal guarantee: "no silent writes". Every Trello write calls audit() first-hand,
so the file is a complete history of what Agent 2 changed.
Location: AUDIT_LOG_PATH env var (default agent2_ops/audit_log.jsonl, git-ignored).
"""
import json
import os
from datetime import datetime, timezone

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "audit_log.jsonl")


def audit(action: str, who: str, target: str, model_version: str, status: str = "ok", **extra):
    entry = {
        "when": datetime.now(timezone.utc).isoformat(),
        "agent": "agent2_ops",
        "who": who,                      # the human who approved the plan
        "action": action,                # create_board / create_list / create_card / ...
        "target": target,                # board name, list name or ticket key
        "model_version": model_version,  # skill-matcher version that made the assignment
        "status": status,
        **extra,
    }
    path = os.getenv("AUDIT_LOG_PATH", DEFAULT_PATH)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
