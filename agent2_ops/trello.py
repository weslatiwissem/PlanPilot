"""Thin Trello REST client (plain `requests`, no extra SDK).

Handles: auth, rate limits / server errors (exponential backoff), finding-or-creating
boards and lists by name, and finding cards that were already created (idempotency).
Docs: https://developer.atlassian.com/cloud/trello/rest/
"""
import os
import re
import time
from typing import Dict, List, Optional

import requests

BASE = "https://api.trello.com/1"
KEY_RE = re.compile(r"planpilot-key: ([0-9a-f]{16})")  # marker we put in card descriptions


class TrelloClient:
    def __init__(self, api_key: str, token: str):
        self.auth = {"key": api_key, "token": token}

    @classmethod
    def from_env(cls) -> "TrelloClient":
        key, token = os.getenv("TRELLO_API_KEY"), os.getenv("TRELLO_API_TOKEN")
        if not key or not token:
            raise RuntimeError("TRELLO_API_KEY / TRELLO_API_TOKEN are not set (see .env.example).")
        return cls(key, token)

    # ------------------------------------------------------------------ low level
    def _call(self, method: str, path: str, retries: int = 5, **params):
        """HTTP call with exponential backoff on 429 (rate limit) and 5xx."""
        for attempt in range(retries + 1):
            resp = requests.request(
                method, f"{BASE}{path}", params={**self.auth, **params}, timeout=30
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries:
                    time.sleep(2**attempt)  # 1s, 2s, 4s, 8s, 16s
                    continue
            if not resp.ok:
                # never echo the URL (it contains the key/token)
                raise RuntimeError(f"Trello {method} {path} failed: {resp.status_code} {resp.text[:200]}")
            return resp.json()

    # ------------------------------------------------------------------ boards / lists
    def get_or_create_board(self, name: str) -> Dict:
        """Reuse TRELLO_BOARD_ID if set, else an open board with this name, else create one."""
        board_id = os.getenv("TRELLO_BOARD_ID")
        if board_id:
            return self._call("GET", f"/boards/{board_id}", fields="name,url")
        boards = self._call("GET", "/members/me/boards", fields="name,url", filter="open")
        for b in boards:
            if b["name"] == name:
                return b
        return self._call("POST", "/boards", name=name, defaultLists="false")

    def ensure_list(self, board_id: str, name: str) -> str:
        """Return the id of the open list called `name`, creating it if missing."""
        for lst in self._call("GET", f"/boards/{board_id}/lists", filter="open"):
            if lst["name"] == name:
                return lst["id"]
        return self._call("POST", "/lists", name=name, idBoard=board_id, pos="bottom")["id"]

    # ------------------------------------------------------------------ cards
    def existing_cards(self, board_id: str) -> Dict[str, Dict]:
        """{planpilot key: card} for every card on the board that we created before."""
        found = {}
        for card in self._call("GET", f"/boards/{board_id}/cards", fields="desc,shortUrl"):
            m = KEY_RE.search(card.get("desc", ""))
            if m:
                found[m.group(1)] = card
        return found

    def create_card(self, list_id: str, name: str, desc: str, member_ids: List[str]) -> Dict:
        return self._call(
            "POST", "/cards", idList=list_id, name=name, desc=desc, idMembers=",".join(member_ids)
        )

    # ------------------------------------------------------------------ members
    def member_id_on_board(self, username: str, board_id: str) -> Optional[str]:
        """Look up a Trello user and add them to the board. None if it fails."""
        try:
            member_id = self._call("GET", f"/members/{username}", fields="id")["id"]
            self._call("PUT", f"/boards/{board_id}/members/{member_id}", type="normal")
            return member_id
        except Exception:
            return None
