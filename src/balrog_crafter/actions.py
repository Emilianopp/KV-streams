from __future__ import annotations

import re
from typing import Optional

CRAFTER_ACTIONS: tuple[str, ...] = (
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
)

_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_ACTION_LINE_RE = re.compile(r"^\s*(?:action|move)\s*[:=-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_ALIASES: dict[str, str] = {}
for _action in CRAFTER_ACTIONS:
    _ALIASES[_key(_action)] = _action
    _ALIASES[_key(_action.replace("Move ", ""))] = _action

_ALIASES.update(
    {
        "": "Noop",
        "wait": "Noop",
        "stay": "Noop",
        "none": "Noop",
        "no op": "Noop",
        "noop": "Noop",
        "west": "Move West",
        "left": "Move West",
        "a": "Move West",
        "east": "Move East",
        "right": "Move East",
        "d": "Move East",
        "north": "Move North",
        "up": "Move North",
        "w": "Move North",
        "south": "Move South",
        "down": "Move South",
        "s": "Move South",
        "interact": "Do",
        "use": "Do",
        "collect": "Do",
        "mine": "Do",
        "chop": "Do",
        "attack": "Do",
        "pickup": "Do",
        "pick up": "Do",
        "place stone": "Place Stone",
        "place table": "Place Table",
        "place crafting table": "Place Table",
        "crafting table": "Place Table",
        "place furnace": "Place Furnace",
        "place plant": "Place Plant",
        "plant": "Place Plant",
        "make wood pickaxe": "Make Wood Pickaxe",
        "craft wood pickaxe": "Make Wood Pickaxe",
        "wood pickaxe": "Make Wood Pickaxe",
        "make wooden pickaxe": "Make Wood Pickaxe",
        "craft wooden pickaxe": "Make Wood Pickaxe",
        "wooden pickaxe": "Make Wood Pickaxe",
        "make stone pickaxe": "Make Stone Pickaxe",
        "craft stone pickaxe": "Make Stone Pickaxe",
        "stone pickaxe": "Make Stone Pickaxe",
        "make iron pickaxe": "Make Iron Pickaxe",
        "craft iron pickaxe": "Make Iron Pickaxe",
        "iron pickaxe": "Make Iron Pickaxe",
        "make wood sword": "Make Wood Sword",
        "craft wood sword": "Make Wood Sword",
        "wood sword": "Make Wood Sword",
        "make wooden sword": "Make Wood Sword",
        "craft wooden sword": "Make Wood Sword",
        "wooden sword": "Make Wood Sword",
        "make stone sword": "Make Stone Sword",
        "craft stone sword": "Make Stone Sword",
        "stone sword": "Make Stone Sword",
        "make iron sword": "Make Iron Sword",
        "craft iron sword": "Make Iron Sword",
        "iron sword": "Make Iron Sword",
    }
)


def extract_action(text: object) -> Optional[str]:
    """Extract and normalize a Crafter action from model output."""

    if not isinstance(text, str):
        return None

    candidates: list[str] = []

    tagged = _ACTION_TAG_RE.search(text)
    if tagged:
        candidates.append(tagged.group(1))

    action_line = _ACTION_LINE_RE.search(text)
    if action_line:
        candidates.append(action_line.group(1))

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
        candidates.extend(line.strip() for line in stripped.splitlines() if line.strip())

    for candidate in candidates:
        cleaned = candidate.strip().strip("`'\".!,;:()[]{}")
        key = _key(cleaned)
        if key in _ALIASES:
            return _ALIASES[key]

    haystack = f" {_key(text)} "
    for key, action in sorted(_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if key and f" {key} " in haystack:
            return action

    return None
