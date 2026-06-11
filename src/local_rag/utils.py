"""Shared utilities with no heavy dependencies.

Kept in their own module so they can be imported from tests and from other
parts of the package without pulling in the model clients or config.
"""

from __future__ import annotations

import json


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply; tolerate stray prose.

    Many small local models wrap their JSON in extra prose or markdown.
    This finder grabs everything between the first '{' and last '}' and
    tries to parse it, returning {} on any failure.
    """
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}
