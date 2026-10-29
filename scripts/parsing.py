"""Evidence text parsed from an extractor reply."""

import json
import re


def clean_model_text(text):
    """Drop code fences and thinking tags so the JSON object can be found."""
    cleaned = "" if text is None else str(text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def balanced_objects(text):
    """Yield top-level JSON objects, ignoring braces that sit inside strings."""
    length = len(text)
    index = 0
    while index < length:
        if text[index] != "{":
            index += 1
            continue
        found, next_index = scan_object(text, index)
        if found is None:
            index += 1
            continue
        yield found
        index = next_index


def scan_object(text, start):
    """Return the object that opens at start, and the index just after it.

    A backslash inside a string must not treat the next quote as the end of
    that string. Returns (None, start) when the object is unfinished.
    """
    depth = 0
    in_string = False
    escaped = False
    for cursor in range(start, len(text)):
        char = text[cursor]
        if in_string:
            escaped, in_string = advance_string(char, escaped)
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : cursor + 1], cursor + 1
    return None, start


def advance_string(char, escaped):
    """Step one character inside a JSON string. Returns (escaped, still_in_string)."""
    if escaped:
        return False, True
    if char == "\\":
        return True, True
    if char == '"':
        return False, False
    return False, True


def yes_no(value):
    """Map a discusses_aspect field to Yes or No, or None when it is unusable."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("yes", "y", "true", "1"):
        return "Yes"
    if text in ("no", "n", "false", "0"):
        return "No"
    return None


def narrative_list(value):
    """Coerce narrative_1 into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def evidence_text(raw):
    """Join extracted sentences. Returns an empty string when the aspect is absent."""
    cleaned = clean_model_text(raw)
    for blob in reversed(list(balanced_objects(cleaned))):
        parsed = parse_evidence_object(blob)
        if parsed is None:
            continue
        return parsed
    return ""


def parse_evidence_object(blob):
    """Return joined narrative text, an empty string for No, or None if unusable."""
    try:
        data = json.loads(blob)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    decision = yes_no(data.get("discusses_aspect"))
    if decision is None:
        return None
    if decision == "No":
        return ""
    return ", ".join(narrative_list(data.get("narrative_1")))
