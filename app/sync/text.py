import json
from typing import List, Optional


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def split_reported_persons(value: Optional[str]) -> List[str]:
    if not value:
        return []
    if not isinstance(value, str):
        value = str(value)
    normalized = value.strip()
    if not normalized:
        return []

    parsed_names = _extract_reported_person_names_from_json(normalized)
    return parsed_names or []


def _extract_reported_person_names_from_json(value: str) -> Optional[List[str]]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None

    seen = set()
    names: List[str] = []

    def append_name(raw_value: object) -> None:
        cleaned = clean_text(raw_value if isinstance(raw_value, str) else None)
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        names.append(cleaned)

    if isinstance(payload, dict):
        append_name(payload.get("mc"))
        return names

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                append_name(item.get("mc"))
            elif isinstance(item, str):
                append_name(item)
        return names

    return None
