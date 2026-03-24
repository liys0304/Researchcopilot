import json
import os
from src.config import MEMORY_FILE


def _ensure_memory_file():
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def save_memory(item: str):
    _ensure_memory_file()
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.append(item)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def search_memory(query: str):
    _ensure_memory_file()
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    hits = []
    q = query.lower()
    for item in data:
        if q in item.lower():
            hits.append(item)
    return hits[:3]