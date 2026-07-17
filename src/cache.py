import json
import os


def load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.decoder.JSONDecodeError:
                return {}
    return {}

def save_cache(cache_file, cache_data):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)