import time

import requests

from src.cache import save_cache

API_URL = "https://alqac-api.ngrok.pro/retrieve"
API_TOKEN = "alqac_pbyJMHOCFXIg1nuF7fwDrbRHN3Nv_Uhu"

def fetch_case_evidence(cache_file: str, cache_data: dict, case: dict) -> tuple[list, str]:
    case_id = case.get("case_id", "")
    case_query = case.get("case_query", "")

    cache_key = str(case_id)

    if cache_key not in cache_data or not isinstance(cache_data[cache_key], list):
        cache_data[cache_key] = []

    if "case_fact" in case and case["case_fact"]:
        return [], case["case_fact"]

    headers = {
        "X-API-Key": API_TOKEN,
        "Content-Type": "application/json",
    }
    payload = {"query": case_query, "case_id": case_id}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        if response.status_code == 429:
            time.sleep(5)
            return fetch_case_evidence(cache_file, cache_data, case)

        response.raise_for_status()
        results = response.json().get("results", [])
        time.sleep(5.1)

        if results:
            top_result = results[0]
            chunk_id = top_result.get("chunk_id")

            existing_chunk_ids = [item.get("chunk_id") for item in cache_data[cache_key]]
            if chunk_id not in existing_chunk_ids:
                cache_data[cache_key].append(top_result)
                save_cache(cache_file, cache_data)
    except Exception as e:
        print(f"[!] API error: {e}")

    cached_list = cache_data.get(cache_key, [])

    if cached_list:
        chunk_ids = [item.get("chunk_id") for item in cached_list]

        combined_text = "\n...AND...\n".join([item.get("text", "") for item in cached_list])
        return chunk_ids, combined_text

    return [], ""




