import json
import time

from src.cache import load_cache
from src.datasource import fetch_case_evidence
from src.llm import predict
from src.rag import build_law_vector_space, retrieve_law

CORPUS_LAW_PATH = "dataset/corpus_law_pub.json"
MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
CACHE_FILE = "cache/cache.json"
INPUT_FILE = "dataset/ALQAC2026_public_test.json"
# INPUT_FILE = "dataset/local_private_test.json"
OUTPUT_FILE = "submission/submission"

if __name__ == "__main__":
    print("[INFO][main] loading embedding model...")
    embedding_model, faiss_index, corpus_texts, metadata = build_law_vector_space(CORPUS_LAW_PATH, EMBEDDING_MODEL)

    print("[INFO][main] loading test cases...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if isinstance(test_cases, dict):
        test_cases = [test_cases]

    print(f"[INFO][main] loaded {len(test_cases)} test cases.")

    submissions = []

    cache_data = load_cache(CACHE_FILE)

    print("[INFO][main] processing test cases...")

    file_name = f"{OUTPUT_FILE}-{time.strftime("%Y%m%d-%H%M%S")}.json"
    for case in test_cases:
        case_id = case.get("case_id")
        case_query = str(case.get("case_query"))

        print("[INFO][main] processing {}".format(case_id))
        case_evidence_ids, case_text = fetch_case_evidence(CACHE_FILE, cache_data, case)

        law_evidence = retrieve_law(case_query, embedding_model, faiss_index, corpus_texts, metadata, top_k=2)
        formatted_law_evidence = [{"law_id": item["law_id"], "aid": item["aid"]} for item in law_evidence]

        prediction = predict(MODEL, case_query, case_text, law_evidence)
        print("[INFO][main] prediction: {}".format(prediction))

        submissions.append({
            "case_id": case_id,
            "prediction": prediction,
            "case_evidence": case_evidence_ids,
            "law_evidence": formatted_law_evidence
        })

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=4)