import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_FOLDER = "cache"

def build_law_vector_space(corpus_law_path: str, model: str):
    embedding_model = SentenceTransformer(model)

    print("[INFO][build_law_vector_space] check model cache available...")
    if os.path.exists(CACHE_FOLDER + "/law_index.faiss"):
        print("[INFO][build_law_vector_space] cache is OK.")
        index = faiss.read_index(CACHE_FOLDER + "/law_index.faiss")
        corpus_texts = np.load(CACHE_FOLDER + "/law_texts.npy", allow_pickle=True).tolist()
        metadata = np.load(CACHE_FOLDER + "/law_metadata.npy", allow_pickle=True).tolist()
        return embedding_model, index, corpus_texts, metadata

    print("[INFO][build_law_vector_space] loading corpus...")
    with open(corpus_law_path, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    if isinstance(law_data, dict):
        law_data = [law_data]

    corpus_texts = []
    metadata = []

    print("[INFO][build_law_vector_space] loading embeddings...")
    for law in law_data:
        law_id = law.get("law_id")

        print("[INFO][build_law_vector_space] processing law {}".format(law_id))

        for article in law.get("content", []):
            corpus_texts.append(article["content_Article"])
            metadata.append({"law_id": law_id, "aid": article["aid"]})

    print("[INFO][build_law_vector_space] serializing embeddings...")
    embeddings = embedding_model.encode(corpus_texts, convert_to_numpy=True)

    print("[INFO][build_law_vector_space] serializing index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("[INFO][build_law_vector_space] cache model...")
    faiss.write_index(index, CACHE_FOLDER + "/law_index.faiss")
    np.save(CACHE_FOLDER + "/law_texts.npy", corpus_texts)
    np.save(CACHE_FOLDER + "/law_metadata.npy", metadata)

    return embedding_model, index, corpus_texts, metadata

def retrieve_law(query: str, embedding_model, index, corpus_texts, metadata, top_k = 3):
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)

    evidences = []
    for idx in indices[0]:
        evidences.append({
            "law_id": metadata[idx]["law_id"],
            "aid": metadata[idx]["aid"],
            "text": corpus_texts[idx],
        })

    return evidences