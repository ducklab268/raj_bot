import os
import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# CONFIG
MODEL_NAME_EMBED = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt2"
MAX_CONTEXT_TOKENS = 500  

BASE_DIR = Path(__file__).parent
BLOG_DIR = BASE_DIR / "blogs"
INDEX_DIR = BASE_DIR / "faiss_index"
INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "meta.json"

_embed_model = None
_faiss_index = None
_meta = None
_llm_pipeline = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(MODEL_NAME_EMBED)
    return _embed_model

def _ensure_index_dir():
    if INDEX_DIR.exists() and INDEX_DIR.is_file():
        INDEX_DIR.unlink()
    INDEX_DIR.mkdir(exist_ok=True)

def build_index(chunk_size_words=200, overwrite=False):
    global _faiss_index, _meta
    _ensure_index_dir()
    if INDEX_FILE.exists() and META_FILE.exists() and not overwrite:
        return load_index()

    docs = []
    for p in sorted(BLOG_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        words = raw.split()
        for i in range(0, len(words), chunk_size_words):
            chunk = " ".join(words[i:i+chunk_size_words])
            docs.append({"title": p.stem, "text": chunk})

    if not docs:
        raise ValueError("❌ No embeddings generated. Check blogs/ content!")

    model = _get_embed_model()
    texts = [d["title"] + "\n" + d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"docs": docs}, f, ensure_ascii=False, indent=2)

    _faiss_index = index
    _meta = {"docs": docs}
    return index, _meta

def load_index():
    global _faiss_index, _meta
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return build_index()
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _faiss_index = index
    _meta = meta
    return index, meta

def _get_llm_pipeline():
    global _llm_pipeline
    if _llm_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
        _llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
        )
    return _llm_pipeline

def _retrieve(query: str, k=4):
    index, meta = load_index()
    model = _get_embed_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = [meta["docs"][i] for i in I[0] if 0 <= i < len(meta["docs"])]
    return results

def _build_prompt(context_docs: List[dict], question: str):
    context = "\n\n".join([f"Title: {d['title']}\n{d['text']}" for d in context_docs])
    prompt = f"You are a helpful assistant. Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    if len(prompt.split()) > MAX_CONTEXT_TOKENS*4:
        words = prompt.split()
        prompt = " ".join(words[-(MAX_CONTEXT_TOKENS*4):])
    return prompt

def get_answer(query: str, k=4):
    load_index()
    docs = _retrieve(query, k=k)
    prompt = _build_prompt(docs, query)
    generator = _get_llm_pipeline()
    out = generator(prompt, max_length=200, num_return_sequences=1)
    text = out[0]["generated_text"]
    if prompt in text:
        return text.split(prompt, 1)[1].strip()
    return text.strip()

if __name__ == "__main__":
    print("Building index...")
    build_index()
    print("Index built.")
