# Hierarchical RAG (Retrieval-Augmented Generation)

Tree‑routed document search and answer generation using topic hierarchies, dense embeddings, and lightweight routing. This repo demonstrates how to organize a corpus into a **topic tree**, route queries to the most relevant subtrees, retrieve focused chunks, and generate concise answers with grounded citations.

> **Status:** Initial public version. Adapt the paths/commands to match your repository layout.

---

## ✨ Key Features

- **Hierarchical indexing:** Builds a topic tree (root → categories → subcategories → leaf nodes) for scalable retrieval on large corpora.
- **Routing-first retrieval:** Routes queries to the best subtree using centroid similarity and optional classifier/ranker.
- **Chunking & embeddings:** Sentence/paragraph chunking + transformer embeddings for efficient similarity search.
- **Hybrid retrieval (optional):** Dense + keyword scoring for robustness on short queries.
- **Context assembly:** Merges top chunks with deduping, windowing, and token budgeting for the generator.
- **Reranking:** (Optional) Cross-encoder or MMR reranking to boost precision@k.
- **Pluggable generator:** Works with HF pipelines or API-based LLMs.
- **Reproducible config:** YAML-driven pipelines for data, model, and tree parameters.

---

## 🏗️ Architecture (High Level)

```
Raw docs → clean/split → embed → cluster → build topic tree
                                       │
                                       ▼
                                 [Index Artifacts]
                                       │
         ┌───────── Query ─────────────┴────────────┐
         ▼                                           ▼
   route to node(s)                          retrieve top-k chunks
         │                                           │
         └───────────── assemble context ────────────┘
                               │
                               ▼
                         generate answer
                         + cite sources
```


---

## ⚙️ Setup

### 1) Environment
- Python 3.10+ recommended
- Create env and install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install typical deps:

```bash
pip install sentence-transformers scikit-learn numpy faiss-cpu \
            pandas pyyaml tqdm nltk transformers
```


## 🚀 Build the Index

1) **Preprocess (clean + chunk):**
```bash
python -m src.preprocess --config config/default.yaml
```

2) **Embed chunks + build vector index:**
```bash
python -m src.embed --config config/default.yaml
```

3) **Build the topic tree (clustering per level):**
```bash
python -m src.build_tree --config config/default.yaml
```

> Artifacts will be written to `artifacts/embeddings`, `artifacts/clusters`, and `artifacts/tree`.

---

## 🔎 Query Flow

**Single query end-to-end:**
```bash
python -m src.generate \
  --config config/default.yaml \
  --query "What causes false positives in CTPA-based PE detection?" \
  --top_k 12
```

**Under the hood:**
1. **Route:** compute similarity to node centroids (or classifier) → pick top nodes.
2. **Retrieve:** get top-k chunks from indexes of those nodes (dense/hybrid).
3. **Rerank (optional):** cross-encode or MMR for diversity.
4. **Assemble:** pack context with dedupe/windowing under token budget.
5. **Generate:** produce answer + cite source chunk IDs/paths.

---

## 🧪 Evaluation

Compute retrieval metrics on a labeled set (query → relevant doc IDs):

```bash
python -m src.eval \
  --config config/default.yaml \
  --qrels data/qrels.tsv \
  --queries data/queries.tsv
```

Outputs: Recall@k, Precision@k, MRR, nDCG; compare **flat vs. hierarchical** routing.

---

## 🧩 Implementation Notes

- **Pruning:** during tree build, prune tiny/low‑quality clusters and push their chunks to siblings or parent.
- **Cold‑start:** keep a small global index to catch broad queries that don’t route well.
- **Caching:** memoize embeddings and per‑node retrieval results for interactive speed.
- **Safety:** always return citations and limit generation to the retrieved context.

---

## 🤝 Contributing

PRs and issues are welcome! Please include:
- brief problem statement,
- minimal repro or dataset snippet,
- metrics before/after (if relevant).

---



Inspired by work on hierarchical retrieval, sentence-transformer embeddings, and practical RAG routing patterns.
