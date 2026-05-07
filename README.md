# MS MARCO RAG Course Project

This project implements a Retrieval-Augmented Generation (RAG) passage search system for the CS 734 project. It uses a filtered MS MARCO passage subset, retrieves top-10 passages, optionally generates a grounded answer from the retrieved evidence, and evaluates retrieval quality with Precision@10.

## Final implementation summary

This version includes:

- MS MARCO judged-subset builder for Windows-friendly local development
- Dense retrieval with Sentence-BERT and FAISS
- BM25 lexical baseline retrieval
- Hybrid retrieval using reciprocal rank fusion over dense and BM25 results
- Lightweight deterministic query variants for weaker manual queries
- Top-10 ranked passage output
- Precision@10 evaluation
- Compact dense vs BM25 vs hybrid evaluation summary
- RAG answer generation with a switchable backend
  - default local backend: `openai/gpt-oss-20b`
  - optional OpenAI API backend
- Streamlit search interface
- Lightweight faceted filtering in the UI
- Clear subset and evaluation limitation notes

## Expected raw files

Place these files in `data/raw/`:

- `collection.tsv`
- `queries.dev.tsv`
- `queries.eval.tsv`
- `qrels.dev.tsv`

## Setup on Windows 11

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional `.env` file:

```env
OPENAI_API_KEY=your_key_here
LOCAL_MODEL_ID=openai/gpt-oss-20b
GENERATOR_BACKEND=local
OPENAI_MODEL=gpt-4.1-mini
LOCAL_DEVICE_MAP=auto
LOCAL_TORCH_DTYPE=auto
TOP_K=10
MAX_CONTEXT_PASSAGES=5
HYBRID_RRF_K=60
HYBRID_CANDIDATE_MULTIPLIER=5
HYBRID_MAX_QUERY_VARIANTS=4
```

## Build the judged subset

This project indexes a filtered judged subset first. That matches the practical course-project scope and avoids requiring a full MS MARCO index on a local machine.

```powershell
python app.py --build-subset
```

This creates:

- `data/subsets/collection.subset.tsv`
- `data/subsets/queries.dev.judged.tsv`
- `data/subsets/qrels.dev.judged.tsv`

## Build the dense index

```powershell
python app.py --build-index
```

This writes:

- `artifacts/faiss/msmarco_subset.faiss`
- `artifacts/faiss/msmarco_subset_metadata.parquet`
- `artifacts/embeddings/msmarco_subset_embeddings.npy`

## Run retrieval

Hybrid is the recommended default because it combines semantic matching from dense retrieval with exact-term matching from BM25.

```powershell
python app.py --load-index --query "what is phloem" --retriever hybrid
python app.py --load-index --query "what is phloem" --retriever dense
python app.py --load-index --query "what is phloem" --retriever bm25
python app.py --load-index --query "what is phloem" --retriever hybrid
python app.py --load-index --query "what is phloem" --retriever hybrid-rerank
```

## Run answer generation

```powershell
python app.py --load-index --query "what is phloem" --retriever hybrid --generate-answer --backend local
python app.py --load-index --query "what is phloem" --retriever hybrid --generate-answer --backend openai
```

The prompt instructs the generator to answer only from retrieved passages and to say when the evidence is insufficient.

## Run evaluation

```powershell
python app.py --load-index --evaluate
```

To also test the stronger but slower reranker:

```powershell
python app.py --load-index --evaluate --include-reranker
```

Evaluation uses:

- 5 random judged MS MARCO dev queries
- 5 manual project queries
- Dense, BM25, and hybrid retrievers
- Precision@10 for judged queries
- A compact comparison summary saved with the detailed CSV tables

Important limitation: the manual queries do not have official MS MARCO qrels, so the code marks them as `judged=False` and leaves their P@10 as blank/NaN. The summary row reports the mean only across judged queries. That avoids incorrectly treating manual queries as zero-relevance failures.

## Run the Streamlit UI

```powershell
streamlit run streamlit_app.py
```

The UI now exposes all final implementation options:

- `hybrid`, `dense`, and `bm25` retrieval
- local or OpenAI generation backend
- grounded answer toggle
- prompt preview
- passage facets
- subset limitation note
- optional dense vs BM25 vs hybrid evaluation summary

## Suggested final workflow

1. Put raw MS MARCO files in `data/raw/`
2. Run `python app.py --build-subset`
3. Run `python app.py --build-index`
4. Test retrieval with `python app.py --load-index --query "..." --retriever hybrid`
5. Test dense and BM25 baselines with `--retriever dense` and `--retriever bm25`
6. Test answers with `--generate-answer`
7. Run `python app.py --load-index --evaluate`
8. Launch the Streamlit UI

## How the remaining improvements were addressed

- Retrieval quality for weaker manual queries was improved by adding hybrid reciprocal-rank fusion and deterministic query variants.
- Subset limitations are now stated in the CLI, README, and Streamlit sidebar.
- The final presentation can now match the implementation exactly: dense, BM25, hybrid retrieval, Streamlit UI, grounded generation, facets, and P@10 evaluation are all implemented.
- Dense versus BM25 evaluation is cleaner because `comparison_summary_evaluation.csv` compares dense, BM25, and hybrid while excluding unjudged manual queries from the judged mean.


## Why P@10 looks small

Many MS MARCO passage qrels contain only one relevant passage for a query. With `top_k = 10`, that means the maximum possible P@10 for those queries is often only `0.10`. Because of that, the evaluator now also reports:

- `Hit@10`: whether at least one relevant passage appears in the top 10
- `Recall@10`: how many known relevant passages were retrieved
- `RR@10` / MRR@10: how highly the first relevant passage was ranked
- `NormP@10`: P@10 normalized by the maximum possible P@10 for that query

For example, a mean P@10 of `0.08` on five single-relevant-passage queries means the system found the relevant passage for four of the five judged queries, which is a mean Hit@10 / normalized P@10 of `0.80`.
