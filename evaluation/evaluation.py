"""Evaluate Dense, BM25, and Hybrid retrieval for RestoRec.

Workflow:
1. Set LABELLING_MODE = True and run the file.
2. Inspect the candidates printed for each question.
3. Add every genuinely relevant document ID to relevant_ids.
4. Set LABELLING_MODE = False and run the file again.
"""

import csv
import math
import sys
import time
from pathlib import Path
from statistics import mean

# Ensure Windows terminals can print emojis and other Unicode characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from src.rag import (
    EMBEDDING_MODEL,
    SOURCE_LIMITS,
    TOTAL_RESULTS,
    get_embedding_model,
    load_all_documents,
)


# Configuration

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_DIR / "retrieval_evaluation_results.csv"
# True: print ground-truth candidates only.
# False: build indexes and run the final evaluation.
LABELLING_MODE = True

# SOURCE_LIMITS, TOTAL_RESULTS and EMBEDDING_MODEL are imported from
# src.rag so the evaluation always uses the live application's settings.

# Retrieve a larger pool before Reciprocal Rank Fusion.
CANDIDATES_PER_SOURCE = 10
RRF_CONSTANT = 60
HYBRID_WEIGHTS = {"bm25": 0.3, "dense": 0.7}
TIMING_REPETITIONS = 3


# Evaluation questions


EVALUATION_QUESTIONS = [
    # Cuisine (5)
    {
        "question": "Where can I find Indian food in Croydon?",
        "category": "Cuisine",
        "search_terms": ["indian", "curry", "tandoori", "biryani", "masala"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find Turkish food in Croydon?",
        "category": "Cuisine",
        "search_terms": ["turkish", "kebab", "meze", "ocakbasi"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find Italian food in Croydon?",
        "category": "Cuisine",
        "search_terms": ["italian", "pizza", "pasta", "risotto"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find Spanish food in Croydon?",
        "category": "Cuisine",
        "search_terms": ["spanish", "tapas", "paella"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find Brazilian food in Croydon?",
        "category": "Cuisine",
        "search_terms": ["brazilian", "brazil", "churrasco", "rodizio"],
        "relevant_ids": set(),
    },

    # Dish or restaurant type (4)
    {
        "question": "Where can I get burgers in Croydon?",
        "category": "Dish",
        "search_terms": ["burger", "hamburger", "cheeseburger"],
        "relevant_ids": {
            "TikTok:822", "TikTok:574", "Places:107", "Places:299",
            "Places:309", "Places:509", "Places:717",
        },
    },
    {
        "question": "Where can I get pizza in Croydon?",
        "category": "Dish",
        "search_terms": ["pizza", "pizzeria"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I get sushi in Croydon?",
        "category": "Dish",
        "search_terms": ["sushi", "maki", "sashimi", "nigiri"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find a good steak in Croydon?",
        "category": "Dish",
        "search_terms": ["steak", "steakhouse", "grill"],
        "relevant_ids": set(),
    },

    # Location (4)
    {
        "question": "Which restaurants are near East Croydon station?",
        "category": "Location",
        "search_terms": ["east croydon", "east croydon station"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I eat in South Croydon?",
        "category": "Location",
        "search_terms": ["south croydon"],
        "relevant_ids": set(),
    },
    {
        "question": "Which restaurants are in Thornton Heath?",
        "category": "Location",
        "search_terms": ["thornton heath"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I eat in Purley?",
        "category": "Location",
        "search_terms": ["purley"],
        "relevant_ids": set(),
    },

    # Dietary requirement (3)
    {
        "question": "Which restaurants serve halal food?",
        "category": "Dietary",
        "search_terms": ["halal"],
        "relevant_ids": {"Places:47"},
    },
    {
        "question": "Where can I find vegan food in Croydon?",
        "category": "Dietary",
        "search_terms": ["vegan", "plant based", "plant-based"],
        "relevant_ids": {"Places:558", "Places:107"},
    },
    {
        "question": "Which restaurants have gluten-free options?",
        "category": "Dietary",
        "search_terms": ["gluten free", "gluten-free", "coeliac", "celiac"],
        "relevant_ids": set(),
    },

    # Price or atmosphere (2)
    {
        "question": "Where can I find affordable restaurants in Croydon?",
        "category": "Price/Atmosphere",
        "search_terms": ["affordable", "cheap", "budget", "good value", "inexpensive"],
        "relevant_ids": set(),
    },
    {
        "question": "Which restaurants are good for a date night in Croydon?",
        "category": "Price/Atmosphere",
        "search_terms": ["date night", "romantic", "cosy", "cozy", "intimate"],
        "relevant_ids": set(),
    },

    # Combined constraints (2)
    {
        "question": "Where can I find halal burgers in Croydon?",
        "category": "Combined",
        "search_terms": ["halal burger", "halal", "burger"],
        "relevant_ids": set(),
    },
    {
        "question": "Where can I find vegan food near East Croydon station?",
        "category": "Combined",
        "search_terms": ["vegan", "plant based", "east croydon"],
        "relevant_ids": set(),
    },
]



# Document helpers


def normalise_source(source):
    """Return the consistent source name used by the project."""
    source_text = str(source).strip().lower()
    if "tiktok" in source_text:
        return "TikTok"
    if "reddit" in source_text:
        return "Reddit"
    if "place" in source_text:
        return "Places"
    return str(source).strip() or "Unknown"


def get_document_id(document):
    """Create a stable Source:row_number ID from document metadata."""
    source = normalise_source(document.metadata.get("source", "Unknown"))
    row_number = document.metadata.get(
        "row_number", document.metadata.get("row", "unknown")
    )
    return f"{source}:{row_number}"


def deduplicate_documents(documents):
    """Remove duplicate documents while preserving ranking."""
    unique_documents = []
    seen = set()
    for document in documents:
        document_id = get_document_id(document)
        if document_id not in seen:
            unique_documents.append(document)
            seen.add(document_id)
    return unique_documents


def preview_text(document, maximum_length=250):
    """Produce a short single-line preview."""
    text = " ".join(document.page_content.split())
    return text[:maximum_length] + "..." if len(text) > maximum_length else text


# Load documents and support ground-truth labelling


def load_evaluation_documents():
    """Load documents with the same functions used by the RAG application."""
    print("Loading documents using the current rag.py loaders...")
    documents_by_source, combined_documents = load_all_documents()

    for source, documents in documents_by_source.items():
        print(f"{source}: {len(documents)} documents")
        for document in documents:
            document.metadata["source"] = source

    total = len(combined_documents)
    if total == 0:
        raise ValueError("No documents were loaded. Check data files and rag.py loaders.")
    print(f"Total: {total} documents")
    return documents_by_source


def search_documents_for_labelling(documents_by_source, search_terms):
    """Return documents containing at least one case-insensitive search term."""
    terms = [term.lower().strip() for term in search_terms if term.strip()]
    matches = []

    for documents in documents_by_source.values():
        for document in documents:
            text = document.page_content.lower()
            if any(term in text for term in terms):
                matches.append(document)

    matches = deduplicate_documents(matches)
    for document in matches:
        print(f"\nID: {get_document_id(document)}")
        print(preview_text(document, maximum_length=500))
    print(f"\nTotal matching documents: {len(matches)}")
    return matches


def run_labelling_searches(documents_by_source):
    """Print candidate documents for every evaluation question."""
    for number, test_case in enumerate(EVALUATION_QUESTIONS, start=1):
        print("\n" + "=" * 90)
        print(f"QUESTION {number}: {test_case['question']}")
        print(f"CATEGORY: {test_case['category']}")
        print(f"SEARCH TERMS: {', '.join(test_case['search_terms'])}")
        print("=" * 90)
        search_documents_for_labelling(
            documents_by_source, test_case["search_terms"]
        )



# Build indexes


def build_retrieval_indexes(documents_by_source):
    """Build the app-style combined FAISS index and per-source BM25 indexes."""
    print("\nLoading MiniLM embedding model...")
    embeddings = get_embedding_model()
    combined_documents = [
        document
        for documents in documents_by_source.values()
        for document in documents
    ]

    print("Building the combined FAISS index used by the application...")
    dense_index = FAISS.from_documents(combined_documents, embeddings)
    bm25_retrievers = {}

    for source, documents in documents_by_source.items():
        if not documents:
            continue
        print(f"Building the BM25 index for {source}...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = min(CANDIDATES_PER_SOURCE, len(documents))
        bm25_retrievers[source] = bm25_retriever

    return dense_index, bm25_retrievers


# Retrieval methods

def retrieve_dense(query, dense_index):
    """Mirror the source-balanced dense retrieval in the current rag.py."""
    selected = []
    seen = set()

    for source, result_limit in SOURCE_LIMITS.items():
        try:
            documents = dense_index.similarity_search(
                query,
                k=result_limit,
                filter={"source": source},
                fetch_k=200,
            )
        except TypeError:
            documents = dense_index.similarity_search(
                query,
                k=result_limit,
                filter={"source": source},
            )

        for document in documents:
            document_id = get_document_id(document)
            if document_id not in seen:
                selected.append(document)
                seen.add(document_id)

    # Match rag.py: fills unused positions if a source has fewer than its quota.
    if len(selected) < TOTAL_RESULTS:
        for document in dense_index.similarity_search(
            query,
            k=TOTAL_RESULTS * 4,
        ):
            document_id = get_document_id(document)
            if document_id not in seen:
                selected.append(document)
                seen.add(document_id)
            if len(selected) == TOTAL_RESULTS:
                break

    return selected[:TOTAL_RESULTS]


def retrieve_bm25(query, bm25_retrievers):
    """Retrieve source-balanced results with BM25."""
    final_documents = []
    for source, result_limit in SOURCE_LIMITS.items():
        retriever = bm25_retrievers.get(source)
        if retriever is not None:
            final_documents.extend(retriever.invoke(query)[:result_limit])
    return deduplicate_documents(final_documents)[:TOTAL_RESULTS]


def reciprocal_rank_fusion(bm25_documents, dense_documents):
    """Combine BM25 and dense rankings using weighted RRF."""
    scores = {}
    document_lookup = {}
    ranked_lists = [
        (bm25_documents, HYBRID_WEIGHTS["bm25"]),
        (dense_documents, HYBRID_WEIGHTS["dense"]),
    ]

    for documents, weight in ranked_lists:
        for rank, document in enumerate(documents, start=1):
            document_id = get_document_id(document)
            document_lookup[document_id] = document
            scores[document_id] = scores.get(document_id, 0.0) + (
                weight / (RRF_CONSTANT + rank)
            )

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [document_lookup[document_id] for document_id in ranked_ids]


def retrieve_hybrid(query, dense_index, bm25_retrievers):
    """Fuse BM25 and dense candidates separately within each source."""
    final_documents = []
    for source, result_limit in SOURCE_LIMITS.items():
        bm25_retriever = bm25_retrievers.get(source)
        if bm25_retriever is None:
            continue

        try:
            dense_candidates = dense_index.similarity_search(
                query,
                k=CANDIDATES_PER_SOURCE,
                filter={"source": source},
                fetch_k=200,
            )
        except TypeError:
            dense_candidates = dense_index.similarity_search(
                query,
                k=CANDIDATES_PER_SOURCE,
                filter={"source": source},
            )
        bm25_candidates = bm25_retriever.invoke(query)[:CANDIDATES_PER_SOURCE]
        fused = reciprocal_rank_fusion(bm25_candidates, dense_candidates)
        final_documents.extend(fused[:result_limit])

    return deduplicate_documents(final_documents)[:TOTAL_RESULTS]



# Metrics

def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the first K results that are relevant."""
    if k == 0:
        return 0.0
    return sum(item in relevant_ids for item in retrieved_ids[:k]) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of all known relevant results found in the first K."""
    if not relevant_ids:
        return 0.0
    return sum(item in relevant_ids for item in retrieved_ids[:k]) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant_ids:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Binary normalised discounted cumulative gain at K."""
    if not relevant_ids:
        return 0.0
    dcg = sum(
        int(document_id in relevant_ids) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0



# Evaluation, output, and summary

def run_with_timing(retrieval_function):
    """Return retrieved documents and mean latency over repeated runs."""
    timings = []
    documents = []
    for _ in range(TIMING_REPETITIONS):
        start_time = time.perf_counter()
        documents = retrieval_function()
        timings.append((time.perf_counter() - start_time) * 1000)
    return documents, mean(timings)


def display_results(method, question, documents, elapsed_ms):
    print("\n" + "=" * 80)
    print(f"Method: {method}")
    print(f"Question: {question}")
    print(f"Average retrieval time: {elapsed_ms:.2f} ms")
    print("=" * 80)
    for rank, document in enumerate(documents, start=1):
        print(f"\nRank {rank}: {get_document_id(document)}")
        print(preview_text(document))


def evaluate(dense_index, bm25_retrievers):
    """Evaluate Dense, BM25, and Hybrid retrieval."""
    metric_results = {"Dense": [], "BM25": [], "Hybrid": []}
    csv_rows = []

    for test_case in EVALUATION_QUESTIONS:
        question = test_case["question"]
        category = test_case["category"]
        relevant_ids = set(test_case["relevant_ids"])
        retrieval_methods = {
            "Dense": lambda: retrieve_dense(question, dense_index),
            "BM25": lambda: retrieve_bm25(question, bm25_retrievers),
            "Hybrid": lambda: retrieve_hybrid(
                question, dense_index, bm25_retrievers
            ),
        }

        print("\n\n" + "#" * 80)
        print(f"QUESTION: {question}")
        print(f"CATEGORY: {category}")
        print(f"GROUND TRUTH: {sorted(relevant_ids)}")
        print("#" * 80)

        for method, retrieval_function in retrieval_methods.items():
            documents, elapsed_ms = run_with_timing(retrieval_function)
            retrieved_ids = [get_document_id(document) for document in documents]
            display_results(method, question, documents, elapsed_ms)

            result = {
                "question": question,
                "category": category,
                "method": method,
                "precision_at_k": None,
                "recall_at_k": None,
                "reciprocal_rank": None,
                "ndcg_at_k": None,
                "average_time_ms": elapsed_ms,
                "retrieved_ids": ";".join(retrieved_ids),
            }

            if relevant_ids:
                result.update(
                    {
                        "precision_at_k": precision_at_k(
                            retrieved_ids, relevant_ids, TOTAL_RESULTS
                        ),
                        "recall_at_k": recall_at_k(
                            retrieved_ids, relevant_ids, TOTAL_RESULTS
                        ),
                        "reciprocal_rank": reciprocal_rank(
                            retrieved_ids, relevant_ids
                        ),
                        "ndcg_at_k": ndcg_at_k(
                            retrieved_ids, relevant_ids, TOTAL_RESULTS
                        ),
                    }
                )
                metric_results[method].append(result)
                print(f"\nPrecision@{TOTAL_RESULTS}: {result['precision_at_k']:.3f}")
                print(f"Recall@{TOTAL_RESULTS}:    {result['recall_at_k']:.3f}")
                print(f"Reciprocal rank:  {result['reciprocal_rank']:.3f}")
                print(f"nDCG@{TOTAL_RESULTS}:      {result['ndcg_at_k']:.3f}")
            else:
                print("\nNo ground-truth IDs assigned; this question is excluded from metric averages.")

            csv_rows.append(result)

    return metric_results, csv_rows


def save_results(csv_rows):
    """Save one row per question and retrieval method."""
    fieldnames = [
        "question", "category", "method", "precision_at_k", "recall_at_k",
        "reciprocal_rank", "ndcg_at_k", "average_time_ms", "retrieved_ids",
    ]
    with RESULTS_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nDetailed results saved to: {RESULTS_FILE}")


def print_summary(metric_results, csv_rows):
    """Print macro averages across labelled questions."""
    print("\n\n" + "=" * 94)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 94)
    print(
        f"\n{'Method':<12}{'Precision@8':>14}{'Recall@8':>12}"
        f"{'MRR':>10}{'nDCG@8':>12}{'Time (ms)':>14}"
    )
    print("-" * 94)

    for method in ["Dense", "BM25", "Hybrid"]:
        labelled = metric_results[method]
        method_rows = [row for row in csv_rows if row["method"] == method]
        average_time = mean(row["average_time_ms"] for row in method_rows)

        if labelled:
            precision = mean(row["precision_at_k"] for row in labelled)
            recall = mean(row["recall_at_k"] for row in labelled)
            mrr = mean(row["reciprocal_rank"] for row in labelled)
            ndcg = mean(row["ndcg_at_k"] for row in labelled)
        else:
            precision = recall = mrr = ndcg = 0.0

        print(
            f"{method:<12}{precision:>14.3f}{recall:>12.3f}"
            f"{mrr:>10.3f}{ndcg:>12.3f}{average_time:>14.2f}"
        )

    labelled_count = sum(bool(case["relevant_ids"]) for case in EVALUATION_QUESTIONS)
    print(f"\nLabelled questions included in quality averages: {labelled_count}/20")
    if labelled_count < len(EVALUATION_QUESTIONS):
        print("Questions with empty relevant_ids were excluded from quality averages.")



# Main


def main():
    print("RestoRec retrieval evaluation")
    documents_by_source = load_evaluation_documents()

    if LABELLING_MODE:
        print("\nLABELLING_MODE is True: printing candidate ground-truth documents.")
        run_labelling_searches(documents_by_source)
        print("\nAdd verified IDs to relevant_ids, then set LABELLING_MODE = False.")
        return

    unlabelled = [
        case["question"] for case in EVALUATION_QUESTIONS if not case["relevant_ids"]
    ]
    if unlabelled:
        print(f"\nWARNING: {len(unlabelled)} questions have no ground truth.")
        print("They will be excluded from Precision, Recall, MRR, and nDCG averages.")

    dense_index, bm25_retrievers = build_retrieval_indexes(documents_by_source)
    metric_results, csv_rows = evaluate(dense_index, bm25_retrievers)
    save_results(csv_rows)
    print_summary(metric_results, csv_rows)


if __name__ == "__main__":
    main()
