"""
Mission 05 — Part 2: Communications & Protocol Officer
Okapi BM25 Lexical Retrieval & Reciprocal Rank Fusion (RRF)

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core
Cadet Assigned: Anas Asif Hussain Shaik (Communications & Protocol Officer)
"""

import math
from collections import Counter
from typing import List, Dict, Tuple, Set


class BM25Okapi:
    """
    Okapi BM25 scoring engine for exact keyword preservation and sparse lexical retrieval.
    BM25(D, Q) = sum_{q_i in Q} IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)))
    """
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.raw_corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)

        # Document frequencies for all unique terms
        self.df: Dict[str, int] = Counter()
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.df[term] += 1

        # Inverse Document Frequency (Robertson-SpÃ¤rck Jones IDF formula)
        self.idf: Dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple clean whitespace & punctuation tokenizer."""
        clean_text = text.lower().replace("-", " - ").replace("/", " / ")
        for char in [",", ".", ";", ":", "(", ")", "[", "]", "{", "}", "\"", "'", "!", "?"]:
            clean_text = clean_text.replace(char, " ")
        return [t for t in clean_text.split() if t]

    def score(self, query: str) -> List[Tuple[int, float]]:
        """Scores each document in the corpus against the query."""
        query_terms = self._tokenize(query)
        scores = []

        for doc_idx, doc in enumerate(self.tokenized_corpus):
            doc_len = self.doc_lens[doc_idx]
            tf = Counter(doc)
            score = 0.0

            for term in query_terms:
                if term in tf:
                    term_tf = tf[term]
                    numerator = term_tf * (self.k1 + 1.0)
                    denominator = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-9)))
                    score += self.idf.get(term, 0.0) * (numerator / denominator)

            scores.append((doc_idx, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def reciprocal_rank_fusion(
    dense_rankings: List[int],
    sparse_rankings: List[int],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Merges dense vector ranking and sparse lexical ranking via Reciprocal Rank Fusion (RRF).
    RRF(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))
    using smoothing constant k = 60.
    """
    rrf_scores: Dict[int, float] = {}

    for rank, doc_id in enumerate(dense_rankings):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(sparse_rankings):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


class DualPathHybridRetriever:
    """
    Couples BM25 lexical search with dense vector embeddings via Reciprocal Rank Fusion.
    Solves both exact serial code matching and high-level semantic synthesis.
    """
    def __init__(self, corpus: List[str], dense_embeddings: List[List[float]] = None, k_rrf: int = 60):
        self.corpus = corpus
        self.bm25 = BM25Okapi(corpus)
        self.k_rrf = k_rrf
        self.dense_embeddings = dense_embeddings

    def search_hybrid(
        self,
        query: str,
        dense_ranked_ids: List[int] = None,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Executes hybrid search combining BM25 lexical scores with provided dense ranking.
        """
        # 1. Sparse BM25 retrieval
        sparse_results = self.bm25.score(query)
        sparse_ranked_ids = [doc_id for doc_id, score in sparse_results]

        # 2. Dense fallback if none provided
        if dense_ranked_ids is None:
            # Semantic keyword heuristic as placeholder dense rank
            dense_ranked_ids = list(range(len(self.corpus)))

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(dense_ranked_ids, sparse_ranked_ids, k=self.k_rrf)

        results = []
        for doc_id, rrf_score in fused[:top_n]:
            sparse_score = next((s for d, s in sparse_results if d == doc_id), 0.0)
            results.append({
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "bm25_score": sparse_score,
                "text": self.corpus[doc_id]
            })
        return results


if __name__ == "__main__":
    test_corpus = [
        "AURA Subsystem 05 contains the dual-path associative memory core in Sector SEC-05-M01.",
        "Pulsar radiation in Crab nebula induces high frequency electromagnetic interference at 30.22 Hz.",
        "Warp vector plasma manifold alignment code PR-882-HEX must be phase locked.",
        "Emergency magnetic deflector shield harmonic code is MAG-SHIELD-77B.",
        "Cryo-coolant emergency purge valve serial number is VLV-904-CRY.",
        "General astrogation protocol: ungrounded parametric models suffer from severe confabulation."
    ]

    print("=== Subsystem 05: BM25 Lexical Retrieval Demonstration ===")
    bm25 = BM25Okapi(test_corpus)
    query_exact = "PR-882-HEX plasma manifold"
    sparse_scores = bm25.score(query_exact)

    for doc_id, sc in sparse_scores[:3]:
        print(f"Doc {doc_id} [BM25: {sc:.4f}] -> {test_corpus[doc_id]}")

    print("\n=== Subsystem 05: Reciprocal Rank Fusion (k=60) Hybrid Search ===")
    # Dense rank: simulated semantic match prioritizing general propulsion
    simulated_dense_ranks = [0, 2, 5, 1, 3, 4]
    sparse_ranks = [d for d, _ in sparse_scores]

    rrf_combined = reciprocal_rank_fusion(simulated_dense_ranks, sparse_ranks, k=60)
    for doc_id, rrf_val in rrf_combined[:3]:
        print(f"Fused Rank -> Doc {doc_id} (RRF Score: {rrf_val:.6f}) => {test_corpus[doc_id]}")
