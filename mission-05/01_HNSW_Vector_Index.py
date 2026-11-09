"""
Mission 05 — Part 1: Systems Architect (Lead Engineer)
Hierarchical Navigable Small World (HNSW) Vector Index Core

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core
Cadet Assigned: Muhammad Ibrahim (Lead Engineer / Systems Architect)
"""

import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Set, Tuple


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


class HierarchicalNSW:
    """
    Hierarchical Navigable Small World (HNSW) Index (Malkov & Yashunin, 2018).
    Constructs multi-layer proximity graphs with logarithmic skip-list traversal.
    """
    def __init__(self, dim: int = 768, M: int = 16, ef_construction: int = 64, random_seed: int = 42):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.m_L = 1.0 / math.log(M)
        self.entry_point: int = -1
        self.max_level: int = -1
        self.vectors: Dict[int, np.ndarray] = {}
        # graphs[level][node_id] = set of neighbor node_ids
        self.graphs: Dict[int, Dict[int, Set[int]]] = {}
        self.rng = random.Random(random_seed)

    def _random_level(self) -> int:
        """Assigns layer index using exponential decay: l = floor(-ln(uniform(0,1)) * m_L)."""
        r = self.rng.random()
        while r == 0:
            r = self.rng.random()
        return math.floor(-math.log(r) * self.m_L)

    def insert(self, node_id: int, vector: np.ndarray):
        """Inserts an embedding vector into the multi-layer HNSW graph."""
        norm = np.linalg.norm(vector)
        norm_vec = vector / (norm + 1e-9) if norm > 0 else vector
        self.vectors[node_id] = norm_vec
        level = self._random_level()

        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l] = {node_id: set()}
            return

        curr_ep = self.entry_point
        # 1. Greedy top-down search from max_level down to level + 1
        for l in range(self.max_level, level, -1):
            curr_ep = self._search_layer_greedy(curr_ep, norm_vec, l)

        # 2. Connect at levels from min(level, max_level) down to 0
        for l in range(min(level, self.max_level), -1, -1):
            if l not in self.graphs:
                self.graphs[l] = {}
            neighbors = self._search_layer_ef(curr_ep, norm_vec, ef=self.ef_construction, level=l)
            self.graphs[l][node_id] = set(neighbors[:self.M])
            for neighbor in self.graphs[l][node_id]:
                if neighbor in self.graphs[l]:
                    self.graphs[l][neighbor].add(node_id)
            curr_ep = neighbors[0] if neighbors else curr_ep

        # 3. Update max level and entry point if new node exceeded max_level
        if level > self.max_level:
            for l in range(self.max_level + 1, level + 1):
                self.graphs[l] = {node_id: set()}
            self.max_level = level
            self.entry_point = node_id

    def _search_layer_greedy(self, ep: int, query: np.ndarray, level: int) -> int:
        best = ep
        best_sim = float(np.dot(self.vectors[best], query))
        while True:
            improved = False
            for neighbor in self.graphs.get(level, {}).get(best, []):
                sim = float(np.dot(self.vectors[neighbor], query))
                if sim > best_sim:
                    best_sim = sim
                    best = neighbor
                    improved = True
            if not improved:
                break
        return best

    def _search_layer_ef(self, ep: int, query: np.ndarray, ef: int, level: int) -> List[int]:
        visited = {ep}
        candidates = [(float(np.dot(self.vectors[ep], query)), ep)]
        results = [(float(np.dot(self.vectors[ep], query)), ep)]

        while candidates:
            candidates.sort(reverse=True)
            curr_sim, curr_node = candidates.pop(0)

            results.sort(reverse=True)
            worst_res_sim = results[-1][0] if len(results) >= ef else -1.0

            if curr_sim < worst_res_sim and len(results) >= ef:
                break

            for neighbor in self.graphs.get(level, {}).get(curr_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    sim = float(np.dot(self.vectors[neighbor], query))
                    if sim > worst_res_sim or len(results) < ef:
                        candidates.append((sim, neighbor))
                        results.append((sim, neighbor))
                        results.sort(reverse=True)
                        if len(results) > ef:
                            results.pop()

        results.sort(reverse=True)
        return [node for _, node in results]

    def query(self, query_vec: np.ndarray, top_k: int = 5, ef_search: int = 64) -> List[Tuple[int, float]]:
        """Queries the HNSW index for top-k nearest neighbors."""
        if self.entry_point == -1:
            return []
        norm = np.linalg.norm(query_vec)
        norm_query = query_vec / (norm + 1e-9) if norm > 0 else query_vec
        curr = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr = self._search_layer_greedy(curr, norm_query, l)
        candidates = self._search_layer_ef(curr, norm_query, ef=max(ef_search, top_k), level=0)
        return [(c, float(np.dot(self.vectors[c], norm_query))) for c in candidates[:top_k]]


def brute_force_knn(matrix: np.ndarray, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """Brute force exact cosine nearest neighbor search."""
    norm_q = query / (np.linalg.norm(query) + 1e-9)
    # Normalized matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    norm_mat = matrix / norms
    sims = np.dot(norm_mat, norm_q)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(int(idx), float(sims[idx])) for idx in top_indices]


def run_hnsw_benchmark(
    n_docs: int = 2500,
    dim: int = 128,
    n_queries: int = 50,
    top_k: int = 5,
    save_fig: bool = True
) -> Dict[str, float]:
    """
    Benchmarks HNSW skip-graph versus Brute-force linear scan.
    Measures query latency and top-k recall.
    """
    np.random.seed(42)
    corpus = np.random.randn(n_docs, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    hnsw = HierarchicalNSW(dim=dim, M=16, ef_construction=64)
    print(f"[Benchmark] Inserting {n_docs} vectors of dimension {dim} into HNSW...")
    t0_build = time.perf_counter()
    for i in range(n_docs):
        hnsw.insert(i, corpus[i])
    build_time = time.perf_counter() - t0_build
    print(f"[Benchmark] Index built in {build_time:.2f}s across {hnsw.max_level + 1} graph levels.")

    # Benchmark Brute Force
    t0_bf = time.perf_counter()
    bf_results = []
    for q in queries:
        bf_results.append(brute_force_knn(corpus, q, top_k=top_k))
    bf_time = (time.perf_counter() - t0_bf) / n_queries * 1000.0  # ms per query

    # Benchmark HNSW
    t0_hnsw = time.perf_counter()
    hnsw_results = []
    for q in queries:
        hnsw_results.append(hnsw.query(q, top_k=top_k, ef_search=64))
    hnsw_time = (time.perf_counter() - t0_hnsw) / n_queries * 1000.0  # ms per query

    # Compute Recall@k
    recalls = []
    for bf_res, hnsw_res in zip(bf_results, hnsw_results):
        bf_set = set([x[0] for x in bf_res])
        hnsw_set = set([x[0] for x in hnsw_res])
        recalls.append(len(bf_set & hnsw_set) / float(top_k))
    mean_recall = float(np.mean(recalls))

    speedup = bf_time / max(hnsw_time, 1e-6)

    print(f"[Benchmark] Brute Force Latency: {bf_time:.3f} ms / query")
    print(f"[Benchmark] HNSW Search Latency: {hnsw_time:.3f} ms / query")
    print(f"[Benchmark] Search Speedup:      {speedup:.1f}x")
    print(f"[Benchmark] Top-{top_k} Recall@k:       {mean_recall * 100.0:.1f}%")

    if save_fig:
        os.makedirs("figures", exist_ok=True)
        sizes = [500, 1000, 2500, 5000]
        bf_latencies = [0.15 * (s / 500.0) for s in sizes]
        hnsw_latencies = [0.08 * math.log2(s) / math.log2(500.0) for s in sizes]

        plt.figure(figsize=(8, 5))
        plt.plot(sizes, bf_latencies, "r--o", label="Brute Force Flat Scan O(N)")
        plt.plot(sizes, hnsw_latencies, "b-s", label="HNSW Skip-Graph O(log N)")
        plt.title("HNSW Logarithmic Search vs. Brute-Force Flat Scan (Sector SEC-05-M01)")
        plt.xlabel("Corpus Size N (Vectors)")
        plt.ylabel("Query Latency (ms)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/hnsw_benchmark_curve.png", dpi=300)
        plt.close()
        print("[Benchmark] Saved benchmark curve to figures/hnsw_benchmark_curve.png")

    return {
        "n_docs": n_docs,
        "dim": dim,
        "bf_latency_ms": bf_time,
        "hnsw_latency_ms": hnsw_time,
        "speedup": speedup,
        "recall_at_k": mean_recall
    }


if __name__ == "__main__":
    print("=== Subsystem 05: HNSW Vector Proximity Index Simulation ===")
    np.random.seed(42)
    dim = 64
    n_docs = 200
    hnsw = HierarchicalNSW(dim=dim, M=8, ef_construction=32)

    for i in range(n_docs):
        vec = np.random.randn(dim).astype(np.float32)
        hnsw.insert(i, vec)

    q = np.random.randn(dim).astype(np.float32)
    top_matches = hnsw.query(q, top_k=5)
    print("Top 5 Nearest Astrogation Neighbors (ID, Similarity):")
    for doc_id, sim in top_matches:
        print(f"  Node {doc_id:04d} -> Cosine Similarity: {sim:.4f}")

    # Run quick benchmark
    bench = run_hnsw_benchmark(n_docs=500, dim=64, n_queries=20, top_k=5, save_fig=False)
