"""
Task 7 — Reranking Module.

Phương pháp:
    - RRF (Reciprocal Rank Fusion): gộp thứ hạng từ nhiều kết quả tìm kiếm.
    - MMR (Maximal Marginal Relevance): cân bằng giữa độ liên quan và sự đa dạng.
    - Cross-encoder reranker: đánh giá cặp (query, document).
"""

import math
from typing import List, Dict, Any, Optional


def rerank_cross_encoder(
    query: str, candidates: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank candidates sử dụng cross-encoder hoặc keyword relevance scoring.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    query_words = set(query.lower().split())
    scored_candidates = []
    
    for item in candidates:
        content = item.get("content", "").lower()
        content_words = set(content.split())
        
        # Calculate term overlap relevance
        if query_words and content_words:
            overlap = len(query_words.intersection(content_words)) / len(query_words)
        else:
            overlap = 0.0
            
        orig_score = float(item.get("score", 0.0))
        new_score = 0.6 * orig_score + 0.4 * overlap
        
        new_item = item.copy()
        new_item["score"] = new_score
        scored_candidates.append(new_item)

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates[:top_k]


def rerank_mmr(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    def cosine_sim(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            doc_emb = candidates[idx].get("embedding", [])
            if doc_emb and query_embedding:
                relevance = cosine_sim(query_embedding, doc_emb)
            else:
                relevance = float(candidates[idx].get("score", 0.0))

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sel_emb = candidates[sel_idx].get("embedding", [])
                if doc_emb and sel_emb:
                    sim = cosine_sim(doc_emb, sel_emb)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    results = []
    for idx in selected_indices:
        item = candidates[idx].copy()
        results.append(item)
    return results[:top_k]


def rerank_rrf(
    ranked_lists: List[List[Dict[str, Any]]], top_k: int = 5, k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    if not ranked_lists:
        return []

    rrf_scores: Dict[str, float] = {}
    content_map: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        if not isinstance(ranked_list, list):
            continue
        for rank, item in enumerate(ranked_list, 1):
            content = item.get("content", "")
            if not content:
                continue
            rrf_scores[content] = rrf_scores.get(content, 0.0) + 1.0 / (k + rank)
            if content not in content_map:
                content_map[content] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    method: str = "rrf",
) -> List[Dict[str, Any]]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval (hoặc list of ranked lists nếu RRF)
        top_k: Số lượng kết quả sau rerank
        method: "rrf" | "cross_encoder" | "mmr"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "rrf":
        # Check if candidates is a list of lists or single list
        if candidates and isinstance(candidates[0], list):
            ranked_lists = candidates
        else:
            # Single list of candidates: create 2 variations (original score + keyword overlap score)
            celist = rerank_cross_encoder(query, candidates, top_k=len(candidates))
            ranked_lists = [candidates, celist]
        return rerank_rrf(ranked_lists, top_k=top_k)

    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)

    elif method == "mmr":
        dummy_query_emb = [0.1] * 384
        return rerank_mmr(dummy_query_emb, candidates, top_k=top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    print(f"Reranked {len(results)} items:")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
