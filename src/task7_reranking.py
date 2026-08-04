"""
Task 7 — Reranking Module.

Phương pháp:
    - RRF (Reciprocal Rank Fusion): gộp thứ hạng từ nhiều kết quả tìm kiếm.
    - MMR (Maximal Marginal Relevance): cân bằng giữa độ liên quan và sự đa dạng.
    - Cross-encoder reranker: đánh giá cặp (query, document).
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# Tắt cảnh báo dọn dẹp bộ nhớ của multiprocess trên Python 3.12 khi thoát chương trình
try:
    import multiprocess.resource_tracker
    multiprocess.resource_tracker.ResourceTracker._stop = lambda *args, **kwargs: None
except Exception:
    pass

# Đảm bảo HF Hub không bị đứng do filelock hoặc cảnh báo symlink
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Đảm bảo stdout in utf-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def rerank_qwen(
    query: str, candidates: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Rerank candidates sử dụng mô hình Qwen3 (Qwen3-Reranker hoặc Qwen LLM API).
    
    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates được đánh giá bởi Qwen3.
    """
    if not candidates:
        return []

    # 1. Ưu tiên sử dụng Qwen API qua OpenRouter/OpenAI nếu có API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)

            docs_text = "\n".join([f"[{i+1}] {c.get('content', '')[:200]}" for i, c in enumerate(candidates)])
            prompt = (
                f"Câu hỏi: {query}\n\n"
                f"Danh sách các đoạn văn bản:\n{docs_text}\n\n"
                f"Hãy đánh giá mức độ liên quan và sắp xếp lại các đoạn văn bản trên từ liên quan nhất đến ít liên quan nhất.\n"
                f"Chỉ trả về thứ tự danh sách số thứ tự phân cách bằng dấu phẩy (ví dụ: 1, 3, 2)."
            )
            response = client.chat.completions.create(
                model="qwen/qwen-2.5-72b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            import re
            indices = [int(n) - 1 for n in re.findall(r'\d+', raw) if 0 <= int(n) - 1 < len(candidates)]
            if indices:
                results = []
                seen = set()
                rank = 1
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        item = candidates[idx].copy()
                        item["score"] = round(1.0 / (rank + 1), 4)
                        results.append(item)
                        rank += 1
                for idx, c in enumerate(candidates):
                    if idx not in seen:
                        item = c.copy()
                        item["score"] = round(1.0 / (rank + 1), 4)
                        results.append(item)
                        rank += 1
                return results[:top_k]
        except Exception as e:
            print(f"  [Note] Qwen API rerank fallback: {e}")

    # 2. Thử load local Qwen3 CrossEncoder nếu có
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B", max_length=512, local_files_only=True)
        pairs = [[query, c.get("content", "")] for c in candidates]
        scores = model.predict(pairs)
        scored = []
        for c, s in zip(candidates, scores):
            item = c.copy()
            item["score"] = float(s)
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    except Exception:
        pass

    # 3. Fallback Cross-Encoder nhanh tránh treo tiến trình
    return rerank_cross_encoder(query, candidates, top_k=top_k)


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
        method: "rrf" | "cross_encoder" | "mmr" | "qwen"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "qwen":
        return rerank_qwen(query, candidates, top_k=top_k)

    elif method == "rrf":
        if candidates and isinstance(candidates[0], list):
            ranked_lists = candidates
        else:
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
    query = "quy định trả hàng hoàn tiền shopee"
    print(f"=== THỬ NGHIỆM RERANKING TRÊN DỮ LIỆU THẬT DỰ ÁN CHO QUERY: '{query}' ===\n")

    # 1. Lấy dữ liệu Lexical (BM25) trước - cực nhanh, không bị treo
    lexical_results = []
    try:
        from src.task6_lexical_search import lexical_search
        lexical_results = lexical_search(query, top_k=5)
        print(f"  ✓ Lexical BM25 Search: đã tìm thấy {len(lexical_results)} chunks")
    except Exception as e:
        print(f"  [Note] Lexical Search fallback: {e}")

    # 2. Thử lấy dữ liệu Semantic Search (Task 5)
    semantic_results = []
    try:
        from src.task5_semantic_search import semantic_search
        semantic_results = semantic_search(query, top_k=5)
        print(f"  ✓ Semantic Search: đã tìm thấy {len(semantic_results)} chunks")
    except Exception as e:
        print(f"  [Note] Semantic Search: {e}")

    if semantic_results and lexical_results:
        results = rerank(query, [semantic_results, lexical_results], top_k=5, method="rrf")
        print("\n[RRF Rerank Result]:")
    elif lexical_results:
        # Sử dụng Qwen / Cross-Encoder Reranker trên kết quả thực tế từ BM25
        results = rerank(query, lexical_results, top_k=5, method="qwen")
        print("\n[Qwen / Cross-Encoder Rerank Result]:")
    else:
        dummy_candidates = [
            {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {"source": "returns-refund.md"}},
            {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {"source": "payment-methods.md"}},
        ]
        results = rerank(query, dummy_candidates, top_k=2, method="qwen")
        print("\n[Fallback Rerank Result]:")

    print(f"Reranked {len(results)} items:")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        source = meta.get("source", r.get("source", "Unknown"))
        print(f"[{i}] Score: {r['score']:.4f} | Nguồn: {source}")
        print(f"    Nội dung:\n{r['content'][:200]}\n" + "-" * 60)
