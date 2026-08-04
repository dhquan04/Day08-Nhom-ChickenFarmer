"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng ban đầu cho MiniLM + corpus lab; cần calibrate lại khi đổi corpus/model.
SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    fetch_k = max(top_k * 2, top_k)

    # Dense và sparse search độc lập nên có thể chạy đồng thời.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, fetch_k)
        sparse_future = executor.submit(lexical_search, query, fetch_k)
        try:
            dense_results = dense_future.result()
        except Exception as exc:
            print(f"  [Note] Semantic search unavailable: {exc}")
            dense_results = []
        try:
            sparse_results = sparse_future.result()
        except Exception as exc:
            print(f"  [Note] Lexical search unavailable: {exc}")
            sparse_results = []

    # RRF chỉ dùng để gộp thứ hạng. Không dùng score RRF để quyết định fallback.
    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    # RRF đã là một bước fusion. Chỉ rerank thêm nếu chọn một phương pháp khác.
    if use_reranking and merged and RERANK_METHOD not in {"rrf", ""}:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Dùng cosine score gốc từ dense search, tuyệt đối không dùng RRF score.
    best_dense_score = float(dense_results[0].get("score", 0.0)) if dense_results else 0.0
    if best_dense_score < score_threshold:
        print(
            f"  [Fallback] Best dense score {best_dense_score:.3f} "
            f"< threshold {score_threshold:.3f}"
        )
        try:
            fallback_results = pageindex_search(query, top_k=top_k)
        except Exception as exc:
            print(f"  [Note] PageIndex fallback unavailable: {exc}")
            fallback_results = []
        if fallback_results:
            for item in fallback_results:
                item["source"] = "pageindex"
            return fallback_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
