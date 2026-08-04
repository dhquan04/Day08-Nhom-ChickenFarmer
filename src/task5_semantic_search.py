"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store ChromaDB.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Tương thích với embedding model (all-MiniLM-L6-v2) và ChromaDB ở Task 4

Bonus (+5đ):
    - Tích hợp HyDE (Hypothetical Document Embeddings) để tối ưu hóa tìm kiếm vector.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Đảm bảo stdout in utf-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import từ Task 4
try:
    from .task4_chunking_indexing import get_collection, get_embedding_model
except ImportError:
    from task4_chunking_indexing import get_collection, get_embedding_model


def generate_hypothetical_document(query: str) -> str:
    """
    Kỹ thuật HyDE (Hypothetical Document Embeddings - Bonus +5đ):
    Sinh ra một câu trả lời/tài liệu giả định dựa trên câu hỏi của user, 
    giúp việc embed vector phản ánh đúng ngữ cảnh của tài liệu cần tìm.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết một đoạn văn bản ngắn (2-3 câu) giả định chứa câu trả lời chi tiết cho câu hỏi về chính sách Shopee dưới đây:"},
                    {"role": "user", "content": query}
                ],
                max_tokens=150,
                temperature=0.3
            )
            hypothetical_doc = response.choices[0].message.content.strip()
            return hypothetical_doc
        except Exception:
            pass

    # Template-based fallback nếu chưa gọi được LLM API
    return f"Quy định và hướng dẫn hỗ trợ khách hàng của Shopee về {query}. Bao gồm thời hạn, các bước thực hiện và điều kiện hoàn tiền thanh toán."


def semantic_search(query: str, top_k: int = 10, use_hyde: bool = False) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng Cosine Similarity trên ChromaDB.

    Args:
        query: Câu truy vấn của người dùng
        top_k: Số lượng kết quả tối đa
        use_hyde: Có kích hoạt HyDE (Hypothetical Document Embeddings) hay không

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0..1]
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            return []

        model = get_embedding_model()
        search_text = generate_hypothetical_document(query) if use_hyde else query
        query_vector = model.encode(search_text).tolist()
    except Exception as e:
        print(f"  [Note] Semantic Search model loading fallback: {e}")
        return []

    # Query ChromaDB (Cosine distance = 1 - cosine similarity)
    fetch_k = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # ChromaDB cosine space: distance = 1 - similarity -> similarity = 1 - distance
        score = max(0.0, 1.0 - dist)
        output.append({
            "content": doc,
            "score": round(score, 4),
            "metadata": meta
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    test_query = "quy định trả hàng hoàn tiền shopee"
    print(f"Testing Semantic Search for query: '{test_query}'")
    results = semantic_search(test_query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['content'][:90]}...")
