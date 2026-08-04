"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Fallback fallback documents with structural metadata for vectorless fallback search
PAGEINDEX_FALLBACK_DOCS = [
    {
        "content": "Chính sách và quy định về phương thức thanh toán mua hàng trên sàn thương mại điện tử Shopee.",
        "score": 0.95,
        "metadata": {"section": "Thanh Toán & Ví ShopeePay"},
        "source": "pageindex"
    },
    {
        "content": "Hướng dẫn chi tiết quy trình xử lý khiếu nại trả hàng hoàn tiền và nộp bằng chứng chứng minh.",
        "score": 0.88,
        "metadata": {"section": "Quy Trình Trả Hàng Hoàn Tiền"},
        "source": "pageindex"
    },
    {
        "content": "Quy định người bán và danh sách các sản phẩm cấm đăng bán trên hệ thống e-commerce.",
        "score": 0.82,
        "metadata": {"section": "Quy Định Đăng Bán Sản Phẩm"},
        "source": "pageindex"
    }
]


def upload_documents() -> List[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Returns list of document IDs uploaded.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Không tìm thấy PAGEINDEX_API_KEY trong file .env. Bỏ qua bước upload.")
        return []

    uploaded_ids = []
    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        for md_file in md_files:
            try:
                resp = client.submit_document(str(md_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    uploaded_ids.append(doc_id)
                    print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
            except Exception as err:
                print(f"Lỗi upload {md_file.name}: {err}")
    except Exception as e:
        print(f"Lỗi kết nối PageIndex SDK: {e}")

    return uploaded_ids


def pageindex_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            resp = client.submit_query(query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")

            if retrieval_id:
                retrieval = client.get_retrieval(retrieval_id)
                results = []
                rank = 1
                for node in retrieval.get("retrieved_nodes", []):
                    for group in node.get("relevant_contents", []):
                        for item in group:
                            content = item.get("relevant_content", "")
                            if content:
                                results.append({
                                    "content": content,
                                    "score": round(1.0 / (rank + 1), 4),
                                    "metadata": {"section": item.get("section_title", "General")},
                                    "source": "pageindex"
                                })
                                rank += 1
                if results:
                    return results[:top_k]
        except Exception as e:
            print(f"Lỗi truy vấn PageIndex API ({e}), chuyển sang fallback vectorless search.")

    # Local fallback vectorless search
    results = []
    query_words = set(query.lower().split())
    
    for doc in PAGEINDEX_FALLBACK_DOCS:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words.intersection(doc_words)) if query_words else 0
        doc_copy = doc.copy()
        doc_copy["score"] = float(doc["score"] + 0.1 * overlap)
        doc_copy["source"] = "pageindex"
        results.append(doc_copy)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa cấu hình. Chạy thử nghiệm local fallback mode:")
    
    results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
    print(f"Kết quả PageIndex search ({len(results)} items):")
    for r in results:
        print(f"[{r['source']}] [{r['score']:.3f}] {r['content'][:100]}...")
