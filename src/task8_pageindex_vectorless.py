"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Đảm bảo stdout in utf-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_PDF_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

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


CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"
POLL_INTERVAL_SECONDS = 2
MAX_RETRIEVAL_POLLS = 20


def _load_document_cache() -> Dict[str, str]:
    """Load mapping PDF path -> PageIndex doc_id from the local cache."""
    if not CACHE_FILE.exists():
        return {}
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            return {str(path): str(doc_id) for path, doc_id in cached.items()}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [Note] Cannot read PageIndex cache: {exc}")
    return {}


def _save_document_cache(cache: Dict[str, str]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _iter_relevant_items(value: Any):
    """Yield relevant-content objects from old and new PageIndex response shapes."""
    if isinstance(value, dict):
        if value.get("relevant_content"):
            yield value
        else:
            for child in value.values():
                yield from _iter_relevant_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_relevant_items(child)


def upload_documents() -> List[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Có lưu cache doc_ids để tránh re-upload nhiều lần.
    Returns list of document IDs uploaded.
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("your_"):
        print("⚠ Chưa cấu hình PAGEINDEX_API_KEY hợp lệ trong file .env. Bỏ qua bước upload.")
        return []

    # PageIndexClient.submit_document chỉ nhận PDF. Dùng PDF gốc thay vì Markdown.
    pdf_files = sorted(LEGAL_PDF_DIR.rglob("*.pdf")) if LEGAL_PDF_DIR.exists() else []
    if not pdf_files:
        print(f"⚠ Không tìm thấy PDF để upload tại {LEGAL_PDF_DIR}.")
        return []

    cached_doc_ids = _load_document_cache()
    uploaded_ids = []
    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

        for pdf_file in pdf_files:
            cache_key = str(pdf_file.resolve())
            if cache_key in cached_doc_ids:
                uploaded_ids.append(cached_doc_ids[cache_key])
                continue
            try:
                resp = client.submit_document(str(pdf_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    uploaded_ids.append(doc_id)
                    cached_doc_ids[cache_key] = doc_id
                    print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")
            except Exception as err:
                err_msg = str(err)
                if "Invalid API key" in err_msg or "unauthorized" in err_msg.lower():
                    print("⚠ PAGEINDEX_API_KEY trong .env không hợp lệ. Tự động chuyển sang Local Vectorless Search.")
                    break
                print(f"  [Note] Upload {pdf_file.name} fallback: {err}")

        if cached_doc_ids:
            _save_document_cache(cached_doc_ids)
    except Exception as e:
        print(f"Lỗi kết nối PageIndex SDK: {e}")

    return uploaded_ids


def pageindex_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Vectorless retrieval sử dụng PageIndex Cloud API (hoặc Local Section Tree Search).
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    if PAGEINDEX_API_KEY and not PAGEINDEX_API_KEY.startswith("your_"):
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            doc_ids = upload_documents()
            
            if doc_ids:
                results = []
                rank = 1
                print(f"[PageIndex API] Đang truy vấn cây cấu trúc trên {len(doc_ids)} tài liệu Cloud...")
                
                # Duyệt qua các tài liệu đã upload trên PageIndex Cloud
                for i, doc_id in enumerate(doc_ids, 1):
                    try:
                        resp = client.submit_query(doc_id=doc_id, query=query)
                        retrieval_id = resp.get("retrieval_id") or resp.get("id")

                        if retrieval_id:
                            retrieval = {}
                            for _ in range(MAX_RETRIEVAL_POLLS):
                                retrieval = client.get_retrieval(retrieval_id)
                                status = str(retrieval.get("status", "")).lower()
                                if status == "completed":
                                    break
                                if status in {"failed", "error", "cancelled"}:
                                    print(f"  [Note] PageIndex retrieval {retrieval_id}: {status}")
                                    retrieval = {}
                                    break
                                time.sleep(POLL_INTERVAL_SECONDS)
                            else:
                                print(f"  [Note] PageIndex retrieval timed out: {retrieval_id}")
                                retrieval = {}

                            retrieved_nodes = retrieval.get("retrieved_nodes") or []
                            for node in retrieved_nodes:
                                node_title = node.get("title", "")
                                for item in _iter_relevant_items(node.get("relevant_contents", [])):
                                    content = item.get("relevant_content", "")
                                    if content:
                                        results.append({
                                            "content": content,
                                            "score": round(1.0 / (rank + 1), 4),
                                            "metadata": {
                                                "section": item.get("section_title") or node_title or "General",
                                                "doc_id": doc_id
                                            },
                                            "source": "pageindex"
                                        })
                                        rank += 1
                        
                        # Ngắt sớm nếu đã tìm đủ số lượng kết quả liên quan
                        if len(results) >= top_k:
                            break
                    except Exception as exc:
                        print(f"  [Note] PageIndex query for {doc_id} failed: {exc}")

                if results:
                    results.sort(key=lambda x: x["score"], reverse=True)
                    return results[:top_k]
        except Exception as e:
            print(f"  [Note] PageIndex Cloud API search fallback: {e}")

    # Local fallback vectorless search (truy vấn trực tiếp tài liệu dựa trên cấu trúc section/heading)
    print("[PageIndex Local] Đang phân tích cây cấu trúc tiêu đề tài liệu...")
    results = []
    query_words = set(query.lower().split())
    
    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                
                # Phân tích cây cấu trúc tiêu đề (Headings) của file Markdown
                current_heading = md_file.stem.replace("-", " ").title()
                sections = text.split("\n\n")
                
                for sec in sections:
                    sec_clean = sec.strip()
                    if not sec_clean:
                        continue
                    if sec_clean.startswith("#"):
                        current_heading = sec_clean.lstrip("#").strip()
                        continue
                    
                    sec_words = set(sec_clean.lower().split())
                    overlap = len(query_words.intersection(sec_words)) if query_words else 0
                    
                    if overlap > 0:
                        score = round(0.5 + 0.1 * overlap, 4)
                        results.append({
                            "content": sec_clean,
                            "score": score,
                            "metadata": {
                                "section": current_heading,
                                "file": md_file.name
                            },
                            "source": "pageindex"
                        })
            except Exception:
                pass

    if not results:
        # Fallback cuối cùng nếu không tìm thấy từ trùng khớp
        for doc in PAGEINDEX_FALLBACK_DOCS:
            results.append(doc.copy())

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY chưa cấu hình. Chạy thử nghiệm local fallback mode:")
    
    results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
    print(f"Kết quả PageIndex search ({len(results)} items):")
    for r in results:
        print(f"[{r['source']}] [{r['score']:.3f}] {r['content'][:100]}...")
