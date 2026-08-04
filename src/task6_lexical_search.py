"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 (rank-bm25).
BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document -> điểm cao
    - Inverse Document Frequency (IDF): từ hiếm -> quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tắt cảnh báo dọn dẹp bộ nhớ của multiprocess trên Python 3.12 khi thoát chương trình
try:
    import multiprocess.resource_tracker
    multiprocess.resource_tracker.ResourceTracker._stop = lambda *args, **kwargs: None
except Exception:
    pass

# Đảm bảo stdout in utf-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

# Default fallback corpus if no markdown files are found in data/standardized/ yet
DEFAULT_CORPUS: List[Dict[str, Any]] = [
    {
        "content": "Chính sách trả hàng và hoàn tiền Shopee Vietnam trong vòng 15 ngày. Khách hàng cần cung cấp bằng chứng hình ảnh và video quay lại sản phẩm khi mở hộp để yêu cầu refund policy.",
        "metadata": {"source": "returns-refund-policy-shopee.md", "category": "legal"}
    },
    {
        "content": "Các phương thức thanh toán hỗ trợ trên Shopee bao gồm: Thẻ tín dụng, Thẻ ghi nợ, Ví ShopeePay, Thanh toán khi nhận hàng (COD) và Chuyển khoản ngân hàng (payment methods).",
        "metadata": {"source": "payment-methods-shopee.md", "category": "legal"}
    },
    {
        "content": "Quy định đăng bán sản phẩm dành cho người bán (seller listing regulations). Các mặt hàng bị cấm đăng bán bao gồm hàng giả, hàng nhái, chất cấm và vũ khí.",
        "metadata": {"source": "product-listing-regulations-shopee.md", "category": "legal"}
    },
    {
        "content": "Hướng dẫn theo dõi đơn hàng và quy trình tra cứu hành trình vận chuyển cho người mua (order tracking guide).",
        "metadata": {"source": "order-tracking-guide.md", "category": "news"}
    },
    {
        "content": "Quy định bảo mật thông tin cá nhân và bảo vệ dữ liệu người dùng trên nền tảng thương mại điện tử (privacy policy).",
        "metadata": {"source": "privacy-policy-shopee.md", "category": "legal"}
    }
]


def load_corpus() -> List[Dict[str, Any]]:
    """
    Load corpus từ các file markdown trong data/standardized/.
    Ưu tiên dùng chunking đồng bộ với Task 4 (RecursiveCharacterTextSplitter).
    Nếu chưa có, dùng fallback parser hoặc DEFAULT_CORPUS.
    """
    corpus = []
    
    # 1. Thử dùng hàm chunking chuẩn từ Task 4 nếu có
    try:
        try:
            from .task4_chunking_indexing import load_documents, chunk_documents
        except ImportError:
            try:
                from task4_chunking_indexing import load_documents, chunk_documents
            except ImportError:
                from src.task4_chunking_indexing import load_documents, chunk_documents
        
        docs = load_documents()
        if docs:
            chunks = chunk_documents(docs)
            for c in chunks:
                meta = c.get("metadata", {}).copy()
                content = c.get("content", "")
                
                # Trích xuất thêm title nếu chưa có
                if "title" not in meta:
                    meta["title"] = meta.get("source", "").replace(".md", "").replace("-", " ").title()
                
                corpus.append({
                    "content": content,
                    "metadata": meta
                })
            if corpus:
                return corpus
    except Exception as e:
        print(f"  [Note] Task 4 chunker loader fallback: {e}")

    # 2. Fallback đọc trực tiếp file markdown nếu Task 4 chưa sẵn sàng
    if STANDARDIZED_DIR.exists():
        md_files = list(STANDARDIZED_DIR.rglob("*.md"))
        for md_file in md_files:
            try:
                text = md_file.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                
                doc_type = "legal" if "legal" in str(md_file.parent) else "news"
                
                # Parse title và url từ header
                title = md_file.stem.replace("-", " ").title()
                url = ""
                for line in text.split("\n")[:10]:
                    if line.startswith("# "):
                        title = line[2:].strip()
                    elif line.startswith("**Source:**"):
                        url = line.replace("**Source:**", "").strip()

                # Tách đoạn văn bản (paragraphs)
                paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                if not paragraphs:
                    paragraphs = [text]
                
                for idx, p in enumerate(paragraphs):
                    corpus.append({
                        "content": p,
                        "metadata": {
                            "source": md_file.name,
                            "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                            "type": doc_type,
                            "title": title,
                            "url": url,
                            "chunk_index": idx
                        }
                    })
            except Exception as e:
                print(f"Lỗi đọc file {md_file}: {e}")

    if not corpus:
        corpus = DEFAULT_CORPUS
    return corpus


CORPUS: List[Dict[str, Any]] = load_corpus()
_BM25_INDEX = None
_TFIDF_INDEX = None


import re

def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản và loại bỏ dấu câu."""
    return re.findall(r'\w+', text.lower())


def get_searchable_text(doc: Dict[str, Any]) -> str:
    """Ghép metadata nguồn, tiêu đề và nội dung để BM25 index trọn vẹn."""
    meta = doc.get("metadata", {})
    source = meta.get("source", "")
    title = meta.get("title", "")
    content = doc.get("content", "")
    return f"{title} {source} {content}"


def build_bm25_index(corpus: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(get_searchable_text(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def build_tfidf_index(corpus: List[Dict[str, Any]]):
    """Xây dựng TF-IDF index để so sánh với BM25 (bonus Task 6)."""
    texts = [get_searchable_text(doc) for doc in corpus]
    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def lexical_search(
    query: str, top_k: int = 10, method: str = "bm25"
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) hoặc "tfidf" (bonus để so sánh).

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global CORPUS, _BM25_INDEX, _TFIDF_INDEX

    if not CORPUS:
        CORPUS = load_corpus()

    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    method = method.lower()
    if method == "bm25":
        if _BM25_INDEX is None:
            _BM25_INDEX = build_bm25_index(CORPUS)
        scores = _BM25_INDEX.get_scores(tokenize(query))
    elif method == "tfidf":
        if _TFIDF_INDEX is None:
            _TFIDF_INDEX = build_tfidf_index(CORPUS)
        vectorizer, matrix = _TFIDF_INDEX
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()
    else:
        raise ValueError("method phải là 'bm25' hoặc 'tfidf'")

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": CORPUS[idx]["content"],
            "score": float(scores[idx]),
            "metadata": CORPUS[idx].get("metadata", {})
        })
    
    # Sort results descending by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    query = "phương thức thanh toán shopee"
    for method in ("bm25", "tfidf"):
        results = lexical_search(query, top_k=3, method=method)
        print(f"=== {method.upper()} | '{query}' ===")
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "Unknown")
            print(f"[{i}] {r['score']:.4f} | {source}")

