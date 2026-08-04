# RAG Evaluation Results

## Thiết lập

- Framework: RAGAS
- Số câu hỏi: 18
- Config A: A — Hybrid (Semantic + TF-IDF + RRF)
- Config B: B — Dense-only

## Overall Scores

| Metric | Config A | Config B | Δ (A - B) |
|---|---:|---:|---:|
| Faithfulness | 0.8287 | 0.6852 | +0.1435 |
| Answer Relevance | 0.4807 | 0.3428 | +0.1379 |
| Context Recall | 0.8889 | 0.6111 | +0.2778 |
| Context Precision | 0.8949 | 0.6813 | +0.2136 |
| **Average** | **0.7733** | **0.5801** | **+0.1932** |

## A/B Comparison Analysis

Config A kết hợp dense retrieval, TF-IDF và RRF; Config B chỉ dùng dense retrieval. Dùng chênh lệch bảng điểm để giải thích trade-off giữa recall của hybrid search và precision của dense-only.

## Worst Performers — Config A

| # | Question | Faithfulness | Relevance | Recall | Precision |
|---:|---|---:|---:|---:|---:|
| 1 | Đơn hàng xuyên biên giới trên Shopee thường giao trong bao lâu? | 0.0000 | 0.0000 | 0.0000 | 0.8875 |
| 2 | Điều kiện tối thiểu để được trả góp 0% qua thẻ tín dụng trên Shopee là gì? | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 3 | Khi yêu cầu trả hàng vì thiếu/sai sản phẩm, cần cung cấp bằng chứng gì? | 1.0000 | 0.3818 | 0.0000 | 0.9167 |

## Recommendations

1. Rà soát các câu ở Bottom 3 và bổ sung/chỉnh metadata hoặc chunk tương ứng.
2. Calibrate `SCORE_THRESHOLD` bằng tập câu hỏi liên quan và ngoài domain.
3. So sánh thêm BM25 với TF-IDF nếu Context Precision chưa đạt kỳ vọng.
