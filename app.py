"""
RAG Chatbot — E-commerce Support (Streamlit)
Kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Bonus features:
    - Hiển thị retrieval_source (Hybrid Search vs PageIndex Fallback)
    - Highlight màu theo score liên quan (xanh/vàng/đỏ)
    - Conversation memory (multi-turn) — nhớ 4 lượt hội thoại gần nhất

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_HISTORY_TURNS = 4  # số lượt hội thoại gần nhất đưa vào LLM (bonus memory)

# =============================================================================
# HELPERS
# =============================================================================


def score_badge(score: float) -> str:
    """Trả về icon màu tương ứng với mức độ liên quan của score."""
    if score >= 0.7:
        return "🟢"
    if score >= 0.4:
        return "🟡"
    return "🔴"


def retrieval_source_badge(source: str) -> str:
    """Hiển thị nhãn trực quan cho nguồn truy xuất (hybrid / pageindex / none)."""
    return {
        "hybrid": "🔍 **Hybrid Search** (Semantic + BM25 + RRF)",
        "pageindex": "📖 **PageIndex Fallback** (Vectorless — kích hoạt do hybrid search điểm thấp)",
        "none": "⚠️ **Không tìm thấy nguồn phù hợp**",
    }.get(source, f"`{source}`")


def render_sources(sources: list[dict]):
    """Render danh sách nguồn tham khảo kèm score highlight."""
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            source_name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0) or 0
            st.markdown(
                f"{score_badge(score)} **[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}`"
            )
            st.text(src.get("content", "")[:300] + "...")
            st.divider()


# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce Support RAG")
    st.caption(
        "Trợ lý hỏi đáp về chính sách thương mại điện tử và hỗ trợ khách hàng "
        "(đổi trả, thanh toán, bảo mật, người bán)"
    )

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    use_memory = st.toggle(
        "🧠 Ghi nhớ hội thoại (multi-turn)",
        value=True,
        help=f"Đưa {MAX_HISTORY_TURNS} lượt hội thoại gần nhất vào prompt để trả lời follow-up chính xác hơn.",
    )

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption(
        "Hybrid Retrieval (Semantic + TF-IDF) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation"
    )

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🛒 E-commerce Support RAG Chatbot")
st.caption("Hệ thống hỏi đáp chính sách e-commerce và trợ giúp khách hàng")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("retrieval_source"):
                st.caption(retrieval_source_badge(msg["retrieval_source"]))
            if msg.get("sources"):
                render_sources(msg["sources"])

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Chuẩn bị lịch sử hội thoại cho bonus conversation memory
    # (chỉ lấy role/content thô, bỏ metadata sources để không phình prompt)
    chat_history = None
    if use_memory:
        past_turns = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]  # bỏ câu hỏi vừa thêm ở trên
        ]
        chat_history = past_turns[-MAX_HISTORY_TURNS:] if past_turns else None

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            retrieval_source = None
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    query, top_k=top_k, chat_history=chat_history
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source")

            except NotImplementedError:
                answer = (
                    "⚠️ **Pipeline chưa sẵn sàng.** Task 9 (retrieval) hoặc Task 10 "
                    "(generation) chưa được implement đầy đủ — hãy đợi các Role liên quan "
                    "hoàn thành rồi thử lại."
                )
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if retrieval_source:
                st.caption(retrieval_source_badge(retrieval_source))
            if sources:
                render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )
