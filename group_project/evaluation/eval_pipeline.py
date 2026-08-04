"""RAGAS evaluation and A/B comparison for the e-commerce RAG chatbot.

Run a cheap structural check first:
    python -m group_project.evaluation.eval_pipeline --dry-run

Run the real LLM-based evaluation (costs API requests):
    python -m group_project.evaluation.eval_pipeline --run
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

load_dotenv(PROJECT_ROOT / ".env")


def load_golden_dataset() -> list[dict[str, str]]:
    """Load and validate the golden dataset."""
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 15:
        raise ValueError("golden_dataset.json phải có tối thiểu 15 câu hỏi.")
    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, 1):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Test case {index} thiếu trường: {', '.join(sorted(missing))}")
    return data


def _dense_only_retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Config B: only dense retrieval, without TF-IDF/RRF/PageIndex."""
    from src.task5_semantic_search import semantic_search

    results = semantic_search(query, top_k=top_k)
    for item in results:
        item["source"] = "dense_only"
    return results


def _answer_from_chunks(query: str, chunks: list[dict[str, Any]]) -> str:
    """Generate an answer using the same prompt/provider path as Task 10."""
    from src.task10_generation import (
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
        _get_llm_client,
        format_context,
        reorder_for_llm,
    )

    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    client, model = _get_llm_client()
    context = format_context(reorder_for_llm(chunks))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (response.choices[0].message.content or "").strip()


def _build_eval_rows(
    retrieve_fn: Callable[[str, int], list[dict[str, Any]]],
    golden_dataset: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Run one retrieval configuration and build RAGAS-compatible rows."""
    rows = []
    for index, item in enumerate(golden_dataset, 1):
        print(f"[{index}/{len(golden_dataset)}] {item['question']}")
        chunks = retrieve_fn(item["question"], 5)
        answer = _answer_from_chunks(item["question"], chunks)
        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": [chunk.get("content", "") for chunk in chunks],
                "ground_truth": item["expected_answer"],
                "expected_context": item["expected_context"],
            }
        )
    return rows


def _ragas_clients():
    """Create the LLM and embedding clients used by RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    if openrouter_key:
        base_url = "https://openrouter.ai/api/v1"
        return (
            ChatOpenAI(model=model, api_key=openrouter_key, base_url=base_url, temperature=0),
            OpenAIEmbeddings(
                model="openai/text-embedding-3-small",
                api_key=openrouter_key,
                base_url=base_url,
            ),
        )
    if openai_key:
        return (
            ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0),
            OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key),
        )
    raise RuntimeError("RAGAS cần OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env.")


def evaluate_with_ragas(rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate generated answers with four required RAGAS metrics."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    llm, embeddings = _ragas_clients()
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )
    frame = result.to_pandas()
    metric_columns = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    scores = {column: float(frame[column].mean()) for column in metric_columns}
    per_question = []
    for row, (_, metrics) in zip(rows, frame.iterrows()):
        per_question.append({**row, **{key: float(metrics[key]) for key in metric_columns}})
    return scores, per_question


def compare_configs(golden_dataset: list[dict[str, str]]) -> dict[str, Any]:
    """Compare hybrid retrieval (A) with dense-only retrieval (B)."""
    from src.task9_retrieval_pipeline import retrieve

    configs = {
        "A — Hybrid (Semantic + TF-IDF + RRF)": retrieve,
        "B — Dense-only": _dense_only_retrieve,
    }
    comparison: dict[str, Any] = {}
    for name, retrieve_fn in configs.items():
        print(f"\n=== Evaluating {name} ===")
        rows = _build_eval_rows(retrieve_fn, golden_dataset)
        scores, per_question = evaluate_with_ragas(rows)
        comparison[name] = {"scores": scores, "per_question": per_question}
    return comparison


def export_results(comparison: dict[str, Any], case_count: int) -> None:
    """Export the RAGAS A/B report to results.md."""
    config_a, config_b = comparison.values()
    labels = list(comparison.keys())
    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevancy"),
        ("Context Recall", "context_recall"),
        ("Context Precision", "context_precision"),
    ]
    lines = [
        "# RAG Evaluation Results",
        "",
        "## Thiết lập",
        "",
        f"- Framework: RAGAS",
        f"- Số câu hỏi: {case_count}",
        f"- Config A: {labels[0]}",
        f"- Config B: {labels[1]}",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A | Config B | Δ (A - B) |",
        "|---|---:|---:|---:|",
    ]
    averages_a, averages_b = [], []
    for label, key in metrics:
        score_a = config_a["scores"][key]
        score_b = config_b["scores"][key]
        averages_a.append(score_a)
        averages_b.append(score_b)
        lines.append(f"| {label} | {score_a:.4f} | {score_b:.4f} | {score_a - score_b:+.4f} |")
    lines.append(
        f"| **Average** | **{sum(averages_a) / len(averages_a):.4f}** | "
        f"**{sum(averages_b) / len(averages_b):.4f}** | "
        f"**{sum(averages_a) / len(averages_a) - sum(averages_b) / len(averages_b):+.4f}** |"
    )

    worst = sorted(
        config_a["per_question"],
        key=lambda row: sum(row[key] for _, key in metrics) / len(metrics),
    )[:3]
    lines.extend(
        [
            "",
            "## A/B Comparison Analysis",
            "",
            "Config A kết hợp dense retrieval, TF-IDF và RRF; Config B chỉ dùng dense retrieval. "
            "Dùng chênh lệch bảng điểm để giải thích trade-off giữa recall của hybrid search và precision của dense-only.",
            "",
            "## Worst Performers — Config A",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(worst, 1):
        lines.append(
            f"| {index} | {row['question']} | {row['faithfulness']:.4f} | "
            f"{row['answer_relevancy']:.4f} | {row['context_recall']:.4f} | "
            f"{row['context_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "1. Rà soát các câu ở Bottom 3 và bổ sung/chỉnh metadata hoặc chunk tương ứng.",
            "2. Calibrate `SCORE_THRESHOLD` bằng tập câu hỏi liên quan và ngoài domain.",
            "3. So sánh thêm BM25 với TF-IDF nếu Context Precision chưa đạt kỳ vọng.",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run LLM/RAGAS evaluation and write results.md")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset only; this is the default")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    if args.limit:
        golden_dataset = golden_dataset[: args.limit]
    print(f"Loaded {len(golden_dataset)} valid golden test cases")
    if not args.run:
        print("Dry run passed. Use --run to call the LLM and RAGAS metrics.")
        return

    comparison = compare_configs(golden_dataset)
    export_results(comparison, len(golden_dataset))
    print(f"Saved report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
