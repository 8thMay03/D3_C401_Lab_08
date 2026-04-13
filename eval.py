"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
"""

import json
import csv
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_answer import rag_answer, call_llm

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Cấu hình baseline (Sprint 2)
BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "use_parent_expansion": False,
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — hybrid + rerank)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "use_parent_expansion": True,
    "label": "variant_hybrid_rerank_parent",
}


# =============================================================================
# HELPER: Extract JSON từ LLM response
# =============================================================================

def _extract_json_from_response(text: str) -> Optional[Dict]:
    """Extract JSON object từ LLM response, handling markdown code blocks."""
    # Thử tìm JSON trong code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Thử tìm JSON trực tiếp
    match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    Dùng LLM-as-Judge để chấm điểm 1-5.
    """
    if answer in ("PIPELINE_NOT_IMPLEMENTED", "") or not chunks_used:
        return {"score": None, "notes": "No answer or no chunks available"}

    context = "\n---\n".join([c.get("text", "") for c in chunks_used[:5]])

    prompt = f"""You are an evaluation judge. Rate the faithfulness of the answer to the retrieved context.

Faithfulness measures: Does the answer ONLY contain information from the provided context? Does it make up any information?

Scoring (1-5):
5: Every claim in the answer is directly supported by the context
4: Almost fully grounded, one minor uncertain detail
3: Mostly grounded, some information may come from model knowledge
2: Many claims not found in the context
1: Answer is not grounded, mostly fabricated

Retrieved Context:
{context}

Answer to evaluate:
{answer}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

    try:
        result = call_llm(prompt)
        data = _extract_json_from_response(result)
        if data and "score" in data:
            return {
                "score": int(data["score"]),
                "notes": data.get("reason", ""),
            }
    except Exception as e:
        print(f"  [score_faithfulness] LLM judge error: {e}")

    return {"score": None, "notes": "LLM judge failed"}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    Dùng LLM-as-Judge để chấm điểm 1-5.
    """
    if answer in ("PIPELINE_NOT_IMPLEMENTED", ""):
        return {"score": None, "notes": "No answer available"}

    prompt = f"""You are an evaluation judge. Rate the relevance of the answer to the question.

Answer Relevance measures: Does the answer directly address the user's question? Is it on-topic?

Scoring (1-5):
5: Answer directly and fully addresses the question
4: Answers correctly but misses minor details
3: Related but not quite on-target
2: Partially off-topic
1: Does not answer the question at all

Question: {query}

Answer: {answer}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

    try:
        result = call_llm(prompt)
        data = _extract_json_from_response(result)
        if data and "score" in data:
            return {
                "score": int(data["score"]),
                "notes": data.get("reason", ""),
            }
    except Exception as e:
        print(f"  [score_answer_relevance] LLM judge error: {e}")

    return {"score": None, "notes": "LLM judge failed"}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Tính bằng: (số expected source được retrieve) / (tổng số expected sources)
    """
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources (abstain test)"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        # Kiểm tra partial match (tên file)
        expected_name = expected.split("/")[-1].replace(".pdf", "").replace(".md", "")
        matched = any(expected_name.lower() in r.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0

    return {
        "score": round(recall * 5),  # Convert to 1-5 scale
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources" +
                 (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có bao phủ đủ thông tin so với expected_answer không?
    Dùng LLM-as-Judge để chấm điểm 1-5.
    """
    if answer in ("PIPELINE_NOT_IMPLEMENTED", ""):
        return {"score": None, "notes": "No answer available"}

    if not expected_answer:
        return {"score": None, "notes": "No expected answer to compare"}

    prompt = f"""You are an evaluation judge. Rate the completeness of the model answer compared to the expected answer.

Completeness measures: Does the answer cover all key points from the expected answer?

Scoring (1-5):
5: All key points from expected answer are covered
4: Missing 1 minor detail
3: Missing some important information
2: Missing many important details
1: Missing most core content

Question: {query}

Expected Answer: {expected_answer}

Model Answer: {answer}

Output ONLY a JSON object: {{"score": <int 1-5>, "reason": "<brief explanation>"}}"""

    try:
        result = call_llm(prompt)
        data = _extract_json_from_response(result)
        if data and "score" in data:
            return {
                "score": int(data["score"]),
                "notes": data.get("reason", ""),
            }
    except Exception as e:
        print(f"  [score_completeness] LLM judge error: {e}")

    return {"score": None, "notes": "LLM judge failed"}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm điểm.
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'='*70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print('='*70)

    for q in test_questions:
        question_id = q["id"]
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi pipeline ---
        try:
            result = rag_answer(
                query=query,
                retrieval_mode=config.get("retrieval_mode", "dense"),
                top_k_search=config.get("top_k_search", 10),
                top_k_select=config.get("top_k_select", 3),
                use_rerank=config.get("use_rerank", False),
                use_parent_expansion=config.get("use_parent_expansion", False),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm (với delay để tránh rate limit) ---
        time.sleep(3)
        faith = score_faithfulness(answer, chunks_used)
        time.sleep(3)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        time.sleep(3)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": complete["score"],
            "completeness_notes": complete["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer[:120]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    # Tính averages (bỏ qua None)
    print(f"\n{'─'*50}")
    print(f"Summary for: {label}")
    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = sum(scores) / len(scores) if scores else None
        print(f"  Average {metric}: {avg:.2f}/5" if avg else f"  Average {metric}: N/A")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo từng câu hỏi và tổng thể.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print('='*70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r[metric] is not None]
        v_scores = [r[metric] for r in variant_results if r[metric] is not None]

        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg and v_avg) else None

        b_str = f"{b_avg:.2f}" if b_avg else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg else "N/A"
        d_str = f"{delta:+.2f}" if delta else "N/A"

        print(f"{metric:<20} {b_str:>10} {v_str:>10} {d_str:>8}")

    # Per-question comparison
    print(f"\n{'Câu':<6} {'Baseline F/R/Rc/C':<22} {'Variant F/R/Rc/C':<22} {'Better?':<10}")
    print("-" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_scores_str = "/".join([
            str(b_row.get(m, "?")) for m in metrics
        ])
        v_scores_str = "/".join([
            str(v_row.get(m, "?")) for m in metrics
        ])

        # So sánh
        b_total = sum(b_row.get(m, 0) or 0 for m in metrics)
        v_total = sum(v_row.get(m, 0) or 0 for m in metrics)
        better = "Variant" if v_total > b_total else ("Baseline" if b_total > v_total else "Tie")

        print(f"{qid:<6} {b_scores_str:<22} {v_scores_str:<22} {better:<10}")

    # Export to CSV
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu vào: {csv_path}")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """
    Tạo báo cáo tóm tắt scorecard dạng markdown.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# Scorecard: {label}
Generated: {timestamp}

## Summary

| Metric | Average Score |
|--------|--------------|
"""
    for metric, avg in averages.items():
        avg_str = f"{avg:.2f}/5" if avg else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += "\n## Per-Question Results\n\n"
    md += "| ID | Category | Faithful | Relevant | Recall | Complete | Notes |\n"
    md += "|----|----------|----------|----------|--------|----------|-------|\n"

    for r in results:
        md += (f"| {r['id']} | {r['category']} | {r.get('faithfulness', 'N/A')} | "
               f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
               f"{r.get('completeness', 'N/A')} | {r.get('faithfulness_notes', '')[:50]} |\n")

    return md


# =============================================================================
# MAIN — Chạy evaluation
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # Kiểm tra test questions
    print(f"\nLoading test questions từ: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi")

        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    # --- Chạy Baseline ---
    print("\n--- Chạy Baseline ---")
    baseline_results = run_scorecard(
        config=BASELINE_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )

    # Save scorecard baseline
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
    scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
    scorecard_path.write_text(baseline_md, encoding="utf-8")
    print(f"\nScorecard lưu tại: {scorecard_path}")

    # --- Chạy Variant ---
    print("\n--- Chạy Variant (hybrid + rerank) ---")
    variant_results = run_scorecard(
        config=VARIANT_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )

    # Save scorecard variant
    variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
    (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
    print(f"Scorecard variant lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")

    # --- A/B Comparison ---
    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv"
        )

    print("\n✅ Sprint 4 hoàn thành!")
