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
  Đổi đồng thời chunking + hybrid + rerank + prompt = không biết biến nào có tác dụng.
"""

import csv
import difflib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_answer import call_llm, rag_answer

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
    "label": "baseline_dense",
}

# Cấu hình variant (Sprint 3 — điều chỉnh theo lựa chọn của nhóm)
VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",   # Hoặc "dense" nếu chỉ đổi rerank
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,           # Hoặc False nếu variant là hybrid không rerank
    "label": "variant_hybrid_rerank",
}


# =============================================================================
# SCORING FUNCTIONS
# 4 metrics từ slide: Faithfulness, Answer Relevance, Context Recall, Completeness
# =============================================================================


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    """Trích JSON object đầu tiên từ output LLM."""
    if not text or not text.strip():
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _clamp_int_score(v: Any, lo: int = 1, hi: int = 5) -> Optional[int]:
    if v is None:
        return None
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, x))


def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness: Câu trả lời có bám đúng chứng cứ đã retrieve không?
    Câu hỏi: Model có tự bịa thêm thông tin ngoài retrieved context không?

    Thang điểm 1-5 (LLM-as-judge nếu có API; không thì heuristic đơn giản).
    """
    if "ERROR:" in answer or "PIPELINE_NOT_IMPLEMENTED" in answer:
        return {
            "score": None,
            "notes": "Pipeline lỗi hoặc chưa chạy — không chấm faithfulness.",
        }

    context = "\n\n---\n\n".join(
        (c.get("text") or "")[:4000] for c in chunks_used[:8]
    )
    if not context.strip():
        context = "(Không có chunk retrieval — chỉ đánh giá abstain hợp lý.)"

    if _llm_available():
        prompt = f"""You evaluate RAG faithfulness. Rate 1-5 only.
5 = Every factual claim in the ANSWER is supported by the CONTEXT (or answer correctly abstains).
1 = Answer adds facts not in context or contradicts context.

CONTEXT:
{context}

ANSWER:
{answer}

Output ONLY valid JSON: {{"score": <int 1-5>, "notes": "<one short sentence>"}}"""
        try:
            raw = call_llm(prompt, max_tokens=256)
            data = _parse_judge_json(raw) or {}
            sc = _clamp_int_score(data.get("score"))
            notes = str(data.get("notes") or data.get("reason") or "").strip()
            if sc is not None:
                return {"score": sc, "notes": notes or "LLM judge"}
        except Exception:
            pass

    # Heuristic (không có API hoặc parse lỗi)
    al = answer.lower()
    abstain = any(
        x in al
        for x in (
            "không đủ",
            "không có thông tin",
            "cannot answer",
            "do not know",
            "don't know",
            "no information",
            "insufficient",
        )
    )
    if not chunks_used and abstain:
        return {"score": 4, "notes": "Heuristic: abstain with empty context"}
    if not chunks_used:
        return {"score": 2, "notes": "Heuristic: no context — cần LLM judge hoặc chấm tay"}
    return {"score": 3, "notes": "Heuristic default; set OPENAI_API_KEY for LLM-as-judge"}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance: Answer có trả lời đúng câu hỏi người dùng hỏi không?
    Câu hỏi: Model có bị lạc đề hay trả lời đúng vấn đề cốt lõi không?

    Thang điểm 1-5:
      5: Answer trả lời trực tiếp và đầy đủ câu hỏi
      4: Trả lời đúng nhưng thiếu vài chi tiết phụ
      3: Trả lời có liên quan nhưng chưa đúng trọng tâm
      2: Trả lời lạc đề một phần
      1: Không trả lời câu hỏi

    """
    if "ERROR:" in answer or "PIPELINE_NOT_IMPLEMENTED" in answer:
        return {"score": None, "notes": "Pipeline lỗi — bỏ qua relevance."}

    if _llm_available():
        prompt = f"""Rate whether the ANSWER addresses the USER QUESTION (1-5).
5 = Direct, on-topic answer.
1 = Off-topic or does not answer the question.

QUESTION:
{query}

ANSWER:
{answer}

Output ONLY JSON: {{"score": <int 1-5>, "notes": "<short>"}}"""
        try:
            raw = call_llm(prompt, max_tokens=256)
            data = _parse_judge_json(raw) or {}
            sc = _clamp_int_score(data.get("score"))
            notes = str(data.get("notes") or "").strip()
            if sc is not None:
                return {"score": sc, "notes": notes or "LLM judge"}
        except Exception:
            pass

    q_tokens = set(re.findall(r"\w+", query.lower()))
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    if not q_tokens:
        return {"score": 3, "notes": "Heuristic: empty query tokens"}
    overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    sc = max(1, min(5, 1 + int(overlap * 4)))
    return {"score": sc, "notes": f"Heuristic token overlap ~{overlap:.2f}"}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall: Retriever có mang về đủ evidence cần thiết không?
    Câu hỏi: Expected source có nằm trong retrieved chunks không?

    Đây là metric đo retrieval quality, không phải generation quality.

    Cách tính đơn giản:
        recall = (số expected source được retrieve) / (tổng số expected sources)

    Ví dụ:
        expected_sources = ["policy/refund-v4.pdf", "sla-p1-2026.pdf"]
        retrieved_sources = ["policy/refund-v4.pdf", "helpdesk-faq.md"]
        recall = 1/2 = 0.5
    """
    if not expected_sources:
        # Câu hỏi không có expected source (ví dụ: "Không đủ dữ liệu" cases)
        return {"score": None, "recall": None, "notes": "No expected sources"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "") for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        expected_file = expected.split("/")[-1].lower()
        stem = expected_file.rsplit(".", 1)[0] if "." in expected_file else expected_file
        matched = any(
            expected_file in r.lower()
            or stem in r.lower()
            or (stem and stem in r.lower().replace("_", "-"))
            for r in retrieved_sources
        )
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources) if expected_sources else 0.0
    # Map recall [0,1] → điểm 1–5
    score_1_5 = max(1, min(5, int(round(recall * 4)) + 1))

    return {
        "score": score_1_5,
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved: {found}/{len(expected_sources)} expected sources"
        + (f". Missing: {missing}" if missing else ""),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness: Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?
    Câu hỏi: Answer có bao phủ đủ thông tin so với expected_answer không?

    Thang điểm 1-5:
      5: Answer bao gồm đủ tất cả điểm quan trọng trong expected_answer
      4: Thiếu 1 chi tiết nhỏ
      3: Thiếu một số thông tin quan trọng
      2: Thiếu nhiều thông tin quan trọng
      1: Thiếu phần lớn nội dung cốt lõi

    """
    if not (expected_answer or "").strip():
        return {"score": None, "notes": "No expected_answer in test set"}

    if "ERROR:" in answer or "PIPELINE_NOT_IMPLEMENTED" in answer:
        return {"score": None, "notes": "Pipeline lỗi — bỏ qua completeness."}

    if _llm_available():
        prompt = f"""Compare MODEL ANSWER to REFERENCE. Rate completeness 1-5.
5 = All key facts from reference appear in the model answer (wording may differ).
1 = Most key facts missing or wrong.

REFERENCE (expected):
{expected_answer}

MODEL ANSWER:
{answer}

Output ONLY JSON: {{"score": <int 1-5>, "notes": "<short>"}}"""
        try:
            raw = call_llm(prompt, max_tokens=256)
            data = _parse_judge_json(raw) or {}
            sc = _clamp_int_score(data.get("score"))
            notes = str(data.get("notes") or "").strip()
            if sc is not None:
                return {"score": sc, "notes": notes or "LLM judge"}
        except Exception:
            pass

    ratio = difflib.SequenceMatcher(
        None, answer.lower().strip(), expected_answer.lower().strip()
    ).ratio()
    sc = max(1, min(5, int(round(ratio * 4)) + 1))
    return {
        "score": sc,
        "notes": f"Heuristic sequence similarity ~{ratio:.2f}",
    }


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

    Args:
        config: Pipeline config (retrieval_mode, top_k, use_rerank, ...)
        test_questions: List câu hỏi (load từ JSON nếu None)
        verbose: In kết quả từng câu

    Returns:
        List scorecard results, mỗi item là một row

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
                use_query_transform=config.get("use_query_transform", False),
                transform_strategy=config.get("transform_strategy", "expansion"),
                verbose=False,
            )
            answer = result["answer"]
            chunks_used = result["chunks_used"]

        except Exception as e:
            answer = f"ERROR: {e}"
            chunks_used = []

        # --- Chấm điểm ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
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
            print(f"  Answer: {answer[:100]}...")
            print(f"  Faithful: {faith['score']} | Relevant: {relevance['score']} | "
                  f"Recall: {recall['score']} | Complete: {complete['score']}")

    for metric in ["faithfulness", "relevance", "context_recall", "completeness"]:
        scores = [r[metric] for r in results if r[metric] is not None]
        if not scores:
            print(f"\nAverage {metric}: N/A (chưa chấm)")
        else:
            avg = sum(scores) / len(scores)
            print(f"\nAverage {metric}: {avg:.2f}")

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
        if b_avg is not None and v_avg is not None:
            delta = v_avg - b_avg
            d_str = f"{delta:+.2f}"
        else:
            delta = None
            d_str = "N/A"

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"

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

        def _sum_scores(row: Dict) -> float:
            s = 0.0
            for m in metrics:
                x = row.get(m)
                if isinstance(x, (int, float)):
                    s += float(x)
            return s

        b_total = _sum_scores(b_row)
        v_total = _sum_scores(v_row)
        better = (
            "Variant"
            if v_total > b_total
            else ("Baseline" if b_total > v_total else "Tie")
        )

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
    """Tạo báo cáo tóm tắt scorecard dạng markdown."""
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
        note = str(r.get("faithfulness_notes", "") or "").replace("|", "/")[:80]
        md += (
            f"| {r['id']} | {r.get('category', '')} | {r.get('faithfulness', 'N/A')} | "
            f"{r.get('relevance', 'N/A')} | {r.get('context_recall', 'N/A')} | "
            f"{r.get('completeness', 'N/A')} | {note} |\n"
        )

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

        # In preview
        for q in test_questions[:3]:
            print(f"  [{q['id']}] {q['question']} ({q['category']})")
        print("  ...")

    except FileNotFoundError:
        print("Không tìm thấy file test_questions.json!")
        test_questions = []

    baseline_results: List[Dict[str, Any]] = []
    variant_results: List[Dict[str, Any]] = []

    print("\n--- Chạy Baseline ---")
    try:
        baseline_results = run_scorecard(
            config=BASELINE_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_md = generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"])
        scorecard_path = RESULTS_DIR / "scorecard_baseline.md"
        scorecard_path.write_text(baseline_md, encoding="utf-8")
        print(f"\nScorecard lưu tại: {scorecard_path}")
    except Exception as e:
        print(f"Lỗi baseline: {e}")
        baseline_results = []

    print("\n--- Chạy Variant ---")
    try:
        variant_results = run_scorecard(
            config=VARIANT_CONFIG,
            test_questions=test_questions,
            verbose=True,
        )
        variant_md = generate_scorecard_summary(
            variant_results, VARIANT_CONFIG["label"]
        )
        (RESULTS_DIR / "scorecard_variant.md").write_text(
            variant_md, encoding="utf-8"
        )
        print(f"\nScorecard variant lưu tại: {RESULTS_DIR / 'scorecard_variant.md'}")
    except Exception as e:
        print(f"Lỗi variant: {e}")
        variant_results = []

    if baseline_results and variant_results:
        compare_ab(
            baseline_results,
            variant_results,
            output_csv="ab_comparison.csv",
        )

    print("\nGợi ý: đặt API key để LLM-as-judge; không có key vẫn chạy heuristic.")
