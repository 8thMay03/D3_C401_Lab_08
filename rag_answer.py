"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import json
import os
import re
from typing import List, Dict, Any

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from index import CHROMA_DB_DIR, get_embedding

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

_cross_encoder = None  # lazy: sentence_transformers.CrossEncoder


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Args:
        query: Câu hỏi của người dùng
        top_k: Số chunk tối đa trả về

    Returns:
        List các dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata (source, section, effective_date, ...)
          - "score": cosine similarity (1 - distance khi space=cosine trong Chroma)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    if not documents or not documents[0]:
        return []

    docs_row = documents[0]
    metas_row = metadatas[0] if metadatas else []
    dists_row = distances[0] if distances else []

    chunks: List[Dict[str, Any]] = []
    for i, text in enumerate(docs_row):
        if text is None:
            continue
        meta = metas_row[i] if i < len(metas_row) and metas_row[i] is not None else {}
        dist = dists_row[i] if i < len(dists_row) else None
        # Với metric cosine: distance ≈ 1 - cosine_similarity → score = 1 - distance
        score = (1.0 - float(dist)) if dist is not None else 0.0
        chunks.append(
            {
                "text": text,
                "metadata": meta,
                "score": score,
            }
        )
    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# Dùng cho Sprint 3 Variant hoặc kết hợp Hybrid
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa

    Returns:
        Giống retrieve_dense: mỗi phần tử có "text", "metadata", "score" (điểm BM25 thô).
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")
    raw = collection.get(include=["documents", "metadatas"])

    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    corpus_texts: List[str] = []
    corpus_metas: List[Dict[str, Any]] = []
    for i, text in enumerate(documents):
        if text is None:
            continue
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
        corpus_texts.append(text)
        corpus_metas.append(meta)

    if not corpus_texts:
        return []

    tokenized_corpus: List[List[str]] = []
    for doc in corpus_texts:
        tokens = doc.lower().split()
        if not tokens:
            tokens = ["_"]
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    n = len(scores)
    top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks: List[Dict[str, Any]] = []
    for idx in top_indices:
        chunks.append(
            {
                "text": corpus_texts[idx],
                "metadata": corpus_metas[idx],
                "score": float(scores[idx]),
            }
        )
    return chunks


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    Phù hợp khi: corpus lẫn lộn ngôn ngữ tự nhiên và tên riêng/mã lỗi/điều khoản

    Args:
        dense_weight: Trọng số nhánh dense trong RRF
        sparse_weight: Trọng số nhánh sparse trong RRF

    Công thức (rank bắt đầu từ 1):
        RRF(doc) = dense_weight * 1/(k + dense_rank) + sparse_weight * 1/(k + sparse_rank)
        với k = 60 (hằng số Cormack et al. / Elasticsearch RRF).

    Khi nào dùng hybrid (từ slide):
    - Corpus có cả câu tự nhiên VÀ tên riêng, mã lỗi, điều khoản
    - Query như "Approval Matrix" khi doc đổi tên thành "Access Control SOP"
    """
    RRF_K = 60
    # Lấy nhiều candidate mỗi nhánh hơn top_k cuối để RRF có đủ chồng lấp xếp hạng
    n_fetch = max(top_k * 2, 20)

    dense_results = retrieve_dense(query, top_k=n_fetch)
    sparse_results = retrieve_sparse(query, top_k=n_fetch)

    def doc_key(chunk: Dict[str, Any]) -> str:
        return chunk.get("text") or ""

    dense_rank: Dict[str, int] = {}
    for rank, ch in enumerate(dense_results, start=1):
        k = doc_key(ch)
        if k and k not in dense_rank:
            dense_rank[k] = rank

    sparse_rank: Dict[str, int] = {}
    for rank, ch in enumerate(sparse_results, start=1):
        k = doc_key(ch)
        if k and k not in sparse_rank:
            sparse_rank[k] = rank

    key_to_chunk: Dict[str, Dict[str, Any]] = {}
    for ch in dense_results + sparse_results:
        k = doc_key(ch)
        if not k:
            continue
        if k not in key_to_chunk:
            key_to_chunk[k] = ch

    if not key_to_chunk:
        return []

    rrf_scores: Dict[str, float] = {}
    for k in key_to_chunk:
        s = 0.0
        if k in dense_rank:
            s += dense_weight * (1.0 / (RRF_K + dense_rank[k]))
        if k in sparse_rank:
            s += sparse_weight * (1.0 / (RRF_K + sparse_rank[k]))
        rrf_scores[k] = s

    ordered_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
        :top_k
    ]

    merged: List[Dict[str, Any]] = []
    for key in ordered_keys:
        row = dict(key_to_chunk[key])
        row["score"] = rrf_scores[key]
        merged.append(row)
    return merged


def _retrieve_for_query(
    q: str,
    retrieval_mode: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if retrieval_mode == "dense":
        return retrieve_dense(q, top_k=top_k)
    if retrieval_mode == "sparse":
        return retrieve_sparse(q, top_k=top_k)
    if retrieval_mode == "hybrid":
        return retrieve_hybrid(q, top_k=top_k)
    raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")


def _merge_retrieval_chunks(
    chunk_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Gộp nhiều lần retrieve (multi-query): giữ chunk có score cao nhất theo nội dung text."""
    best: Dict[str, Dict[str, Any]] = {}
    for lst in chunk_lists:
        for ch in lst:
            t = ch.get("text") or ""
            if not t:
                continue
            sc = float(ch.get("score", 0.0))
            if t not in best or sc > float(best[t].get("score", 0.0)):
                best[t] = dict(ch)
    return sorted(
        best.values(),
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )


def _parse_json_array_from_llm(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    m = re.search(r"\[[\s\S]*?\]", text)
    fragment = m.group(0) if m else text
    try:
        data = json.loads(fragment)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return []


# =============================================================================
# RERANK (Sprint 3 alternative)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder (query–passage relevance).

    Cross-encoder: chấm lại "chunk nào thực sự trả lời câu hỏi này?"
    MMR (Maximal Marginal Relevance) có thể bổ sung sau nếu cần giảm trùng lặp.

    Funnel logic (từ slide):
      Search rộng (top-20) → Rerank (top-6) → Select (top-3)

    Khi nào dùng rerank:
    - Dense/hybrid trả về nhiều chunk nhưng có noise
    - Muốn chắc chắn chỉ 3-5 chunk tốt nhất vào prompt

    Model mặc định: CROSS_ENCODER_MODEL (cross-encoder/ms-marco-MiniLM-L-6-v2).
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [[query, c.get("text") or ""] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for chunk, score in ranked[:top_k]:
        row = dict(chunk)
        row["score"] = float(score)
        out.append(row)
    return out


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.

    Format: structured snippets với source, section, score (từ slide).
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        # TODO: Tùy chỉnh format nếu muốn (thêm effective_date, department, ...)
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Xây dựng grounded prompt theo 4 quy tắc từ slide:
    1. Evidence-only: Chỉ trả lời từ retrieved context
    2. Abstain: Thiếu context thì nói không đủ dữ liệu
    3. Citation: Gắn source/section khi có thể
    4. Short, clear, stable: Output ngắn, rõ, nhất quán

    TODO Sprint 2:
    Đây là prompt baseline. Trong Sprint 3, bạn có thể:
    - Thêm hướng dẫn về format output (JSON, bullet points)
    - Thêm ngôn ngữ phản hồi (tiếng Việt vs tiếng Anh)
    - Điều chỉnh tone phù hợp với use case (CS helpdesk, IT support)
    """
    prompt = f"""Answer only from the retrieved context below.
If the context is insufficient to answer the question, say you do not know and do not make up information.
Cite the source field (in brackets like [1]) when possible.
Keep your answer short, clear, and factual.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    Ưu tiên OPENAI_API_KEY + LLM_MODEL; nếu không có thì GOOGLE_API_KEY + GEMINI_MODEL.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        msg = response.choices[0].message.content
        return (msg or "").strip()

    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        import google.generativeai as genai

        genai.configure(api_key=google_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": max_tokens,
            },
        )
        text = getattr(response, "text", None) or ""
        if not text and response.candidates:
            parts = response.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        return text.strip()

    raise RuntimeError(
        "Thiếu API key: đặt OPENAI_API_KEY hoặc GOOGLE_API_KEY trong môi trường / .env"
    )


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3 alternative)
# =============================================================================


def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall (cần OPENAI_API_KEY hoặc GOOGLE_API_KEY).

    Strategies:
      - "expansion": Thêm diễn đạt / từ liên quan → JSON array
      - "decomposition": Tách thành 2-4 sub-query → JSON array
      - "hyde": Sinh đoạn văn giả định trả lời (một chuỗi) để dense search
    """
    if strategy not in ("expansion", "decomposition", "hyde"):
        return [query]

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        return [query]

    if strategy == "hyde":
        prompt = f"""Write a short hypothetical factual passage (3-5 sentences) that could answer the question.
Use the same language as the question. Output only the passage, no title or bullet points.

Question: {query}
"""
        raw = call_llm(prompt, max_tokens=384)
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text).strip()
        return [text if text else query]

    if strategy == "expansion":
        prompt = f"""Generate 2-4 alternative search phrasings or related terms for retrieval (same language as the question).
Output ONLY a JSON array of strings. No markdown, no explanation.

Question: {query}
"""
    else:  # decomposition
        prompt = f"""Break this question into 2-4 simpler sub-questions for document search (same language as the question).
Output ONLY a JSON array of strings. No markdown, no explanation.

Question: {query}
"""

    raw = call_llm(prompt, max_tokens=256)
    variants = _parse_json_array_from_llm(raw)
    seen = set()
    out: List[str] = []
    for s in [query] + variants:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:6]


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    use_query_transform: bool = False,
    transform_strategy: str = "expansion",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → (transform) → retrieve → (rerank) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        use_query_transform: Gọi transform_query() rồi retrieve từng biến thể (Sprint 3)
        transform_strategy: "expansion" | "decomposition" | "hyde"
        verbose: In thêm thông tin debug

    Returns:
        Dict với answer, sources, chunks_used, query, config
    """
    config: Dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "use_query_transform": use_query_transform,
        "transform_strategy": transform_strategy if use_query_transform else None,
    }

    # --- Bước 1: Retrieve ---
    if use_query_transform:
        sub_queries = transform_query(query, strategy=transform_strategy)
        if verbose:
            print(f"[RAG] Sub-queries ({transform_strategy}): {sub_queries}")
        chunk_lists = [
            _retrieve_for_query(sq, retrieval_mode, top_k_search) for sq in sub_queries
        ]
        candidates = _merge_retrieval_chunks(chunk_lists)[:top_k_search]
    else:
        candidates = _retrieve_for_query(query, retrieval_mode, top_k_search)
    for candidate in candidates:
        print(candidate["text"])
        print(candidate["score"])
        print("-" * 100)

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            meta = c.get("metadata") or {}
            print(
                f"  [{i+1}] score={c.get('score', 0):.3f} | {meta.get('source', '?')}"
            )

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    if verbose:
        print(f"[RAG] After select: {len(candidates)} chunks")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list(
        {
            (c.get("metadata") or {}).get("source", "unknown")
            for c in candidates
        }
    )

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(query: str) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    TODO Sprint 3:
    Chạy hàm này để thấy sự khác biệt giữa dense, sparse, hybrid.
    Dùng để justify tại sao chọn variant đó cho Sprint 3.

    A/B Rule (từ slide): Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = ["dense", "sparse", "hybrid"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        try:
            result = rag_answer(query, retrieval_mode=strategy, verbose=False)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    # Test queries từ data/test_questions.json
    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì?",  # Query không có trong docs → kiểm tra abstain
    ]

    print("\n--- Sprint 2: Test Baseline (Dense) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    # print("\n--- Sprint 3: So sánh strategies ---")
    # compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")

    print("\nGợi ý: đặt OPENAI_API_KEY hoặc GOOGLE_API_KEY; index Chroma (rag_lab) cần đã build.")
