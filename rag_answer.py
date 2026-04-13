"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Thêm rerank (cross-encoder)
  - Thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)
USE_PARENT_EXPANSION = True  # Tự động lấy parent chunk khi tìm thấy child

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Model caches
_rerank_model = None


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
          - "score": cosine similarity score
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            # distances trong ChromaDB cosine = 1 - similarity → Score = 1 - distance
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
            })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).
    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    """
    import chromadb
    from rank_bm25 import BM25Okapi
    from index import CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")
    all_data = collection.get(include=["documents", "metadatas"])

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    if not documents:
        return []

    # Tokenize
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top-k theo BM25 score
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    chunks = []
    for idx in top_indices:
        chunks.append({
            "text": documents[idx],
            "metadata": metadatas[idx],
            "score": float(scores[idx]),
        })

    return chunks


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.4,
    sparse_weight: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).
    Mạnh ở: giữ được cả nghĩa (dense) lẫn keyword chính xác (sparse)
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # RRF: score(doc) = weight * 1/(60 + rank)
    rrf_scores = {}  # key → {"chunk": chunk_data, "score": float}

    for rank, chunk in enumerate(dense_results):
        key = chunk["text"][:200]  # Dùng text prefix làm key
        if key not in rrf_scores:
            rrf_scores[key] = {"chunk": chunk, "score": 0.0}
        rrf_scores[key]["score"] += dense_weight * (1.0 / (60 + rank))

    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"][:200]
        if key not in rrf_scores:
            rrf_scores[key] = {"chunk": chunk, "score": 0.0}
        rrf_scores[key]["score"] += sparse_weight * (1.0 / (60 + rank))

    # Sort theo RRF score giảm dần
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        {**item["chunk"], "score": item["score"]}
        for item in sorted_results[:top_k]
    ]


# =============================================================================
# PARENT CONTEXT EXPANSION (cho hierarchical chunking)
# Khi tìm được child chunk → lấy thêm parent chunk để có ngữ cảnh rộng
# =============================================================================

def expand_with_parents(
    chunks: List[Dict[str, Any]],
    max_parents: int = 2,
) -> List[Dict[str, Any]]:
    """
    Mở rộng kết quả retrieval bằng cách thêm parent chunks.

    Khi tìm được child chunk (vd: chunk_id="policy_refund_v4_2.1"),
    tự động lấy parent chunk (parent_id="policy_refund_v4_2") để có
    ngữ cảnh đầy đủ hơn cho LLM.

    Flow: child found → lookup parent_id → fetch parent from ChromaDB → merge
    """
    import chromadb
    from index import CHROMA_DB_DIR

    # Tìm parent_ids cần fetch
    parent_ids_needed = set()
    existing_ids = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        existing_ids.add(meta.get("chunk_id", ""))
        parent_id = meta.get("parent_id", "")
        if parent_id and meta.get("chunk_type") == "child":
            parent_ids_needed.add(parent_id)

    # Bỏ parent đã có trong kết quả
    parent_ids_needed -= existing_ids

    if not parent_ids_needed:
        return chunks

    # Fetch parent chunks từ ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        collection = client.get_collection("rag_lab")

        parent_ids_list = list(parent_ids_needed)[:max_parents]
        parent_results = collection.get(
            ids=parent_ids_list,
            include=["documents", "metadatas"]
        )

        parent_chunks = []
        for doc, meta in zip(parent_results["documents"], parent_results["metadatas"]):
            parent_chunks.append({
                "text": doc,
                "metadata": meta,
                "score": 0.0,  # Parent không có score từ search
            })

        # Merge: children trước, parents sau (context bổ sung)
        return chunks + parent_chunks

    except Exception as e:
        print(f"[expand_with_parents] Lỗi: {e}")
        return chunks


# =============================================================================
# RERANK (Sprint 3)
# Cross-encoder để chấm lại relevance sau search rộng
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank các candidate chunks bằng cross-encoder.
    Funnel: Search rộng (top-10) → Rerank → Select (top-3)
    """
    global _rerank_model

    if not candidates:
        return []

    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        print("[Rerank] Loading cross-encoder model...")
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = _rerank_model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {**chunk, "score": float(score)}
        for chunk, score in ranked[:top_k]
    ]


# =============================================================================
# QUERY TRANSFORMATION (Sprint 3)
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.
    Strategies:
      - "expansion": Thêm từ đồng nghĩa, alias, tên cũ
      - "decomposition": Tách query phức tạp thành sub-queries
    """
    prompt = ""

    if strategy == "expansion":
        prompt = f"""Given the query: '{query}'
Generate 2-3 alternative phrasings or related terms in Vietnamese.
Output as a JSON array of strings only, no explanation.
Example: ["phrasing 1", "phrasing 2"]"""

    elif strategy == "decomposition":
        prompt = f"""Break down this complex query into 2-3 simpler sub-queries in Vietnamese: '{query}'
Output as a JSON array of strings only, no explanation.
Example: ["sub-query 1", "sub-query 2"]"""

    else:
        return [query]

    try:
        result = call_llm(prompt)
        # Parse JSON từ response
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            alternatives = json.loads(match.group(0))
            return [query] + [str(a) for a in alternatives]
    except Exception as e:
        print(f"[transform_query] Lỗi: {e}")

    return [query]


# =============================================================================
# GENERATION — GROUNDED ANSWER FUNCTION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói danh sách chunks thành context block để đưa vào prompt.
    Format: structured snippets với source, section, chunk_id, score.
    Mỗi chunk có số thứ tự [1], [2], ... để model dễ trích dẫn.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        chunk_id = meta.get("chunk_id", "")
        chunk_type = meta.get("chunk_type", "")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if chunk_id:
            header += f" | id={chunk_id}"
        if chunk_type:
            header += f" ({chunk_type})"
        if score > 0:
            header += f" | score={score:.2f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """Xây dựng prompt chuẩn và yêu cầu mô hình trả lời đầy đủ chi tiết, có kèm bắt buộc Abstain."""
    prompt = f"""Answer ONLY based on the retrieved context below.

CRITICAL RULES FOR ABSTAINING (GROUNDING):
- Nếu tài liệu (context) không chứa câu trả lời cho câu hỏi (hoàn toàn thiếu thông tin), bạn phải từ chối trả lời một cách lịch sự, ví dụ: "Xin lỗi, không có thông tin trong tài liệu cho câu hỏi này". Tuyệt đối không tự bịa thông tin.
- Nếu tài liệu chỉ nhắc đến một phần nhưng không đủ chi tiết để trả lời trọn vẹn, hãy trình bày những gì có trong tài liệu và nói rõ đoạn thông tin bị thiếu. Không lấy kiến thức từ bên ngoài để đắp vào.

Cite the source field (in brackets like [1]) when possible.
IMPORTANT: Provide a comprehensive and detailed answer that covers all aspects of the user's question found in the context. Do not leave out important steps, conditions, or details.
Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi OpenAI ChatGPT API để sinh câu trả lời.
    Dùng temperature=0 để output ổn định cho evaluation.
    Bao gồm cơ chế exponential backoff retry để chống Rate Limit (429).
    """
    import time
    from openai import OpenAI
    import openai

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    max_retries = 3
    base_delay = 5

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = base_delay * (2 ** attempt)
            print(f"\n[call_llm] Lỗi Rate limit (429). Đang đợi {wait_time}s để thử lại (lần {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise e
            print(f"\n[call_llm] Lỗi Server. Đang đợi {base_delay}s để thử lại (lần {attempt+1}/{max_retries})...")
            time.sleep(base_delay)
            
    return "ERROR: LLM call failed after retries"


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    use_parent_expansion: bool = USE_PARENT_EXPANSION,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → retrieve → (rerank) → (parent expand) → generate.

    Args:
        query: Câu hỏi
        retrieval_mode: "dense" | "sparse" | "hybrid"
        top_k_search: Số chunk lấy từ vector store (search rộng)
        top_k_select: Số chunk đưa vào prompt (sau rerank/select)
        use_rerank: Có dùng cross-encoder rerank không
        use_parent_expansion: Có lấy parent chunk khi tìm thấy child không
        verbose: In thêm thông tin debug

    Returns:
        Dict với:
          - "answer": câu trả lời grounded
          - "sources": list source names trích dẫn
          - "chunks_used": list chunks đã dùng
          - "query": query gốc
          - "config": cấu hình pipeline đã dùng
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "use_parent_expansion": use_parent_expansion,
    }

    # --- Bước 1: Retrieve ---
    if retrieval_mode == "dense":
        candidates = retrieve_dense(query, top_k=top_k_search)
    elif retrieval_mode == "sparse":
        candidates = retrieve_sparse(query, top_k=top_k_search)
    elif retrieval_mode == "hybrid":
        candidates = retrieve_hybrid(query, top_k=top_k_search)
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: {retrieval_mode}")

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:3]):
            cid = c['metadata'].get('chunk_id', '?')
            ctype = c['metadata'].get('chunk_type', '?')
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | {cid} ({ctype}) | {c['metadata'].get('source', '?')}")

    # --- Bước 2: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
    else:
        candidates = candidates[:top_k_select]

    # --- Bước 2.5: Parent expansion (hierarchical chunking) ---
    if use_parent_expansion:
        candidates = expand_with_parents(candidates)
        if verbose:
            print(f"[RAG] After parent expansion: {len(candidates)} chunks")

    if verbose:
        print(f"[RAG] Final chunks for prompt: {len(candidates)}")
        for c in candidates:
            cid = c['metadata'].get('chunk_id', '?')
            ctype = c['metadata'].get('chunk_type', '?')
            print(f"  → {cid} ({ctype})")

    # --- Bước 3: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Prompt:\n{prompt[:500]}...\n")

    # --- Bước 4: Generate ---
    answer = call_llm(prompt)

    # --- Bước 5: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

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
    A/B Rule: Chỉ đổi MỘT biến mỗi lần.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    strategies = [
        {"label": "dense", "mode": "dense", "rerank": False, "parent": False},
        {"label": "dense + parent", "mode": "dense", "rerank": False, "parent": True},
        {"label": "dense + rerank", "mode": "dense", "rerank": True, "parent": False},
        {"label": "hybrid", "mode": "hybrid", "rerank": False, "parent": False},
        {"label": "hybrid + parent", "mode": "hybrid", "rerank": False, "parent": True},
        {"label": "hybrid + rerank + parent", "mode": "hybrid", "rerank": True, "parent": True},
    ]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy['label']} ---")
        try:
            result = rag_answer(
                query,
                retrieval_mode=strategy["mode"],
                use_rerank=strategy["rerank"],
                use_parent_expansion=strategy["parent"],
                verbose=False,
            )
            print(f"Answer: {result['answer'][:200]}")
            print(f"Sources: {result['sources']}")
            chunk_ids = [c['metadata'].get('chunk_id', '?') for c in result['chunks_used']]
            print(f"Chunks: {chunk_ids}")
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
        print(f"\n{'─'*50}")
        print(f"Query: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {e}")

    # Sprint 3: So sánh strategies
    print("\n\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies("Approval Matrix để cấp quyền là tài liệu nào?")
    compare_retrieval_strategies("SLA xử lý ticket P1 là bao lâu?")

    print("\n✅ Sprint 2 + 3 hoàn thành!")
