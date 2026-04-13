"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# TODO Sprint 1: Điều chỉnh chunk size và overlap theo quyết định của nhóm
# Gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or (line.isupper() and len(line.strip()) > 2):
                continue
            else:
                # Bắt đầu nội dung khi không có dòng === (metadata đã xong)
                header_done = True
                content_lines.append(line)
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)

    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Chia section dài thành nhiều chunk: ưu tiên cắt tại \\n\\n, \\n, hoặc khoảng trắng;
    giữ overlap giữa hai chunk liên tiếp.
    """
    meta = {**base_metadata, "section": section}
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_chars:
        return [{"text": text, "metadata": meta.copy()}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    n = len(text)
    min_break = max(chunk_chars // 5, 80)

    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            window = text[start:end]
            br = window.rfind("\n\n")
            if br == -1 or br < min_break:
                br = window.rfind("\n")
            if br == -1 or br < min_break:
                br = window.rfind(" ")
            if br != -1 and br >= min_break:
                end = start + br

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "metadata": meta.copy()})

        if end >= n:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks



# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

_embedding_st_model = None  # lazy SentenceTransformer


def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    - Nếu có OPENAI_API_KEY: dùng OpenAI (EMBEDDING_MODEL, mặc định text-embedding-3-small).
    - Ngược lại: Sentence Transformers local (ST_EMBEDDING_MODEL, mặc định paraphrase-multilingual-MiniLM-L12-v2).
    """
    text = (text or "").strip() or " "

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding

    global _embedding_st_model
    if _embedding_st_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_st_model = SentenceTransformer(
            os.getenv("ST_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        )
    vec = _embedding_st_model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def _normalize_chroma_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma chỉ nhận giá trị primitive; chuỗi hóa các kiểu khác."""
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → lưu ChromaDB.
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = sorted(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw_text, str(filepath))
        chunks = chunk_document(doc)
        if not chunks:
            print("    → 0 chunks (bỏ qua)")
            continue

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            text = chunk["text"]
            emb = get_embedding(text)
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(text)
            metadatas.append(_normalize_chroma_metadata(chunk["metadata"]))

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"    → Đã index {len(chunks)} chunks")

    print(f"\nHoàn thành! Tổng số chunks đã lưu: {total_chunks}")

# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In n chunk đầu trong ChromaDB để kiểm tra chất lượng index.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        print(f"\n=== Top {min(n, len(docs))} chunks trong index ===\n")
        if not docs:
            print("(Index trống — chạy build_index() trước.)")
            return

        for i, (doc, meta) in enumerate(zip(docs, metas)):
            meta = meta or {}
            preview = (doc or "")[:120]
            print(f"[Chunk {i+1}]")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Section: {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Text preview: {preview}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])
        metas = results.get("metadatas") or []

        print(f"\nTổng chunks: {len(metas)}")

        departments: Dict[str, int] = {}
        missing_date = 0
        missing_source = 0
        for meta in metas:
            meta = meta or {}
            dept = meta.get("department") or "unknown"
            departments[dept] = departments.get(dept, 0) + 1
            ed = meta.get("effective_date")
            if ed in ("unknown", "", None):
                missing_date += 1
            src = meta.get("source")
            if not src or str(src).strip() == "":
                missing_source += 1

        print("Phân bố theo department:")
        for dept, count in sorted(departments.items(), key=lambda x: -x[1]):
            print(f"  {dept}: {count} chunks")
        print(f"Chunks thiếu effective_date (unknown/rỗng): {missing_date}")
        print(f"Chunks thiếu source: {missing_source}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {doc['metadata']}")
        print(f"  Số chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Bước 3: Build index (yêu cầu implement get_embedding)
    print("\n--- Build Full Index ---")
    print("Lưu ý: Cần implement get_embedding() trước khi chạy bước này!")
    # Uncomment dòng dưới sau khi implement get_embedding():
    # build_index()

    # Bước 4: Kiểm tra index
    # Uncomment sau khi build_index() thành công:
    # list_chunks()
    # inspect_metadata_coverage()

    print("\nSprint 1 setup hoàn thành!")
    print("Việc cần làm:")
    print("  1. Implement get_embedding() - chọn OpenAI hoặc Sentence Transformers")
    print("  2. Implement phần TODO trong build_index()")
    print("  3. Chạy build_index() và kiểm tra với list_chunks()")
    print("  4. Nếu chunking chưa tốt: cải thiện _split_by_size() để split theo paragraph")
