"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo 2 phương pháp:
    + Flat chunking: cắt theo section, split theo paragraph
    + Hierarchical (Parent-Child): section = parent, sub-parts = children
  - Gắn metadata JSON: source, section, department, effective_date, chunk_id, parent_id
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Mỗi chunk có metadata: source, section, effective_date, chunk_id, parent_id, chunk_type
  ✓ Có thể kiểm tra chunk bằng list_chunks()
  ✓ Metadata JSON file exported cho reference
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
METADATA_JSON_PATH = Path(__file__).parent / "data" / "chunks_metadata.json"

# Chunk size và overlap — chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk

# Chunking strategy: "flat" hoặc "hierarchical"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hierarchical")


# =============================================================================
# EMBEDDING MODEL (cached globally)
# =============================================================================

_embedding_model = None


def _get_embedding_model():
    """Load và cache embedding model (sentence-transformers)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        print(f"[Embedding] Loading model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.
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
            elif line.strip() == "" or line.isupper():
                continue
            else:
                # Dòng note/ghi chú trước section đầu → thêm vào nội dung
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
# STEP 2A: FLAT CHUNKING (phương pháp gốc)
# Cắt theo section heading, split theo paragraph nếu quá dài
# =============================================================================

def chunk_document_flat(doc: Dict[str, Any], doc_stem: str) -> List[Dict[str, Any]]:
    """
    Flat chunking: mỗi section → 1 hoặc nhiều chunk ngang hàng.
    ID dạng: "{doc_stem}_{index}" (vd: "policy_refund_v4_0", "policy_refund_v4_1")
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    sections = re.split(r"(===.*?===)", text)
    current_section = "General"
    current_section_text = ""
    chunk_index = 0

    for part in sections:
        if re.match(r"===.*?===", part):
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                    doc_stem=doc_stem,
                    start_index=chunk_index,
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
            doc_stem=doc_stem,
            start_index=chunk_index,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    doc_stem: str,
    start_index: int,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """Split text theo paragraph với overlap."""
    chunk_id = f"{doc_stem}_{start_index}"

    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {
                **base_metadata,
                "section": section,
                "chunk_id": chunk_id,
                "parent_id": "",
                "chunk_type": "flat",
            },
        }]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    idx = start_index

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_chars:
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    **base_metadata,
                    "section": section,
                    "chunk_id": f"{doc_stem}_{idx}",
                    "parent_id": "",
                    "chunk_type": "flat",
                },
            })
            idx += 1

            # Overlap
            if len(current_chunk) > overlap_chars:
                overlap = current_chunk[-overlap_chars:]
                sent_start = overlap.find(". ")
                if sent_start != -1:
                    overlap = overlap[sent_start + 2:]
                current_chunk = overlap + "\n\n" + para
            else:
                current_chunk = current_chunk + "\n\n" + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {
                **base_metadata,
                "section": section,
                "chunk_id": f"{doc_stem}_{idx}",
                "parent_id": "",
                "chunk_type": "flat",
            },
        })

    return chunks


# =============================================================================
# STEP 2B: HIERARCHICAL CHUNKING (Parent-Child)
# Section = Parent chunk, sub-parts = Child chunks
# Parent ID: "{doc_stem}_{section_index}" (vd: "policy_refund_v4_1")
# Child ID:  "{doc_stem}_{section_index}.{child_index}" (vd: "policy_refund_v4_1.1")
# =============================================================================

def chunk_document_hierarchical(doc: Dict[str, Any], doc_stem: str) -> List[Dict[str, Any]]:
    """
    Hierarchical (Parent-Child) chunking:

    - Parent chunk = toàn bộ section (text đầy đủ)
      ID: "{doc_stem}_{section_index}", ví dụ "policy_refund_v4_1"

    - Child chunk = phần nhỏ trong section (split theo paragraph)
      ID: "{doc_stem}_{section_index}.{child_index}", ví dụ "policy_refund_v4_1.1"

    Retrieval flow:
      1. Search tìm child chunks (nhỏ, focused)
      2. Khi tìm được child → lấy thêm parent chunk để có ngữ cảnh rộng hơn
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Split theo heading "=== ... ==="
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""
    section_index = 0

    for part in sections:
        if re.match(r"===.*?===", part):
            # Xử lý section trước
            if current_section_text.strip():
                section_index += 1
                section_chunks = _create_parent_children(
                    text=current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                    doc_stem=doc_stem,
                    section_index=section_index,
                )
                chunks.extend(section_chunks)

            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Section cuối cùng
    if current_section_text.strip():
        section_index += 1
        section_chunks = _create_parent_children(
            text=current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
            doc_stem=doc_stem,
            section_index=section_index,
        )
        chunks.extend(section_chunks)

    return chunks


def _create_parent_children(
    text: str,
    base_metadata: Dict,
    section: str,
    doc_stem: str,
    section_index: int,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Tạo parent chunk + child chunks cho một section.

    - Luôn tạo 1 parent chunk chứa toàn bộ section text
    - Nếu section dài hơn chunk_chars → tạo thêm child chunks
    - Nếu section ngắn → chỉ có parent, không có children
    """
    parent_id = f"{doc_stem}_{section_index}"

    # ── PARENT CHUNK: toàn bộ section ──
    parent_chunk = {
        "text": text,
        "metadata": {
            **base_metadata,
            "section": section,
            "chunk_id": parent_id,
            "parent_id": "",          # Parent không có parent
            "chunk_type": "parent",
        }
    }

    chunks = [parent_chunk]

    # ── CHILD CHUNKS: chỉ tạo nếu section đủ dài ──
    if len(text) <= chunk_chars:
        # Section ngắn → parent đủ rồi, không cần children
        return chunks

    paragraphs = text.split("\n\n")
    current_chunk = ""
    child_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_chars:
            child_index += 1
            child_id = f"{parent_id}.{child_index}"

            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    **base_metadata,
                    "section": section,
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                },
            })

            # Overlap: giữ phần cuối chunk trước
            if len(current_chunk) > overlap_chars:
                overlap = current_chunk[-overlap_chars:]
                sent_start = overlap.find(". ")
                if sent_start != -1:
                    overlap = overlap[sent_start + 2:]
                current_chunk = overlap + "\n\n" + para
            else:
                current_chunk = current_chunk + "\n\n" + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Child chunk cuối
    if current_chunk.strip():
        child_index += 1
        child_id = f"{parent_id}.{child_index}"
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": {
                **base_metadata,
                "section": section,
                "chunk_id": child_id,
                "parent_id": parent_id,
                "chunk_type": "child",
            },
        })

    return chunks


# =============================================================================
# STEP 2: CHUNK (dispatcher)
# =============================================================================

def chunk_document(doc: Dict[str, Any], doc_stem: str, strategy: str = CHUNKING_STRATEGY) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu, chọn strategy:
      - "flat": mỗi section → 1+ chunk ngang hàng
      - "hierarchical": section = parent, sub-parts = children (có chunk_id phân cấp)
    """
    if strategy == "hierarchical":
        return chunk_document_hierarchical(doc, doc_stem)
    else:
        return chunk_document_flat(doc, doc_stem)


# =============================================================================
# STEP 3: EMBED + STORE
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.
    Dùng Sentence Transformers (local, không cần API key).
    Model: paraphrase-multilingual-MiniLM-L12-v2 (hỗ trợ tiếng Việt)
    """
    model = _get_embedding_model()
    return model.encode(text).tolist()


def build_index(
    docs_dir: Path = DOCS_DIR,
    db_dir: Path = CHROMA_DB_DIR,
    strategy: str = CHUNKING_STRATEGY,
) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.
    Hỗ trợ 2 strategies: "flat" và "hierarchical".
    Export metadata JSON file cho reference.
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    print(f"Chunking strategy: {strategy}")
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))

    # Xóa collection cũ nếu có để rebuild sạch
    try:
        client.delete_collection("rag_lab")
        print("  Đã xóa collection cũ.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    all_metadata = []  # Để export JSON
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"\n  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # Preprocess
        doc = preprocess_document(raw_text, str(filepath))

        # Chunk (with doc_stem for unique IDs)
        doc_stem = filepath.stem
        chunks = chunk_document(doc, doc_stem=doc_stem, strategy=strategy)

        # Thống kê
        parents = [c for c in chunks if c["metadata"]["chunk_type"] == "parent"]
        children = [c for c in chunks if c["metadata"]["chunk_type"] == "child"]
        flats = [c for c in chunks if c["metadata"]["chunk_type"] == "flat"]

        if strategy == "hierarchical":
            print(f"    → {len(parents)} parent chunks, {len(children)} child chunks")
        else:
            print(f"    → {len(flats)} flat chunks")

        # Embed và upsert từng chunk
        for chunk in chunks:
            chunk_id = chunk["metadata"]["chunk_id"]
            embedding = get_embedding(chunk["text"])
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
            )

            # Lưu metadata cho JSON export
            all_metadata.append({
                "chunk_id": chunk_id,
                "parent_id": chunk["metadata"].get("parent_id", ""),
                "chunk_type": chunk["metadata"]["chunk_type"],
                "source": chunk["metadata"]["source"],
                "section": chunk["metadata"]["section"],
                "department": chunk["metadata"]["department"],
                "effective_date": chunk["metadata"]["effective_date"],
                "access": chunk["metadata"]["access"],
                "text_length": len(chunk["text"]),
                "text_preview": chunk["text"][:100] + "...",
            })

        total_chunks += len(chunks)

    # Export metadata JSON
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Metadata JSON saved: {METADATA_JSON_PATH}")

    print(f"\n✅ Hoàn thành! Tổng số chunks: {total_chunks}")
    print(f"   Strategy: {strategy}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 10) -> None:
    """In ra n chunk đầu tiên để kiểm tra chất lượng index."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            chunk_type = meta.get("chunk_type", "?")
            chunk_id = meta.get("chunk_id", "?")
            parent_id = meta.get("parent_id", "")
            marker = "📦" if chunk_type == "parent" else ("🧩" if chunk_type == "child" else "📄")

            print(f"{marker} [Chunk {i+1}]")
            print(f"  chunk_id:       {chunk_id}")
            if parent_id:
                print(f"  parent_id:      {parent_id}")
            print(f"  chunk_type:     {chunk_type}")
            print(f"  Source:         {meta.get('source', 'N/A')}")
            print(f"  Section:        {meta.get('section', 'N/A')}")
            print(f"  Effective Date: {meta.get('effective_date', 'N/A')}")
            print(f"  Department:     {meta.get('department', 'N/A')}")
            print(f"  Text preview:   {doc[:120]}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """Kiểm tra phân phối metadata trong toàn bộ index."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        total = len(results["metadatas"])
        print(f"\nTổng chunks: {total}")

        # Thống kê
        departments = {}
        sources = {}
        chunk_types = {}
        parent_count = 0
        child_count = 0
        missing_date = 0

        for meta in results["metadatas"]:
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1

            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

            ct = meta.get("chunk_type", "unknown")
            chunk_types[ct] = chunk_types.get(ct, 0) + 1

            if ct == "parent":
                parent_count += 1
            elif ct == "child":
                child_count += 1

            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

        print("\n📊 Phân bố theo chunk_type:")
        for ct, count in chunk_types.items():
            print(f"  {ct}: {count} chunks")

        print("\n📊 Phân bố theo department:")
        for dept, count in departments.items():
            print(f"  {dept}: {count} chunks")

        print("\n📊 Phân bố theo source:")
        for src, count in sources.items():
            print(f"  {src}: {count} chunks")

        print(f"\n⚠️  Chunks thiếu effective_date: {missing_date}")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


def print_hierarchy(db_dir: Path = CHROMA_DB_DIR) -> None:
    """In cây phân cấp parent-child để visualize structure."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas", "documents"])

        # Group by parent
        parents = {}
        children = {}

        for meta, doc in zip(results["metadatas"], results["documents"]):
            chunk_id = meta.get("chunk_id", "")
            chunk_type = meta.get("chunk_type", "")
            parent_id = meta.get("parent_id", "")
            section = meta.get("section", "")
            source = meta.get("source", "")

            if chunk_type == "parent":
                parents[chunk_id] = {
                    "section": section,
                    "source": source,
                    "text_len": len(doc),
                }
            elif chunk_type == "child":
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append({
                    "chunk_id": chunk_id,
                    "text_len": len(doc),
                    "preview": doc[:80],
                })

        print(f"\n{'='*60}")
        print("🌳 Chunk Hierarchy (Parent → Children)")
        print('='*60)

        for pid in sorted(parents.keys()):
            p = parents[pid]
            print(f"\n📦 {pid}")
            print(f"   Section: {p['section']}")
            print(f"   Source:  {p['source']}")
            print(f"   Length:  {p['text_len']} chars")

            if pid in children:
                for child in sorted(children[pid], key=lambda c: c["chunk_id"]):
                    print(f"   🧩 {child['chunk_id']} ({child['text_len']} chars)")
                    print(f"      {child['preview']}...")
            else:
                print(f"   (no children — section fits in one chunk)")

    except Exception as e:
        print(f"Lỗi: {e}")


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

    # Bước 2: Test preprocess + chunking (preview)
    print("\n--- Test preprocess + hierarchical chunking ---")
    for filepath in doc_files[:1]:
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc, doc_stem=filepath.stem, strategy="hierarchical")
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata: {json.dumps(doc['metadata'], ensure_ascii=False, indent=4)}")
        print(f"  Tổng chunks: {len(chunks)}")
        parents = [c for c in chunks if c["metadata"]["chunk_type"] == "parent"]
        children = [c for c in chunks if c["metadata"]["chunk_type"] == "child"]
        print(f"  Parents: {len(parents)}, Children: {len(children)}")
        for chunk in chunks[:4]:
            m = chunk["metadata"]
            print(f"\n  [{m['chunk_type'].upper()}] ID={m['chunk_id']}", end="")
            if m["parent_id"]:
                print(f" (parent={m['parent_id']})", end="")
            print(f"\n  Section: {m['section']}")
            print(f"  Text: {chunk['text'][:120]}...")

    # Bước 3: Build index (hierarchical by default)
    print("\n--- Build Full Index (hierarchical) ---")
    build_index(strategy="hierarchical")

    # Bước 4: Kiểm tra
    list_chunks(n=8)
    inspect_metadata_coverage()
    print_hierarchy()

    print("\n✅ Sprint 1 hoàn thành!")
