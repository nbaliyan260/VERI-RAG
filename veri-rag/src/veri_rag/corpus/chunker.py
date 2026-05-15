"""Document chunker — splits documents into fixed-size overlapping chunks."""

from __future__ import annotations

from veri_rag.config.schema import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
) -> list[Chunk]:
    """Split a document into overlapping chunks by character count.

    Args:
        document: Source document to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        min_chunk_size: Minimum characters for a valid chunk.

    Returns:
        List of Chunk objects.
    """
    text = document.text.strip()
    if not text:
        return []

    # If the entire document is smaller than chunk_size, return it as one chunk
    if len(text) <= chunk_size:
        return [
            Chunk(
                document_id=document.document_id,
                source=document.source,
                text=text,
                metadata={**document.metadata, "chunk_index": 0, "total_chunks": 1},
                is_attack=document.is_attack,
                attack_type=document.attack_type,
                trust_score=1.0 if not document.is_attack else 0.0,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for the last sentence-ending punctuation within the chunk
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)
            break_point = max(last_period, last_newline)

            if break_point > start + min_chunk_size:
                end = break_point + 1

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= min_chunk_size:
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    source=document.source,
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                    },
                    is_attack=document.is_attack,
                    attack_type=document.attack_type,
                    trust_score=1.0 if not document.is_attack else 0.0,
                )
            )
            chunk_index += 1

        start = end - chunk_overlap
        if start >= len(text):
            break

    # Update total_chunks metadata
    for c in chunks:
        c.metadata["total_chunks"] = len(chunks)

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
) -> list[Chunk]:
    """Chunk a list of documents."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size, chunk_overlap, min_chunk_size)
        )
    return all_chunks
