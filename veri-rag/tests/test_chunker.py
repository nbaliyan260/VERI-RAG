from veri_rag.config.schema import Document
from veri_rag.corpus.chunker import chunk_document


def test_chunk_document_splits_long_text():
    doc = Document(text="A. " * 200, document_id="d1")
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20, min_chunk_size=30)
    assert len(chunks) >= 2
    assert all(c.document_id == "d1" for c in chunks)


def test_small_document_single_chunk():
    doc = Document(text="Short policy text.", document_id="d2")
    chunks = chunk_document(doc, chunk_size=500)
    assert len(chunks) == 1
