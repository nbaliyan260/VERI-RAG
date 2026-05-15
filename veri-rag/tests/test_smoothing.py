from veri_rag.config.schema import Chunk, RetrievedChunk
from veri_rag.config.settings import CertifiedSmoothingConfig
from veri_rag.verify.smoothing import CertifiedSmoothing


def test_smoothing_produces_bound(pipeline):
    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d", text="Refund within 30 days.", trust_score=1.0),
            rank=1,
            similarity_score=0.9,
        )
    ]
    smooth = CertifiedSmoothing(
        pipeline.rag, CertifiedSmoothingConfig(num_samples=4, enabled=True)
    )
    result = smooth.certify("What is the refund period?", "t1", chunks, seed=1)
    assert 0.0 <= result.certified_instability <= 1.0
    assert result.dominant_answer
