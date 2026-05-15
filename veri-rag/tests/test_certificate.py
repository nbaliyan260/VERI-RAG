from veri_rag.config.schema import (
    RepairAction,
    RepairActionType,
    RepairResult,
    RiskFeatures,
    RiskScore,
    VerificationTests,
)
from veri_rag.config.settings import VerificationConfig
from veri_rag.verify.verifier import Verifier


def test_certificate_passes_clean_repair():
    verifier = Verifier(VerificationConfig())
    repair = RepairResult(
        query_id="q1",
        original_answer="7 days [c_x]",
        repaired_answer="Refunds are allowed within 30 days. [c_a]",
        removed_chunk_ids=["c_x"],
    )
    risk = {
        "c_a": RiskScore(
            chunk_id="c_a",
            features=RiskFeatures(),
            risk_score=0.1,
        )
    }
    from veri_rag.config.schema import Chunk, RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d", text="30 days refund", trust_score=1.0),
            rank=1,
            similarity_score=0.8,
        )
    ]
    cert = verifier.verify("q1", repair, risk, repaired_chunks=chunks, gold_answer="30 days")
    assert cert.certificate_score > 0
    assert cert.verification_tests.no_sensitive_leakage
