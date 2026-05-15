from veri_rag.config.schema import AttackType


def test_poisoning_attack_success_on_baseline(pipeline):
    row = pipeline.run_with_attack(
        "q001",
        "What is the refund period?",
        AttackType.POISONING,
        "30 days",
        defense="none",
    )
    assert row["attack_success_baseline"] is True


def test_veri_rag_repair_reduces_attack(pipeline):
    row = pipeline.run_with_attack(
        "q001",
        "What is the refund period?",
        AttackType.POISONING,
        "30 days",
        defense="veri_rag",
    )
    assert row.get("removed_chunks")
    assert row["certificate_score"] > 0
