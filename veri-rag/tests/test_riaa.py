from veri_rag.config.schema import AttackType


def test_riaa_pairwise_runs(pipeline):
    row = pipeline.run_with_attack(
        "q001",
        "What is the refund period?",
        AttackType.POISONING,
        "30 days",
        defense="veri_rag",
        enable_pairwise=True,
    )
    assert "certificate_score" in row
    assert row.get("certified_bound_post") is not None


def test_adaptive_attack_generates(pipeline):
    from veri_rag.attacks.adaptive import AdaptiveAttack
    from veri_rag.config.schema import AttackSpec

    chunks = AdaptiveAttack().generate(
        AttackSpec(
            attack_type=AttackType.ADAPTIVE,
            target_query="refund",
            target_wrong_answer="seven-day window",
        )
    )
    assert "ignore" not in chunks[0].text.lower()
