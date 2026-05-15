from veri_rag.config.schema import AttackType


def test_full_pipeline_produces_certificate_file(pipeline, project_root):
    pipeline.run_with_attack(
        "q001",
        "What is the refund period?",
        AttackType.POISONING,
        "30 days",
        defense="veri_rag",
    )
    cert = project_root / "outputs" / "certificates" / "q001.json"
    assert cert.exists()
