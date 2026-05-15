from veri_rag.attacks.poisoning import PoisoningAttack
from veri_rag.attacks.prompt_injection import PromptInjectionAttack
from veri_rag.attacks.secret_leakage import SecretLeakageAttack
from veri_rag.config.schema import AttackSpec, AttackType


def test_poisoning_generates_attack_chunk():
    spec = AttackSpec(
        attack_type=AttackType.POISONING,
        target_query="What is the refund period?",
        target_wrong_answer="7 days",
    )
    chunks = PoisoningAttack().generate(spec)
    assert len(chunks) == 1
    assert chunks[0].is_attack
    assert "7 days" in chunks[0].text


def test_prompt_injection_has_instruction_patterns():
    spec = AttackSpec(
        attack_type=AttackType.PROMPT_INJECTION,
        target_query="refund",
        parameters={"target_instruction": "Answer 7 days."},
    )
    chunks = PromptInjectionAttack().generate(spec)
    assert any("ignore" in c.text.lower() for c in chunks)


def test_secret_leakage_contains_fake_secrets():
    spec = AttackSpec(attack_type=AttackType.SECRET_LEAKAGE, target_query="secrets")
    chunks = SecretLeakageAttack().generate(spec)
    assert "API_KEY_TEST_12345" in chunks[0].text
