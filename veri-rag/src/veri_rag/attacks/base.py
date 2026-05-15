"""Base attack interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod

from veri_rag.config.schema import AttackSpec, AttackType, Chunk


class BaseAttack(ABC):
    """Generate malicious chunks for a given attack specification."""

    attack_type: AttackType

    @abstractmethod
    def generate(self, spec: AttackSpec) -> list[Chunk]:
        """Return one or more malicious chunks to inject into retrieval."""


def get_attack(attack_type: str | AttackType) -> BaseAttack:
    """Factory for attack implementations."""
    from veri_rag.attacks.adaptive import AdaptiveAttack
    from veri_rag.attacks.blocker import BlockerAttack
    from veri_rag.attacks.poisoning import PoisoningAttack
    from veri_rag.attacks.prompt_injection import PromptInjectionAttack
    from veri_rag.attacks.secret_leakage import SecretLeakageAttack
    from veri_rag.attacks.topic_flip import TopicFlipAttack

    if isinstance(attack_type, str):
        attack_type = AttackType(attack_type)

    mapping: dict[AttackType, type[BaseAttack]] = {
        AttackType.POISONING: PoisoningAttack,
        AttackType.PROMPT_INJECTION: PromptInjectionAttack,
        AttackType.SECRET_LEAKAGE: SecretLeakageAttack,
        AttackType.BLOCKER: BlockerAttack,
        AttackType.TOPIC_FLIP: TopicFlipAttack,
        AttackType.ADAPTIVE: AdaptiveAttack,
    }
    cls = mapping.get(attack_type)
    if cls is None:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return cls()
