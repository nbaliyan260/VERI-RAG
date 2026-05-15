from veri_rag.rag.llm_health import resolve_profile_with_fallback


def test_resolve_mock_unchanged():
    profile, warning = resolve_profile_with_fallback("mock", fallback_mock=True)
    assert profile == "mock"
    assert warning is None


def test_resolve_openai_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    profile, warning = resolve_profile_with_fallback("openai", fallback_mock=True)
    assert profile == "mock"
    assert warning is not None


def test_resolve_auto_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from veri_rag.rag.llm_health import resolve_auto_profile

    assert resolve_auto_profile() == "claude_haiku"
