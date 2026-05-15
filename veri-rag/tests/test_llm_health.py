from veri_rag.rag.llm_health import resolve_profile_with_fallback


def test_resolve_mock_unchanged():
    profile, warning = resolve_profile_with_fallback("mock", fallback_mock=True)
    assert profile == "mock"
    assert warning is None


def test_resolve_openai_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    profile, warning = resolve_profile_with_fallback("openai", fallback_mock=True)
    assert profile == "mock"
    assert warning is not None
