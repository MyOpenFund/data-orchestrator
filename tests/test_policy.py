"""The run-outcome rule (issue #17) and the supported-mime configuration."""
import pytest

from data_orchestrator import policy, routing


def _outcome(ingested, empty, failed, ratio=0.10):
    return policy.run_outcome(docs_ingested=ingested, docs_empty=empty,
                              docs_failed=failed, ratio=ratio)


def test_nothing_ingested_with_a_failure_is_degraded_even_at_ratio_one():
    assert _outcome(0, 0, 1, ratio=1.0) == ("degraded", 3)


def test_ratio_exactly_at_threshold_is_ok_and_above_is_degraded():
    assert _outcome(9, 0, 1) == ("ok", 0)            # 1/10 = 0.10, not above
    assert _outcome(8, 0, 1) == ("degraded", 3)      # 1/9 > 0.10


def test_empty_documents_count_as_attempts():
    # 1 failed of 10 attempted (4 ingested + 5 empty): exactly 10 %, ok
    assert _outcome(4, 5, 1) == ("ok", 0)


def test_ratio_zero_degrades_on_any_failure():
    assert _outcome(1000, 0, 1, ratio=0.0) == ("degraded", 3)


def test_ratio_one_disables_rule_b():
    assert _outcome(1, 0, 99, ratio=1.0) == ("ok", 0)


def test_resume_only_run_is_ok_without_dividing_by_zero():
    assert _outcome(0, 0, 0) == ("ok", 0)


def test_empty_only_run_is_ok():
    assert _outcome(0, 3, 0) == ("ok", 0)


@pytest.mark.parametrize("raw,expected", [(None, 0.10), ("", 0.10), ("  ", 0.10),
                                          ("0", 0.0), ("1", 1.0), (" 0.25 ", 0.25)])
def test_ratio_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(policy.DEGRADED_RATIO_ENV, raising=False)
    else:
        monkeypatch.setenv(policy.DEGRADED_RATIO_ENV, raw)
    assert policy.degraded_failure_ratio() == expected


@pytest.mark.parametrize("raw", ["-0.1", "1.5", "abc", "nan", "10%"])
def test_ratio_env_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv(policy.DEGRADED_RATIO_ENV, raw)
    with pytest.raises(ValueError, match=policy.DEGRADED_RATIO_ENV):
        policy.degraded_failure_ratio()


def test_default_mimes_are_every_engine_format_and_exclude_html(monkeypatch):
    monkeypatch.delenv(routing.SUPPORTED_MIME_ENV, raising=False)
    mimes = routing.supported_mime_types()
    assert mimes == tuple(routing.MIME_EXTENSIONS)
    assert "text/html" not in mimes
    assert set(routing.supported_extensions(mimes)) == {
        ".pdf", ".md", ".txt", ".csv", ".docx", ".xlsx", ".pptx"}


def test_mime_env_is_trimmed_lowercased_and_deduplicated(monkeypatch):
    monkeypatch.setenv(routing.SUPPORTED_MIME_ENV,
                       " Application/PDF , application/pdf,,text/markdown ")
    assert routing.supported_mime_types() == ("application/pdf", "text/markdown")


def test_blank_mime_env_means_default(monkeypatch):
    monkeypatch.setenv(routing.SUPPORTED_MIME_ENV, "   ")
    assert routing.supported_mime_types() == routing.DEFAULT_SUPPORTED_MIME_TYPES


@pytest.mark.parametrize("raw", [",", " , ,"])
def test_mime_env_with_no_entry_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(routing.SUPPORTED_MIME_ENV, raw)
    with pytest.raises(ValueError, match="lists no mime type"):
        routing.supported_mime_types()


def test_mime_env_with_a_type_the_engine_cannot_read_is_rejected(monkeypatch):
    monkeypatch.setenv(routing.SUPPORTED_MIME_ENV, "application/pdf,text/html")
    with pytest.raises(ValueError, match="text/html"):
        routing.supported_mime_types()
