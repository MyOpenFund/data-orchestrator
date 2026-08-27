"""Golden tests pinning _point_id byte-stability across the engine rewrite."""
from rag_orchestrator.core import _point_id


GOLDEN_1 = 5299910714168844021
GOLDEN_2 = 5572097760344244118
GOLDEN_3 = 6912429553972229415
GOLDEN_4 = 703449921473339669


def test_point_id_golden_values():
    # Pinned outputs of the pre-rewrite implementation. If these change, ids of
    # every existing collection would drift — never accept a change here.
    assert _point_id("c184d44f298ff622", 1, 0) == GOLDEN_1
    assert _point_id("c184d44f298ff622", 1, 1) == GOLDEN_2
    assert _point_id("c184d44f298ff622", 2, 0) == GOLDEN_3
    assert _point_id("0" * 16, 0, 0) == GOLDEN_4


def test_point_id_is_deterministic_and_positive():
    a = _point_id("abc", 3, 7)
    assert a == _point_id("abc", 3, 7)
    assert 0 <= a < 2**63
