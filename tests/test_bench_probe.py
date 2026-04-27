"""bionpu.bench.probe — energy-reader availability probe tests.

GPL-3.0. (c) 2026 OpenSensor.

The probe must be deterministic in shape regardless of host
capabilities and never fabricate counters (POWER_DOMAINS.md §1.4 +
§3 are the load-bearing rules).
"""

from __future__ import annotations

from bionpu.bench.probe import ProbeReport, ReaderStatus, probe_readers


def test_probe_returns_one_status_per_device() -> None:
    report = probe_readers()
    assert isinstance(report, ProbeReport)
    devices = [r.device for r in report.readers]
    assert devices == ["cpu", "gpu", "npu"]


def test_each_status_has_source_and_detail() -> None:
    report = probe_readers()
    for r in report.readers:
        assert isinstance(r, ReaderStatus)
        assert r.device in {"cpu", "gpu", "npu"}
        assert isinstance(r.available, bool)
        assert r.source != ""
        assert r.detail != ""


def test_unavailable_readers_report_stub_source() -> None:
    """POWER_DOMAINS.md §1.4: never fabricate. When unavailable,
    source must be 'stub' and detail must explain why."""
    report = probe_readers()
    for r in report.readers:
        if not r.available:
            assert r.source == "stub", (
                f"unavailable {r.device} reader reported source={r.source!r}"
            )
            assert len(r.detail) > 10  # non-trivial reason


def test_to_json_is_round_trippable() -> None:
    """The probe payload must be plain-JSON-shaped for CI tooling."""
    import json

    report = probe_readers()
    blob = json.dumps(report.to_json())
    parsed = json.loads(blob)
    assert parsed["readers"][0]["device"] == "cpu"
    assert parsed["readers"][1]["device"] == "gpu"
    assert parsed["readers"][2]["device"] == "npu"


def test_all_real_consistent_with_per_device_status() -> None:
    report = probe_readers()
    expected_all_real = all(r.available for r in report.readers)
    assert report.all_real() is expected_all_real
