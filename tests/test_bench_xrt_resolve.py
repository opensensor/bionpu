"""bionpu.bench.energy.xrt — xrt-smi binary resolution tests.

GPL-3.0. (c) 2026 OpenSensor.

The probe must prefer the AMD vendor xrt-smi at
/opt/xilinx/xrt/bin/xrt-smi over Ubuntu's libxrt-utils generic build
at /usr/bin/xrt-smi: the latter is older Xilinx-FPGA-only and does
NOT enumerate AMD NPUs. Resolution preference matters because
``shutil.which("xrt-smi")`` will pick up /usr/bin first by default.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from bionpu.bench.energy.xrt import _XRT_SMI_DEFAULT, _resolve_xrt_smi


def _make_fake_xrt(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_resolve_uses_explicit_when_given(tmp_path: Path) -> None:
    fake = tmp_path / "explicit-xrt-smi"
    _make_fake_xrt(fake)
    assert _resolve_xrt_smi(explicit=fake) == str(fake)


def test_resolve_returns_none_for_missing_explicit(tmp_path: Path) -> None:
    assert _resolve_xrt_smi(explicit=tmp_path / "does-not-exist") is None


def test_resolve_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "env-xrt-smi"
    _make_fake_xrt(fake)
    monkeypatch.setenv("BIONPU_XRT_SMI", str(fake))
    monkeypatch.setattr(
        "bionpu.bench.energy.xrt._XRT_SMI_DEFAULT",
        str(tmp_path / "vendor-not-here"),
    )
    monkeypatch.setattr("shutil.which", lambda _: str(tmp_path / "ubuntu-not-here"))
    assert _resolve_xrt_smi() == str(fake)


def test_resolve_prefers_vendor_path_over_ubuntu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vendor build at /opt/xilinx/xrt/bin/xrt-smi must beat
    the Ubuntu libxrt-utils build at /usr/bin/xrt-smi — the latter
    can't enumerate AMD NPUs, the former can."""
    vendor = tmp_path / "vendor" / "xrt-smi"
    ubuntu = tmp_path / "usr-bin" / "xrt-smi"
    _make_fake_xrt(vendor)
    _make_fake_xrt(ubuntu)
    monkeypatch.setattr("bionpu.bench.energy.xrt._XRT_SMI_DEFAULT", str(vendor))
    monkeypatch.setattr("shutil.which", lambda _: str(ubuntu))
    monkeypatch.delenv("BIONPU_XRT_SMI", raising=False)
    assert _resolve_xrt_smi() == str(vendor)


def test_resolve_falls_back_to_path_when_vendor_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ubuntu = tmp_path / "ubuntu-xrt-smi"
    _make_fake_xrt(ubuntu)
    monkeypatch.setattr("bionpu.bench.energy.xrt._XRT_SMI_DEFAULT", "/non/existent/path")
    monkeypatch.setattr("shutil.which", lambda _: str(ubuntu))
    monkeypatch.delenv("BIONPU_XRT_SMI", raising=False)
    assert _resolve_xrt_smi() == str(ubuntu)


def test_resolve_returns_none_when_nothing_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bionpu.bench.energy.xrt._XRT_SMI_DEFAULT", "/non/existent/path")
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.delenv("BIONPU_XRT_SMI", raising=False)
    assert _resolve_xrt_smi() is None
