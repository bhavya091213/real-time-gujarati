import dataclasses
import subprocess

import pytest

from gujusub import threads
from gujusub.engine import ASREngine, Transcript, Word


def _sysctl(stdout):
    return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout)


def test_word_is_frozen():
    w = Word("hi", 0.0, 0.08, 0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.text = "x"


def test_transcript_empty():
    t = Transcript.empty()
    assert t.text == ""
    assert t.words == ()


def test_protocol_is_structural():
    class Fake:
        lang = "gu"

        def transcribe(self, audio):
            return ""

        def transcribe_detailed(self, audio):
            return Transcript.empty()

        def warmup(self):
            return None

    assert isinstance(Fake(), ASREngine)


def test_perf_cores_reads_sysctl_on_macos(monkeypatch):
    monkeypatch.delenv("GUJUSUB_THREADS", raising=False)
    monkeypatch.setattr(threads.sys, "platform", "darwin")
    monkeypatch.setattr(threads.subprocess, "run", _sysctl("6\n"))
    assert threads.perf_cores() == 6


def test_perf_cores_falls_back_when_sysctl_fails(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("sysctl")

    monkeypatch.delenv("GUJUSUB_THREADS", raising=False)
    monkeypatch.setattr(threads.sys, "platform", "darwin")
    monkeypatch.setattr(threads.subprocess, "run", boom)
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 10)
    assert threads.perf_cores() == 8


def test_perf_cores_fallback_floor(monkeypatch):
    monkeypatch.delenv("GUJUSUB_THREADS", raising=False)
    monkeypatch.setattr(threads.sys, "platform", "linux")
    monkeypatch.setattr(threads.os, "cpu_count", lambda: None)
    assert threads.perf_cores() == 2


def test_perf_cores_bad_sysctl_output_falls_back(monkeypatch):
    monkeypatch.delenv("GUJUSUB_THREADS", raising=False)
    monkeypatch.setattr(threads.sys, "platform", "darwin")
    monkeypatch.setattr(threads.subprocess, "run", _sysctl("?"))
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 6)
    assert threads.perf_cores() == 4


def test_env_override(monkeypatch):
    monkeypatch.setenv("GUJUSUB_THREADS", "3")
    assert threads.perf_cores() == 3
    assert threads.ct2_threads() == 2


@pytest.mark.parametrize("bad", ["0", "-1", "lots"])
def test_bad_env_override_ignored(monkeypatch, bad):
    monkeypatch.setenv("GUJUSUB_THREADS", bad)
    monkeypatch.setattr(threads.sys, "platform", "linux")
    monkeypatch.setattr(threads.os, "cpu_count", lambda: 12)
    assert threads.perf_cores() == 10


def test_ct2_threads_is_half(monkeypatch):
    monkeypatch.setenv("GUJUSUB_THREADS", "8")
    assert threads.ct2_threads() == 4


def test_ort_session_options(monkeypatch):
    monkeypatch.setenv("GUJUSUB_THREADS", "5")
    opts = threads.ort_session_options()
    assert opts.intra_op_num_threads == 5
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"
