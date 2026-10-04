"""Thread-count policy for ONNX Runtime and CTranslate2.

Measured on an M1 Pro (perf.md section 5): ORT intra-op threads = performance
cores with spin-waiting off is fastest and stops idle ORT threads competing
with the translator. ``GUJUSUB_THREADS`` overrides the detected core count.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

ENV_THREADS = "GUJUSUB_THREADS"
MIN_THREADS = 2
SYSCTL_PERF_CORES = ["sysctl", "-n", "hw.perflevel0.physicalcpu"]


def _env_threads() -> int | None:
    raw = os.environ.get(ENV_THREADS)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        logger.warning("ignoring invalid %s=%r", ENV_THREADS, raw)
        return None
    return value


def _macos_perf_cores() -> int | None:
    try:
        out = subprocess.run(
            SYSCTL_PERF_CORES, capture_output=True, text=True, check=True, timeout=2
        ).stdout
        value = int(out.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return value if value > 0 else None


def perf_cores() -> int:
    """Number of performance cores to use for compute thread pools."""
    override = _env_threads()
    if override is not None:
        return override
    if sys.platform == "darwin":
        cores = _macos_perf_cores()
        if cores is not None:
            return cores
    return max(MIN_THREADS, (os.cpu_count() or 4) - 2)


def ct2_threads() -> int:
    """intra_threads for the CTranslate2 translator (it shares the CPU with ASR)."""
    return max(MIN_THREADS, perf_cores() // 2)


def ort_session_options():
    """SessionOptions for every ORT session: intra = perf cores, no spin-waiting."""
    import onnxruntime  # deferred: keeps importing this module cheap

    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = perf_cores()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts
