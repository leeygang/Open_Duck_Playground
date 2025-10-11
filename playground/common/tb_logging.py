"""
Unified TensorBoard logging wrapper + JAX monitoring shim.

Goals:
- Provide a single global TensorBoard writer for the entire app.
- Let the runner inject its own writer/logdir (no duplicate writers or dirs).
- Forward jax.monitoring.record_scalar to the same writer.
- Offer a simple add_scalar(name, value, step) API for consistency.
"""

import os
import time
import types
import threading
import jax
from torch.utils.tensorboard import SummaryWriter


_writer = None  # type: SummaryWriter | None
_log_dir = None  # type: str | None
_flush_interval_s = 2.0
_last_flush = 0.0
_lock = threading.Lock()

def _ensure_writer():
    """Create a default writer if none set, using runs/<timestamp>."""
    global _writer, _log_dir
    if _writer is None:
        default_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
        set_log_dir(default_dir)

def set_writer(w: SummaryWriter):
    """Inject an existing SummaryWriter to be used globally."""
    global _writer, _log_dir
    with _lock:
        _writer = w
        # Try to discover log_dir for message; SummaryWriter doesn't expose it directly
        try:
            _log_dir = getattr(w, "log_dir", None) or getattr(w, "_logdir", None)
        except Exception:
            _log_dir = None
    if _log_dir:
        print(f"[tb_logging] TensorBoard logging active. Log dir: {_log_dir}")
    else:
        print("[tb_logging] TensorBoard logging active (injected writer)")

def set_log_dir(dir_path: str):
    """Create a new SummaryWriter at the given directory and set it globally."""
    global _writer, _log_dir
    with _lock:
        if _writer is not None:
            try:
                _writer.flush()
                _writer.close()
            except Exception:
                pass
        os.makedirs(dir_path, exist_ok=True)
        _writer = SummaryWriter(log_dir=dir_path)
        _log_dir = dir_path
    print(f"[tb_logging] ✅ TensorBoard logging active. Log dir: {dir_path}")

def add_scalar(name: str, value, step=None):
    """Forward scalar metrics to TensorBoard and print to console."""
    _ensure_writer()
    try:
        v = jax.device_get(value)
    except Exception:
        v = value
    try:
        v = float(v)
    except Exception:
        return  # skip non-numeric values

    s = int(step) if step is not None else None

    # Log to TensorBoard
    try:
        if _writer is not None:
            _writer.add_scalar(name, v, s)
            # Throttle flushes to reduce overhead
            global _last_flush
            now = time.time()
            if now - _last_flush > _flush_interval_s:
                _writer.flush()
                _last_flush = now
    except Exception:
        pass

    # Print to console (formatted)
    # if s is not None:
    #     print(f"[Metrics] step={s:<6} {name:<40} {v:.6f}")
    # else:
    #     print(f"[Metrics] {name:<40} {v:.6f}")

# Make sure jax.monitoring exists
if not hasattr(jax, "monitoring"):
    jax.monitoring = types.SimpleNamespace()

def _record_scalar_forward(name, value, step=None, **kwargs):
    add_scalar(name, value, step=step)

jax.monitoring.record_scalar = _record_scalar_forward
