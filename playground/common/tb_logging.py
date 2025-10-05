"""
tb_logging.py — TensorBoard + console shim for JAX monitoring

Intercepts calls to jax.monitoring.record_scalar(...) and:
  1. Logs them to TensorBoard
  2. Prints them to console in real time
"""

import os
import time
import types
import jax
from torch.utils.tensorboard import SummaryWriter


# Create timestamped log directory under 'runs'
_log_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=_log_dir)

def _record_scalar_forward(name, value, step=None, **kwargs):
    """Forward scalar metrics to TensorBoard and print to console."""
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
        writer.add_scalar(name, v, s)
        writer.flush()
    except Exception:
        pass

    # Print to console (formatted)
    if s is not None:
        print(f"[Metrics] step={s:<6} {name:<40} {v:.6f}")
    else:
        print(f"[Metrics] {name:<40} {v:.6f}")

# Make sure jax.monitoring exists
if not hasattr(jax, "monitoring"):
    jax.monitoring = types.SimpleNamespace()

jax.monitoring.record_scalar = _record_scalar_forward

print(f"[tb_logging] ✅ TensorBoard logging active. Log dir: {_log_dir}")
print(f"[tb_logging] Scalars will also be printed to console.\n")
