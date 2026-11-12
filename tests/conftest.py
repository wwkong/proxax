from __future__ import annotations

import os

# Prevent XLA from attempting to preallocate 75% of total GPU VRAM (9 GiB) at
# startup when sharing the GPU with the host desktop environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
