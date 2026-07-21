from __future__ import annotations

import os
from pathlib import Path


def _bootstrap_windows_dll_search_path() -> None:
    if os.name != 'nt':
        return

    for env_name in ('SPEEXDSP_BIN_DIR', 'SPEEXDSP_LIBRARY_DIR'):
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        os.add_dll_directory(str(path))


_bootstrap_windows_dll_search_path()

from ._speexdsp import *
