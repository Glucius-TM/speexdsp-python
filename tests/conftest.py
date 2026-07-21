import os
from pathlib import Path

_DLL_HANDLES = []


def _add_windows_dll_dirs() -> None:
    if os.name != 'nt':
        return

    for env_name in ('SPEEXDSP_BIN_DIR', 'SPEEXDSP_LIBRARY_DIR'):
        raw_path = os.environ.get(env_name)
        if not raw_path:
            continue

        path = Path(raw_path)
        if not path.exists():
            continue

        _DLL_HANDLES.append(os.add_dll_directory(str(path)))


_add_windows_dll_dirs()
