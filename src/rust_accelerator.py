"""Load the dl_manager Rust accelerator without importing the full dl_manager package."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from config import DL_MANAGER_DIR


class RustAcceleratorNotBuiltError(RuntimeError):
    pass


def _find_extension_file(dl_manager_dir: Path) -> Path | None:
    # Prefer an extension file that exactly matches this interpreter's ABI
    # (e.g. "accelerator.cp312-win_amd64.pyd" on Windows/CPython 3.12,
    # "accelerator.cpython-312-x86_64-linux-gnu.so" on Linux).
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = dl_manager_dir / f"accelerator{suffix}"
        if candidate.exists():
            return candidate

    # Fallback: any accelerator extension we can find (may not match this
    # interpreter's ABI, but better than nothing if the pattern above misses).
    fallback_files = sorted(
        list(dl_manager_dir.glob("accelerator.cpython-*.so"))  # Linux / macOS
        + list(dl_manager_dir.glob("accelerator.cp*-*.pyd"))  # Windows
        + list(dl_manager_dir.glob("accelerator*.pyd"))  # Windows, any naming
    )
    return fallback_files[0] if fallback_files else None


@lru_cache(maxsize=1)
def get_accelerator() -> ModuleType:
    dl_manager_dir = DL_MANAGER_DIR / "dl_manager"
    extension_file = _find_extension_file(dl_manager_dir)
    if extension_file is None:
        raise RustAcceleratorNotBuiltError(
            "Rust accelerator not built. From the dl_manager directory "
            f"({dl_manager_dir}), run:\n"
            "  python setup.py build_ext --inplace"
        )

    module_name = "dl_manager.accelerator"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            extension_file,
        )
        if spec is None or spec.loader is None:
            raise RustAcceleratorNotBuiltError(
                f"Could not load Rust accelerator from {extension_file}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return sys.modules[module_name]