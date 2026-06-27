"""Load the dl_manager Rust accelerator without importing the full dl_manager package."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from config import DL_MANAGER_DIR


class RustAcceleratorNotBuiltError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_accelerator() -> ModuleType:
    dl_manager_dir = DL_MANAGER_DIR / "dl_manager"
    extension_files = sorted(dl_manager_dir.glob("accelerator.cpython-*.so"))
    if not extension_files:
        raise RustAcceleratorNotBuiltError(
            "Rust accelerator not built. Run:\n"
            "  ./scripts/build_dl_manager.sh"
        )

    module_name = "dl_manager.accelerator"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            extension_files[0],
        )
        if spec is None or spec.loader is None:
            raise RustAcceleratorNotBuiltError(
                f"Could not load Rust accelerator from {extension_files[0]}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return sys.modules[module_name]
