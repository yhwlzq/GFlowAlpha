import os

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "no")
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPEX", "no")
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_FUNCTORCH", "no")

import juliacall  # noqa: E402,F401 — 必须在 torch 之前加载, 避免 segfault
