from __future__ import annotations

if __package__:
    from .cli import main
else:
    # Supports invocation as `python3 vio_py ...` from the `python/` directory.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vio_py.cli import main

raise SystemExit(main())
