"""Compatibility entry point; prefer ``python -m aoi_v2x_reproduction``."""

from aoi_v2x_reproduction.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
