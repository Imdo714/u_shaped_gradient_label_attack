"""Compatibility entry point for the single-holdout reconstruction experiment."""

from .reconstruction.reconstruct_holdout_image import build_parser, main, run


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]

