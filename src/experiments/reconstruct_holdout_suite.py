"""Compatibility entry point for the multi-holdout ablation suite."""

from .reconstruction.reconstruct_holdout_suite import CONDITIONS, build_parser, main, run


if __name__ == "__main__":
    main()


__all__ = ["CONDITIONS", "build_parser", "run"]
