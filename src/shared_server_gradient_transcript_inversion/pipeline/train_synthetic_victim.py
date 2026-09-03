from __future__ import annotations

from .train_letter_victim import build_parser as _base_parser
from .train_letter_victim import run


def build_parser():
    parser = _base_parser()
    parser.description = (
        "Train a synthetic-document victim f-g-h model without horizontal flips."
    )
    parser.set_defaults(
        data="workspace/data/synthetic_document_experiment/victim",
        output=(
            "workspace/results/shared_server_gradient_transcript_inversion/"
            "document_c/victim_model"
        ),
        batch_size=16,
        image_size=256,
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]

