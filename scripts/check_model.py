"""Run the model quality gate on the committed model in models/.

What this is:
  The "model-gate" job in CI runs this before anything is deployed. It uses
  the same checks the training pipeline applies before promoting a model
  (src/quality_gate.py), so a model that couldn't be promoted can't be
  deployed either.

Usage:
  uv run python -m scripts.check_model

Exit code:
  0 if every check passes, 1 otherwise. CI treats a non-zero exit code as a
  failed step, which turns the job red and blocks merging into main.
"""

import sys

from src.quality_gate import check_saved_model


def main() -> int:
    report = check_saved_model()
    print(report.as_text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
