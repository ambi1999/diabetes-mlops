<!--
  PULL REQUEST TEMPLATE
  GitHub pre-fills every new pull request with this text. A pull request asks to
  merge a branch into main; CI runs on it, and main only accepts it once the
  required checks pass (branch protection). The checklist is a reminder of the
  project's standards. Delete lines that don't apply.
-->

## What does this change?

<!-- One or two sentences, in plain words. -->

## Why?

<!-- The problem it solves or the improvement it makes. -->

## Checklist

- [ ] `uv run pre-commit run --all-files` passes (formatting, lint, lockfile)
- [ ] `uv run pytest --cov` passes, coverage still at or above the minimum
- [ ] New or changed settings live in their proper home (see "one home per kind of setting" in the README)
- [ ] New code and config have plain-language comments
- [ ] If `models/` changed: `uv run python -m scripts.check_model` passes, and the demo baseline was updated if intended
- [ ] If dependencies changed: `uv.lock` and `ui/requirements.txt` are committed
- [ ] If a demo notebook changed: it was re-run as a handout (see the README)
- [ ] README updated if the way something works or is used changed

## How was it tested?

<!-- e.g. "ran the pipeline", "ran the smoke test against docker compose" -->
