## Description

<!-- Briefly describe what this PR does and why. -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behaviour)
- [ ] Documentation / comments only
- [ ] Build / CI / dependency change
- [ ] Refactor (no behaviour change)

## How has this been tested?

<!-- List test files and the command used. -->

```bash
uv run pytest -m "not gpu"
```

## Checklist

- [ ] Tests pass locally: `uv run pytest -m "not gpu"` (no GPU) or `uv run pytest tests/test_v2.py` (with GPU)
- [ ] I have added or updated tests to cover my changes (where applicable).
- [ ] I have added or updated docstrings for any changed public functions.

## Screenshots / output (if relevant)

<!-- Paste terminal output or screenshots here. -->
