# COMSOL Assets

Place local `.mph` models here during development; binary models are ignored by Git by default. Commit human-readable model cards and exported result schemas.

Suggested structure:

```text
comsol/
  model_cards/
  exports/
  scripts/
  local_models/   # ignored binaries
```

Use `battery_lab.comsol_bridge.ComsolSession` for deterministic automation.
