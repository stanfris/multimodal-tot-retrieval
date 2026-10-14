# Coding philosophy

Prefer the smallest correct implementation that is readable and maintainable.

- Validate user input, configuration, external data, and filesystem/artifact safety
  at their boundaries with explicit exceptions.
- Trust internal contracts; use assertions for programmer errors and invariants.
  Assertions may be disabled with Python `-O`, so do not use them for safety checks.
- Let library exceptions propagate unless handling adds useful context or cleanup.
- Reuse existing functionality and dependencies; avoid duplicated logic and
  wrappers or abstractions without a concrete purpose.
- Let Hydra/OmegaConf own composition, overrides, and configuration resolution.
  Keep application checks for supported values and actual execution constraints.
- Avoid speculative features, compatibility layers, and fallback behavior that
  conceals errors. Preserve dataset formats, YAML semantics, output layout,
  retrieval behavior, and evaluation behavior.
- Review the full change for redundant checks and obsolete code, simplify it,
  and run relevant tests and checks before finishing.
