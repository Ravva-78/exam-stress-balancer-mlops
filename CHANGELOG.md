# Changelog

## [Unreleased]

### Added
- GitHub repository polish: MIT License, Render deployment blueprint (`render.yaml`), standardized issue and pull request templates, and CD workflow.

### Fixed
- Non-string `stress_level` crashed the API with a 500; invalid values (and unknown strings such as `"banana"`) now return 422 instead of silently defaulting to medium stress.
- `difficulty` is now validated (`easy|medium|hard`); unknown values previously missed the Q-table and silently scored 0.
- Home page and `/predict` crashed on current Starlette (legacy `TemplateResponse` signature).
- Removed the broken `/simulate` stub; `GET /api/history` now bounds `limit` (1-50).
- `/api/*` responses no longer echo null client-schema fields in `input_state`; the request payload is no longer mutated by the validator.
- Replaced deprecated `@app.on_event("startup")` with a lifespan handler; reported version is now consistent (3.1.0).
- `docs/architecture.md` now matches the real discretizer bins and weighting thresholds.

### Changed
- CI runs the full test suite (previously only `test_wellness.py`; 24 tests were silently failing and have been rewritten for the current API).
- Removed the stale duplicate `docker/` directory (its Dockerfile omitted `templates/` and `static/` and referenced a missing `logs/` dir).
- `data/session_history.json` is no longer tracked in git.

## [v3.1.0] - 2026-07-23

### Added
- **Dual Schema Support**: The `/api/predict` and `/api/explain` endpoints now natively support the client's payload structure (`stress_level`, `hours_studied`, `days_until_exam`, `current_performance`) in addition to the internal ML schema. Incoming client fields are dynamically mapped via Pydantic model validation.
- **Critical Stress Safety Override**: Introduced a hard safety boundary in the `hybrid_predict` engine. If a student's stress level is critical (>=95), the system ignores the RL models entirely and strictly recommends `Sleep/Break`. This prevents "Performance mode" from dangerously suggesting intense study during burnout.

### Changed
- **Explainability Engine Update**: The `build_explanation` function now explicitly documents when the critical stress safety override triggers, ensuring complete AI transparency.
