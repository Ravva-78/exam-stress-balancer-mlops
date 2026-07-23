# Changelog

## [v3.1.0] - 2026-07-23

### Added
- **Dual Schema Support**: The `/api/predict` and `/api/explain` endpoints now natively support the client's payload structure (`stress_level`, `hours_studied`, `days_until_exam`, `current_performance`) in addition to the internal ML schema. Incoming client fields are dynamically mapped via Pydantic model validation.
- **Critical Stress Safety Override**: Introduced a hard safety boundary in the `hybrid_predict` engine. If a student's stress level is critical (>=95), the system ignores the RL models entirely and strictly recommends `Sleep/Break`. This prevents "Performance mode" from dangerously suggesting intense study during burnout.

### Changed
- **Explainability Engine Update**: The `build_explanation` function now explicitly documents when the critical stress safety override triggers, ensuring complete AI transparency.
