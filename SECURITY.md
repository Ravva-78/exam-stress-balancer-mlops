# Security Policy

## Critical Stress Safety Feature

The Exam Stress Balancer includes a hard-coded **Safety Override Boundary**. 

Because this application provides actionable advice to students who may be undergoing severe psychological stress, we cannot rely solely on the statistical outputs of the Reinforcement Learning agents. 

If a student's stress level is determined to be **Critical** (Internal `stress` >= 95, or Client `stress_level` == "critical"):
1. The AI immediately halts all RL processing.
2. The recommended action is unconditionally forced to `Sleep/Break` (Action 2).
3. The explanation payload flags this event to the user: *"Safety override active: Critical stress (>=95) detected — ignoring RL model and forcing immediate rest."*

## Input Validation

All incoming requests to the API are strictly validated using Pydantic `v2`. 
- We employ a **Dual Schema Validator** that safely parses both the legacy/client format (`stress_level`, `hours_studied`) and the internal ML format (`fatigue`, `stress`).
- Inputs are clamped to strict bounds (e.g., `retention` between `0.0` and `1.0`). Out-of-bound or malformed inputs return a `422 Unprocessable Entity` immediately.

## Rate Limiting

Rate limiting is **currently not implemented natively** within the FastAPI application.
For production deployments (like Render), it is highly recommended to place this API behind a reverse proxy (e.g., Cloudflare, NGINX) or an API Gateway that handles IP-based rate limiting to prevent abuse.

## Reporting a Vulnerability

If you discover a security vulnerability or a dangerous edge case in the RL logic (e.g., the system recommending intense study for a highly fatigued individual), please do **NOT** create a public GitHub issue.

Please report it directly via email to the repository maintainer at **security@examstressbalancer.com** or via Twitter DM at **@examstressAI**. We aim to review and patch logical vulnerabilities within 48 hours.
