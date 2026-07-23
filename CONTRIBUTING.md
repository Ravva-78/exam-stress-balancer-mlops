# Contributing to Exam Stress Balancer

We welcome contributions! Whether it's tweaking the Reinforcement Learning rewards, adding new features to the API, or fixing bugs, your help is appreciated.

## Local Setup

1. Fork and clone the repository.
2. Set up your virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## Pre-Pull Request Evaluation (Critical!)

Before submitting a Pull Request that alters the API logic, schema, or RL pipeline, you **must** run the Reliability Evaluation script to ensure no regressions occurred.

```bash
python scripts/eval_endpoint.py
```
This script tests 10 diverse inputs (including edge cases like 0 hours studied, critical stress, and high urgency) against the `/api/explain` endpoint. **Your PR will not be merged if Schema Integrity or Logic Accuracy drops below 100%.**

## Reinforcement Learning Development

The core RL agents are located in `src/rl/agent/`.
If you are adding new state dimensions or altering the reward function, ensure you update:
1. `StudentEnvironment`: Modifies how rewards are calculated.
2. `state_discretizer.py`: Modifies how continuous state is binned.
3. The `compare_baselines` stage in `dvc.yaml` to ensure your new agent outperforms the random baseline.

## Branch Strategy

- `main` is our stable, deployable branch.
- For new features, branch off `main` using the format `feature/<your-feature>` or `fix/<your-fix>`.
- Keep PRs scoped to single, atomic changes.
