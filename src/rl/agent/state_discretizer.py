def discretize_state(state):
    """
    Convert continuous state into discrete bins
    """

    # Fatigue (0–100)
    if state["fatigue"] < 33:
        fatigue_level = "LOW"
    elif state["fatigue"] < 66:
        fatigue_level = "MEDIUM"
    else:
        fatigue_level = "HIGH"

    # Stress (0–100)
    if state["stress"] < 33:
        stress_level = "LOW"
    elif state["stress"] < 66:
        stress_level = "MEDIUM"
    else:
        stress_level = "HIGH"

    # Retention (0–1)
    if state["retention"] < 0.33:
        retention_level = "LOW"
    elif state["retention"] < 0.66:
        retention_level = "MEDIUM"
    else:
        retention_level = "HIGH"

    # Days Left
    if state["days_left"] <= 2:
        urgency = "HIGH"
    elif state["days_left"] <= 5:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    # Difficulty
    difficulty = state["difficulty"].upper()

    return (
        fatigue_level,
        stress_level,
        retention_level,
        urgency,
        difficulty
    )
