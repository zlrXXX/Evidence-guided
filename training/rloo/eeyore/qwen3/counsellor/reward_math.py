from __future__ import annotations
from typing import Mapping
RESISTANCE_UTILITY = {'REJECTING_DISENGAGED': {'STAY': 0.25, 'BETTER': 0.75}, 'AMBIVALENT': {'WORSE': -1.0, 'STAY': 0.2, 'BETTER': 1.0}, 'COOPERATIVE': {'WORSE': -1.0, 'STAY': 0.5, 'BETTER': 0.75}, 'COLLABORATIVE_EXPLORATION': {'WORSE': -1.0, 'STAY': 1.0}}

def expected_resistance_utility(current_state: str, probabilities: Mapping[str, float]) -> float:
    if current_state not in RESISTANCE_UTILITY:
        raise ValueError(f'unknown resistance state: {current_state}')
    utilities = RESISTANCE_UTILITY[current_state]
    unknown = set(probabilities) - {'WORSE', 'STAY', 'BETTER'}
    if unknown:
        raise ValueError(f'unknown resistance directions: {sorted(unknown)}')
    return sum((float(probabilities.get(direction, 0.0)) * utility for direction, utility in utilities.items()))

def positive_transition_gate(framework_reward: float, empathy_reward: float) -> float:
    """Map normalized [-1, 1] judge rewards to [0, 1] and use the weaker gate."""
    framework_gate = min(1.0, max(0.0, (float(framework_reward) + 1.0) / 2.0))
    empathy_gate = min(1.0, max(0.0, (float(empathy_reward) + 1.0) / 2.0))
    return min(framework_gate, empathy_gate)

def gated_resistance_reward(raw_utility: float, framework_reward: float, empathy_reward: float) -> tuple[float, float]:
    """Keep negative transition utility intact; gate only positive transition credit."""
    raw = float(raw_utility)
    gate = positive_transition_gate(framework_reward, empathy_reward)
    return (min(raw, 0.0) + gate * max(raw, 0.0), gate)
