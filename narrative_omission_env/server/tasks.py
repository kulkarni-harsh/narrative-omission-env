"""Task definitions and graders for the Narrative Omission environment."""

from typing import Any, Dict

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy": {
        "max_steps": 10,
        "num_sources": 3,
        "num_omitted_fields": 1,
        "has_red_herring": False,
    },
    "medium": {
        "max_steps": 15,
        "num_sources": 4,
        "num_omitted_fields": 3,
        "has_red_herring": False,
    },
    "hard": {
        "max_steps": 20,
        "num_sources": 4,
        "num_omitted_fields": 4,
        "has_red_herring": True,
    },
}


def grade_easy(synthesis: dict, ground_truth: Any) -> float:
    score = 0.0
    if (
        synthesis.get("responsible_party", "").lower().strip()
        == ground_truth.responsible_party.lower().strip()
    ):
        score += 0.6
    submitted = set(synthesis.get("omitted_fields", []))
    true_omissions = {"responsible_party"}
    overlap = submitted & true_omissions
    score += 0.4 * (len(overlap) / len(true_omissions))
    return min(score, 1.0)


def grade_medium(synthesis: dict, ground_truth: Any) -> float:
    score = 0.0
    true_omitted = {"responsible_party", "cover_up_detail", "financial_impact"}
    submitted = set(synthesis.get("omitted_fields", []))
    precision = len(submitted & true_omitted) / max(len(submitted), 1)
    recall = len(submitted & true_omitted) / len(true_omitted)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    score += 0.5 * f1
    if synthesis.get("responsible_party", "").lower() == ground_truth.responsible_party.lower():
        score += 0.3
    if ground_truth.cover_up_detail.lower() in synthesis.get("cover_up_detail", "").lower():
        score += 0.2
    return min(score, 1.0)


def grade_hard(synthesis: dict, ground_truth: Any, state: Any) -> float:
    score = 0.0
    true_omitted = {"responsible_party", "cover_up_detail", "financial_impact", "casualties"}
    submitted = set(synthesis.get("omitted_fields", []))
    precision = len(submitted & true_omitted) / max(len(submitted), 1)
    recall = len(submitted & true_omitted) / len(true_omitted)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    score += 0.4 * f1
    if synthesis.get("red_herring_source") == ground_truth.red_herring_source_id:
        score += 0.2
    if synthesis.get("responsible_party", "").lower() == ground_truth.responsible_party.lower():
        score += 0.2
    if ground_truth.cover_up_detail.lower() in synthesis.get("cover_up_detail", "").lower():
        score += 0.1
    efficiency = 1.0 - (state.step_count / state.max_steps)
    score += 0.1 * max(0.0, efficiency)
    return min(score, 1.0)


def grade_task(task_name: str, synthesis: dict, ground_truth: Any, state: Any) -> float:
    if task_name == "easy":
        return grade_easy(synthesis, ground_truth)
    elif task_name == "medium":
        return grade_medium(synthesis, ground_truth)
    elif task_name == "hard":
        return grade_hard(synthesis, ground_truth, state)
    return 0.0


def compute_reward(task_name: str, synthesis: dict, ground_truth: Any, state: Any) -> float:
    """Alias used by environment for reward computation."""
    return grade_task(task_name, synthesis, ground_truth, state)
