"""Data models for the Narrative Omission Detection Environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class NarrativeAction(Action):
    """Action for the Narrative Omission environment."""

    action_type: Literal[
        "read_source",
        "cross_reference",
        "identify_gap",
        "synthesize",
        "skip",
    ] = Field(..., description="Type of action to take")
    source_id: Optional[str] = Field(None, description="Source to read or identify gap in")
    source_id_b: Optional[str] = Field(None, description="Second source for cross_reference")
    claimed_omitted_field: Optional[str] = Field(
        None, description="Field claimed to be omitted (for identify_gap)"
    )
    synthesis: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Final synthesis dict with keys: responsible_party, omitted_fields, "
            "cover_up_detail, red_herring_source (hard only)"
        ),
    )


class NarrativeObservation(Observation):
    """Observation from the Narrative Omission environment."""

    available_sources: List[str] = Field(
        default_factory=list, description="Source IDs available this episode"
    )
    sources_read: List[str] = Field(
        default_factory=list, description="Source IDs the agent has already read"
    )
    last_action_result: str = Field(
        default="", description="Text result of the last action"
    )
    partial_reward: float = Field(default=0.0, description="Reward earned this step")
    gaps_correctly_identified: int = Field(
        default=0, description="Running count of correct identify_gap calls"
    )
    gaps_incorrectly_identified: int = Field(
        default=0, description="Running count of incorrect identify_gap calls"
    )
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=10, description="Maximum steps for this task")
    budget_remaining: float = Field(
        default=1.0, description="Normalized budget remaining (0–1)"
    )
    task_name: str = Field(default="easy", description="Current task difficulty")
    synthesis_feedback: Optional[str] = Field(
        None, description="Feedback from synthesize action (only after synthesize)"
    )


class NarrativeState(State):
    """Internal state for the Narrative Omission environment."""

    task_name: str = Field(default="easy")
    max_steps: int = Field(default=10)
    current_event_id: str = Field(default="")
    sources_read: List[str] = Field(default_factory=list)
    gaps_found: List[str] = Field(default_factory=list)
    accumulated_reward: float = Field(default=0.0)
