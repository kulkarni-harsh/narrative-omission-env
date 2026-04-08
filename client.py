"""Narrative Omission Detection Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import NarrativeAction, NarrativeObservation, NarrativeState


class NarrativeEnv(EnvClient[NarrativeAction, NarrativeObservation, NarrativeState]):
    """
    Client for the Narrative Omission Detection Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example (sync):
        >>> with NarrativeEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_name="easy")
        ...     obs = result.observation
        ...     result = env.step(NarrativeAction(action_type="read_source", source_id="src_0"))
    """

    def _step_payload(self, action: NarrativeAction) -> Dict:
        return action.model_dump(exclude={"metadata"}, exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[NarrativeObservation]:
        obs_data = payload.get("observation", {})
        observation = NarrativeObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> NarrativeState:
        return NarrativeState(**payload)
