"""Narrative Omission Detection Environment Implementation."""

import os
import uuid
from typing import Dict, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import NarrativeAction, NarrativeObservation, NarrativeState
    from .event_generator import EventGenerator
    from .tasks import TASK_CONFIG, grade_task
except ImportError:  # running as server.app (flat)
    from models import NarrativeAction, NarrativeObservation, NarrativeState
    from server.event_generator import EventGenerator
    from server.tasks import TASK_CONFIG, grade_task


class NarrativeEnvironment(Environment):
    """
    RL environment where agents detect deliberate narrative omissions
    across multiple synthetic news sources covering the same event.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "easy") -> None:
        super().__init__()
        self.task_name = task_name
        self._generator = EventGenerator(seed=42)
        self._state: Optional[NarrativeState] = None
        self._current_event = None
        self._articles: Dict[str, dict] = {}
        self._accumulated_reward = 0.0
        # Pre-initialize so step() never crashes on a fresh instance
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, task_name: Optional[str] = None, **kwargs) -> NarrativeObservation:  # type: ignore[override]
        if task_name:
            self.task_name = task_name
        config = TASK_CONFIG[self.task_name]
        self._current_event, self._articles = self._generator.sample(self.task_name)
        self._accumulated_reward = 0.0
        self._state = NarrativeState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self.task_name,
            max_steps=config["max_steps"],
            current_event_id=self._current_event.event_id,
            sources_read=[],
            gaps_found=[],
            accumulated_reward=0.0,
        )
        return NarrativeObservation(
            available_sources=list(self._articles.keys()),
            sources_read=[],
            last_action_result="Episode started. Available sources listed above.",
            partial_reward=0.0,
            gaps_correctly_identified=0,
            gaps_incorrectly_identified=0,
            step_count=0,
            max_steps=config["max_steps"],
            budget_remaining=1.0,
            done=False,
            reward=0.0,
            task_name=self.task_name,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: NarrativeAction, **kwargs) -> NarrativeObservation:  # type: ignore[override]
        self._state.step_count += 1
        step_reward = 0.0
        result_text = ""
        done = False
        synthesis_feedback = None

        if action.action_type == "read_source":
            step_reward, result_text = self._handle_read(action)
        elif action.action_type == "cross_reference":
            step_reward, result_text = self._handle_cross_reference(action)
        elif action.action_type == "identify_gap":
            step_reward, result_text = self._handle_identify_gap(action)
        elif action.action_type == "synthesize":
            step_reward, result_text, synthesis_feedback, done = self._handle_synthesize(action)
        elif action.action_type == "skip":
            step_reward = -0.02
            result_text = "Skipped. No action taken."

        # Efficiency decay after step 12
        if self._state.step_count > 12:
            step_reward -= 0.01

        self._accumulated_reward += step_reward
        self._state.accumulated_reward = self._accumulated_reward

        if self._state.step_count >= self._state.max_steps:
            done = True
        self._state.done = done if hasattr(self._state, "done") else done

        return NarrativeObservation(
            available_sources=list(self._articles.keys()),
            sources_read=list(self._state.sources_read),
            last_action_result=result_text,
            partial_reward=step_reward,
            gaps_correctly_identified=len(self._state.gaps_found),
            gaps_incorrectly_identified=0,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            budget_remaining=max(0.0, 1.0 - self._state.step_count / self._state.max_steps),
            done=done,
            reward=step_reward,
            task_name=self.task_name,
            synthesis_feedback=synthesis_feedback,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read(self, action: NarrativeAction):
        src_id = action.source_id
        if src_id not in self._articles:
            return -0.05, f"Source '{src_id}' not found. Available: {list(self._articles.keys())}"
        article = self._articles[src_id]
        if src_id in self._state.sources_read:
            return 0.0, f"[Already read] {article['text']}"
        self._state.sources_read.append(src_id)
        return 0.05, f"[{article['source_name']} — {article['framing']}]\n{article['text']}"

    def _handle_cross_reference(self, action: NarrativeAction):
        a, b = action.source_id, action.source_id_b
        if a not in self._state.sources_read or b not in self._state.sources_read:
            return -0.02, "You must read both sources before cross-referencing."
        art_a = self._articles[a]
        art_b = self._articles[b]
        diffs = []
        for field in ["responsible_party", "cover_up_detail", "financial_impact", "casualties", "why"]:
            val_a = art_a["fields"].get(field)
            val_b = art_b["fields"].get(field)
            if val_a and not val_b:
                diffs.append(f"Field '{field}': present in {a}, ABSENT in {b}")
            elif val_b and not val_a:
                diffs.append(f"Field '{field}': present in {b}, ABSENT in {a}")
        real_gap = len(diffs) > 0
        reward = 0.10 if real_gap else 0.0
        result = (
            "Comparison:\n" + "\n".join(diffs)
            if diffs
            else "No structural gaps found between these two sources."
        )
        return reward, result

    def _handle_identify_gap(self, action: NarrativeAction):
        src_id = action.source_id
        field = action.claimed_omitted_field
        if src_id not in self._articles:
            return -0.05, f"Unknown source: {src_id}"
        article = self._articles[src_id]
        true_omissions = set(article["omitted_fields"])
        if field in true_omissions:
            key = f"{src_id}:{field}"
            if key not in self._state.gaps_found:
                self._state.gaps_found.append(key)
            return 0.20, f"Correct. '{src_id}' does omit '{field}'."
        return -0.10, f"Incorrect. '{src_id}' does not omit '{field}'."

    def _handle_synthesize(self, action: NarrativeAction):
        synthesis = action.synthesis or {}
        final_score = grade_task(
            self.task_name, synthesis, self._current_event, self._state
        )
        feedback = (
            f"Final score: {final_score:.2f}\n"
            f"Responsible party correct: "
            f"{synthesis.get('responsible_party', '').lower() == self._current_event.responsible_party.lower()}\n"
            f"Cover up detail found: "
            f"{self._current_event.cover_up_detail.lower() in synthesis.get('cover_up_detail', '').lower()}"
        )
        return final_score, "Synthesis submitted.", feedback, True

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> NarrativeState:
        return self._state
