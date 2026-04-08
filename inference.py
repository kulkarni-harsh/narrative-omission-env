"""
Inference Script — Narrative Omission Detection Environment
===========================================================
Required environment variables:
    API_BASE_URL        LLM endpoint (default: HuggingFace router)
    MODEL_NAME          Model identifier
    HF_TOKEN / API_KEY  API key
    LOCAL_IMAGE_NAME    Docker image name for the environment server
    TASK_NAME           Task difficulty: easy | medium | hard (default: easy)

STDOUT FORMAT (strictly enforced by benchmark harness):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from models import NarrativeAction, NarrativeObservation
from client import NarrativeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = "narrative_omission"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Action string serialiser (for [STEP] logs)
# ---------------------------------------------------------------------------


def action_str(action: NarrativeAction) -> str:
    if action.action_type == "read_source":
        return f"read_source({action.source_id})"
    if action.action_type == "cross_reference":
        return f"cross_reference({action.source_id},{action.source_id_b})"
    if action.action_type == "identify_gap":
        return f"identify_gap({action.source_id},{action.claimed_omitted_field})"
    if action.action_type == "synthesize":
        payload = json.dumps(action.synthesis, separators=(",", ":"))
        return f"synthesize({payload})"
    return action.action_type


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an investigative analyst examining multiple news sources that cover the same event.
    Some sources deliberately omit critical facts. Your job is to cross-reference sources,
    detect structural gaps, and identify what is collectively hidden.

    Possible omittable fields:
      - responsible_party   Who caused or is accountable for the event
      - cover_up_detail     The key suppressed fact or internal knowledge
      - financial_impact    Economic damage or costs concealed
      - casualties          Human harm or death toll hidden

    Respond ONLY with valid JSON. No prose, no markdown fences.
    """
).strip()


def call_llm(client: OpenAI, messages: list) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0.2,
            max_tokens=512,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def ask_for_synthesis(
    client: OpenAI, task_name: str, source_texts: dict, obs: NarrativeObservation
) -> dict:
    """Ask the LLM to produce a synthesis dict given all gathered source content."""
    sources_block = "\n\n".join(
        f"=== {sid} ===\n{text}" for sid, text in source_texts.items()
    )
    has_red_herring = task_name == "hard"
    schema_note = (
        'Keys: "responsible_party" (str), "omitted_fields" (list[str]), '
        '"cover_up_detail" (str), "red_herring_source" (str — which source ID is a distraction).'
        if has_red_herring
        else 'Keys: "responsible_party" (str), "omitted_fields" (list[str]), "cover_up_detail" (str).'
    )

    prompt = textwrap.dedent(
        f"""
        Task difficulty: {task_name}
        Gaps correctly identified so far: {obs.gaps_correctly_identified}

        Here are all news sources:
        {sources_block}

        Produce a JSON synthesis object. {schema_note}
        Choose omitted_fields only from: responsible_party, cover_up_detail, financial_impact, casualties.
        """
    ).strip()

    raw = call_llm(client, [{"role": "user", "content": prompt}])
    try:
        return json.loads(raw)
    except Exception:
        print(f"[DEBUG] Synthesis JSON parse failed, raw: {raw[:200]}", flush=True)
        return {
            "responsible_party": "unknown",
            "omitted_fields": ["responsible_party"],
            "cover_up_detail": "",
        }


def ask_for_gaps(
    client: OpenAI, task_name: str, source_texts: dict, sources: List[str]
) -> List[tuple]:
    """Ask the LLM which (source_id, field) gaps to identify before synthesising."""
    sources_block = "\n\n".join(
        f"=== {sid} ===\n{text}" for sid, text in source_texts.items()
    )
    prompt = textwrap.dedent(
        f"""
        Task difficulty: {task_name}
        Sources: {sources}

        News source content:
        {sources_block}

        Which (source_id, field) pairs have deliberate omissions?
        Respond with a JSON array of objects: [{{"source_id": "src_0", "field": "responsible_party"}}, ...]
        Only include pairs you are confident about (max 3).
        Fields must be one of: responsible_party, cover_up_detail, financial_impact, casualties.
        """
    ).strip()

    raw = call_llm(client, [{"role": "user", "content": prompt}])
    try:
        pairs = json.loads(raw)
        return [(p["source_id"], p["field"]) for p in pairs if isinstance(p, dict)]
    except Exception:
        print(f"[DEBUG] Gap JSON parse failed: {raw[:200]}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await NarrativeEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=TASK_NAME)
        obs: NarrativeObservation = result.observation
        sources = list(obs.available_sources)
        source_texts: dict = {}
        step = 0

        # ------------------------------------------------------------------
        # Phase 1: Read every source
        # ------------------------------------------------------------------
        for src_id in sources:
            if result.done or step >= obs.max_steps:
                break
            step += 1
            act = NarrativeAction(action_type="read_source", source_id=src_id)
            result = await env.step(act)
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            source_texts[src_id] = obs.last_action_result
            log_step(step=step, action=action_str(act), reward=reward, done=result.done, error=None)
            if result.done:
                break

        # ------------------------------------------------------------------
        # Phase 2: Cross-reference all unique pairs
        # ------------------------------------------------------------------
        if not result.done:
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    if result.done or step >= obs.max_steps - 2:
                        break
                    step += 1
                    act = NarrativeAction(
                        action_type="cross_reference",
                        source_id=sources[i],
                        source_id_b=sources[j],
                    )
                    result = await env.step(act)
                    obs = result.observation
                    reward = result.reward or 0.0
                    rewards.append(reward)
                    steps_taken = step
                    log_step(
                        step=step, action=action_str(act), reward=reward, done=result.done, error=None
                    )
                    if result.done:
                        break

        # ------------------------------------------------------------------
        # Phase 3: Identify gaps (LLM-guided, budget-aware)
        # ------------------------------------------------------------------
        if not result.done:
            gap_budget = max(0, obs.max_steps - step - 1)  # reserve 1 step for synthesize
            if gap_budget > 0:
                gap_pairs = ask_for_gaps(client, TASK_NAME, source_texts, sources)
                for src_id, field in gap_pairs[:gap_budget]:
                    if result.done or step >= obs.max_steps - 1:
                        break
                    step += 1
                    act = NarrativeAction(
                        action_type="identify_gap",
                        source_id=src_id,
                        claimed_omitted_field=field,
                    )
                    result = await env.step(act)
                    obs = result.observation
                    reward = result.reward or 0.0
                    rewards.append(reward)
                    steps_taken = step
                    log_step(
                        step=step, action=action_str(act), reward=reward, done=result.done, error=None
                    )
                    if result.done:
                        break

        # ------------------------------------------------------------------
        # Phase 4: Synthesize
        # ------------------------------------------------------------------
        if not result.done:
            step += 1
            synthesis = ask_for_synthesis(client, TASK_NAME, source_texts, obs)
            act = NarrativeAction(action_type="synthesize", synthesis=synthesis)
            result = await env.step(act)
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step, action=action_str(act), reward=reward, done=result.done, error=None
            )

        # Final score is the last reward (synthesis grade from grade_task), not the cumulative sum.
        # Fallback to accumulated sum capped at 1.0 if synthesis was never reached.
        score = rewards[-1] if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Agent error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
