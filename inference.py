"""
inference.py — Narrative Omission Detection Baseline
Runs LLM agent against all 3 tasks.
Emits mandatory [START] [STEP] [END] stdout format.
"""

import json
import os

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:8000")

client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are a media analyst. You have access to synthetic news sources covering the same event.
Your goal is to detect deliberate omissions and identify the responsible party.

At each step output ONLY a JSON object with this schema:
{
  "action_type": "read_source" | "cross_reference" | "identify_gap" | "synthesize" | "skip",
  "source_id": "...",
  "source_id_b": "...",
  "claimed_omitted_field": "...",
  "synthesis": {
    "responsible_party": "...",
    "omitted_fields": [...],
    "cover_up_detail": "...",
    "red_herring_source": "..."
  }
}
Only include fields relevant to the chosen action_type.
No prose, no markdown. Pure JSON only."""


def get_agent_action(obs_text: str, history: list):
    history.append({"role": "user", "content": obs_text})
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        max_tokens=500,
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": raw})
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return {"action_type": "skip"}, raw


def run_task(task_name: str) -> float:
    from narrative_omission_env import NarrativeAction, NarrativeEnv

    print(f"[START] task={task_name} env=narrative_omission model={MODEL_NAME}", flush=True)

    rewards = []
    step = 0
    score = 0.0
    success = False

    try:
        with NarrativeEnv(base_url=SPACE_URL).sync() as env:
            result = env.reset(task_name=task_name)
            obs = result.observation
            history = []
            done = False

            while not done and step < obs.max_steps:
                obs_text = (
                    f"Available sources: {obs.available_sources}\n"
                    f"Sources read: {obs.sources_read}\n"
                    f"Last result: {obs.last_action_result}\n"
                    f"Steps remaining: {obs.max_steps - obs.step_count}\n"
                    f"Correct gaps so far: {obs.gaps_correctly_identified}\n"
                )
                action_dict, raw = get_agent_action(obs_text, history)
                action = NarrativeAction(**action_dict)
                result = env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                step += 1
                rewards.append(reward)

                safe_raw = raw[:80].replace("\n", " ")
                print(
                    f"[STEP] step={step} action={safe_raw} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null",
                    flush=True,
                )

            score = sum(rewards)
            success = score >= 0.5

    except Exception as e:
        print(
            f"[STEP] step={step + 1} action=error reward=0.00 done=true error={str(e)}",
            flush=True,
        )
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
