# Narrative Omission Detection Environment

An RL environment where agents learn to detect **deliberate narrative omissions** across multiple synthetic news sources covering the same event. Agents must read, cross-reference, and identify structural gaps — then synthesize a complete picture of what biased sources collectively hide.

## Quick Start (Python)

```python
from narrative_omission_env import NarrativeAction, NarrativeOmissionEnv

env = NarrativeOmissionEnv(base_url="<SPACE_URL>")
obs = env.reset()
print(obs.available_sources)  # ['src_0', 'src_1', 'src_2']

# Read all sources
for src in obs.available_sources:
    result = env.step(NarrativeAction(action_type="read_source", source_id=src))

# Cross-reference two sources
result = env.step(NarrativeAction(
    action_type="cross_reference",
    source_id="src_0",
    source_id_b="src_1",
))

# Identify a gap
result = env.step(NarrativeAction(
    action_type="identify_gap",
    source_id="src_0",
    claimed_omitted_field="responsible_party",
))

# Submit final analysis
result = env.step(NarrativeAction(
    action_type="synthesize",
    synthesis={
        "responsible_party": "NovaChem Industries",
        "omitted_fields": ["responsible_party", "cover_up_detail"],
        "cover_up_detail": "Internal memos show management knew about the corroded valves",
    },
))
print(result.observation.synthesis_feedback)
```

## Action Space

**NarrativeAction** fields:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | Literal | One of: `read_source`, `cross_reference`, `identify_gap`, `synthesize`, `skip` |
| `source_id` | str \| None | Source to read, cross-reference, or identify gap in |
| `source_id_b` | str \| None | Second source for `cross_reference` |
| `claimed_omitted_field` | str \| None | Field claimed omitted (for `identify_gap`) |
| `synthesis` | dict \| None | Final analysis dict (for `synthesize`) |

**Synthesis dict keys:**
- `responsible_party` (str) — who is responsible
- `omitted_fields` (list[str]) — fields hidden across sources
- `cover_up_detail` (str) — the key suppressed fact
- `red_herring_source` (str, hard only) — which source was a distraction

## Observation Space

**NarrativeObservation** fields:

| Field | Type | Description |
|-------|------|-------------|
| `available_sources` | list[str] | Source IDs in this episode |
| `sources_read` | list[str] | Sources the agent has read |
| `last_action_result` | str | Text feedback from last action |
| `step_count` | int | Current step number |
| `max_steps` | int | Step budget for this task |
| `budget_remaining` | float | Normalized budget (0–1) |
| `gaps_correctly_identified` | int | Correct `identify_gap` calls so far |
| `done` | bool | Episode complete? |
| `reward` | float | Reward from last step |
| `synthesis_feedback` | str \| None | Final scoring breakdown |

## Reward Structure

| Action | Reward |
|--------|--------|
| Read new source | +0.05 |
| Cross-reference (gap found) | +0.10 |
| Identify gap correctly | +0.20 |
| Identify gap incorrectly | −0.10 |
| Skip | −0.02 |
| Synthesize | up to 1.0 |
| Efficiency decay (step > 12) | −0.01/step |

## Task Difficulties

| Task | Sources | Max Steps | Notes |
|------|---------|-----------|-------|
| `easy` | 3 | 10 | One source omits `responsible_party` |
| `medium` | 4 | 15 | Multi-field omissions, misleading credibility signals |
| `hard` | 4 | 20 | One source is a red herring distraction |

Set via environment variable: `TASK_NAME=medium`

## About the News Data

Events are **synthetically generated** from 20 curated templates (industrial accidents, political decisions, corporate actions, public health incidents) combined with 6 bias profiles. A pool of 200 deterministic events is built at startup (seed=42). No external APIs are used — the environment is fully self-contained and reproducible.

## Web Interface

The Space includes an interactive **Investigative Analyst** playground at `/web`:
- Step-by-step guided workflow with tabs for each action type
- Formatted article cards per source
- Contextual dropdowns (source IDs, omittable fields)
- Progress tracker and scoring display
- Synthesis form with checkboxes for omitted fields

## API Endpoints

- `POST /reset` — start a new episode
- `POST /step` — take an action
- `GET /state` — get current state
- `GET /health` — health check
- `GET /docs` — OpenAPI documentation
- `WS /ws` — WebSocket for low-latency sessions
