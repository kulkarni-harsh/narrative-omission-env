"""Custom Gradio UI for the Narrative Omission Detection Environment."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

FIELD_CHOICES = [
    "responsible_party",
    "cover_up_detail",
    "financial_impact",
    "casualties",
    "why",
]

ACTION_TYPES = [
    "read_source",
    "cross_reference",
    "identify_gap",
    "synthesize",
    "skip",
]

HOW_TO_PLAY = """
## How to Play

You are an **investigative analyst**. Multiple news outlets have covered the same event — but each omits key facts to protect certain interests.

**Your goal:** Read across sources, find the gaps, identify the responsible party, and submit a complete synthesis.

### Step-by-step workflow

1. **Read sources** — Click "Read Source" and pick a source ID to read that outlet's article
2. **Cross-reference** — Compare two sources you've read to spot structural gaps
3. **Identify gaps** — Call out which specific field a source omits (rewards +0.20 each)
4. **Synthesize** — Submit your final analysis with the responsible party, omitted fields, and cover-up detail

### Scoring
| Action | Reward |
|--------|--------|
| Read new source | +0.05 |
| Cross-reference (gap found) | +0.10 |
| Identify gap correctly | +0.20 |
| Identify gap incorrectly | -0.10 |
| Final synthesis | up to 1.0 |
| Efficiency decay (step > 12) | -0.01/step |

### Omittable fields
`responsible_party` · `cover_up_detail` · `financial_impact` · `casualties` · `why`
"""

DIFFICULTY_TIPS = {
    "easy": "3 sources, 10 steps. One source omits the responsible party.",
    "medium": "4 sources, 15 steps. Multiple fields omitted with misleading credibility signals.",
    "hard": "4 sources, 20 steps. One source is a red herring — identify which one!",
}


def _parse_obs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract observation fields from a reset/step response."""
    obs = data.get("observation", {}) or {}
    return {
        "available_sources": obs.get("available_sources", []),
        "sources_read": obs.get("sources_read", []),
        "last_action_result": obs.get("last_action_result", ""),
        "step_count": obs.get("step_count", 0),
        "max_steps": obs.get("max_steps", 10),
        "budget_remaining": obs.get("budget_remaining", 1.0),
        "gaps_correctly_identified": obs.get("gaps_correctly_identified", 0),
        "done": obs.get("done", False) or data.get("done", False),
        "reward": data.get("reward", 0.0),
        "task_name": obs.get("task_name", "easy"),
        "synthesis_feedback": obs.get("synthesis_feedback"),
    }


def _progress_md(obs: Dict[str, Any]) -> str:
    steps = obs["step_count"]
    max_steps = obs["max_steps"]
    budget = obs["budget_remaining"]
    gaps = obs["gaps_correctly_identified"]
    task = obs["task_name"]
    bar_fill = int(budget * 20)
    bar = "█" * bar_fill + "░" * (20 - bar_fill)
    tip = DIFFICULTY_TIPS.get(task, "")
    done_badge = " 🏁 **Episode complete**" if obs["done"] else ""
    return (
        f"**Task:** `{task}`  {done_badge}\n\n"
        f"{tip}\n\n"
        f"**Steps:** {steps} / {max_steps}  |  "
        f"**Budget:** `{bar}` {budget:.0%}  |  "
        f"**Gaps found:** {gaps}"
    )


def _sources_md(obs: Dict[str, Any]) -> str:
    available = obs["available_sources"]
    read = obs["sources_read"]
    if not available:
        return "*No episode started. Click Reset to begin.*"
    lines = ["**Available sources:**\n"]
    for src in available:
        status = "✅ read" if src in read else "📰 unread"
        lines.append(f"- `{src}` — {status}")
    return "\n".join(lines)


def _result_md(obs: Dict[str, Any]) -> str:
    result = obs["last_action_result"]
    reward = obs["reward"]
    if not result:
        return "*Click Reset to start an episode, then take actions.*"
    sign = "+" if reward >= 0 else ""
    feedback = obs.get("synthesis_feedback") or ""
    out = f"**Last result** (reward: {sign}{reward:.2f}):\n\n{result}"
    if feedback:
        out += f"\n\n---\n\n**Synthesis feedback:**\n\n{feedback}"
    return out


def build_narrative_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str = "Narrative Omission Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Custom Gradio UI for the Narrative Omission environment."""

    with gr.Blocks(
        title="Narrative Omission — Investigative Analyst",
        theme=gr.themes.Soft(),
        css="""
            .source-card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 4px 0; }
            .done-banner { background: #d4edda; border-radius: 8px; padding: 12px; }
        """,
    ) as demo:
        gr.Markdown("# 🔍 Narrative Omission — Investigative Analyst")
        gr.Markdown(
            "Read across biased news sources, find what they collectively hide, "
            "and submit your analysis. Each source omits different facts."
        )

        with gr.Row():
            # ── Left panel: instructions + progress + sources ──────────────
            with gr.Column(scale=1):
                with gr.Accordion("How to Play", open=True):
                    gr.Markdown(HOW_TO_PLAY)

                progress_display = gr.Markdown("*No episode started.*")
                sources_display = gr.Markdown("*No episode started.*")

            # ── Right panel: actions + output ──────────────────────────────
            with gr.Column(scale=2):
                result_display = gr.Markdown(
                    "*Click **Reset** to start a new episode.*"
                )

                gr.Markdown("---")
                gr.Markdown("### Actions")

                with gr.Tabs() as action_tabs:

                    # ── Tab 1: Read Source ─────────────────────────────────
                    with gr.TabItem("📖 Read Source"):
                        gr.Markdown(
                            "Read a news article from one source. "
                            "You earn **+0.05** for each new source read."
                        )
                        read_source_dd = gr.Dropdown(
                            choices=[],
                            label="Source ID",
                            info="Select a source to read",
                            allow_custom_value=True,
                        )
                        read_btn = gr.Button("Read Source", variant="primary")

                    # ── Tab 2: Cross-Reference ─────────────────────────────
                    with gr.TabItem("🔄 Cross-Reference"):
                        gr.Markdown(
                            "Compare two sources you've already read. "
                            "Earn **+0.10** when a structural gap is found."
                        )
                        xref_src_a = gr.Dropdown(
                            choices=[],
                            label="Source A",
                            allow_custom_value=True,
                        )
                        xref_src_b = gr.Dropdown(
                            choices=[],
                            label="Source B",
                            allow_custom_value=True,
                        )
                        xref_btn = gr.Button("Cross-Reference", variant="primary")

                    # ── Tab 3: Identify Gap ────────────────────────────────
                    with gr.TabItem("🎯 Identify Gap"):
                        gr.Markdown(
                            "Claim that a source omits a specific field. "
                            "**+0.20** if correct, **-0.10** if wrong."
                        )
                        gap_source_dd = gr.Dropdown(
                            choices=[],
                            label="Source ID",
                            info="Which source are you calling out?",
                            allow_custom_value=True,
                        )
                        gap_field_dd = gr.Dropdown(
                            choices=FIELD_CHOICES,
                            label="Omitted field",
                            info="Which field does this source omit?",
                        )
                        gap_btn = gr.Button("Identify Gap", variant="primary")

                    # ── Tab 4: Synthesize ──────────────────────────────────
                    with gr.TabItem("📝 Synthesize"):
                        gr.Markdown(
                            "Submit your final analysis. This ends the episode. "
                            "Fill in everything you've discovered."
                        )
                        synth_responsible = gr.Textbox(
                            label="Responsible party",
                            placeholder="e.g. NovaChem Industries",
                            info="Who is responsible for the incident?",
                        )
                        synth_omitted = gr.CheckboxGroup(
                            choices=FIELD_CHOICES,
                            label="Fields collectively omitted across sources",
                            info="Which fields are hidden by at least one source?",
                        )
                        synth_cover_up = gr.Textbox(
                            label="Cover-up detail",
                            placeholder="Describe the key cover-up detail you found...",
                            lines=3,
                            info="The critical hidden fact that sources are suppressing",
                        )
                        synth_red_herring = gr.Dropdown(
                            choices=[],
                            label="Red herring source (hard mode only)",
                            info="Which source was a deliberate distraction? Leave blank if not applicable.",
                            allow_custom_value=True,
                        )
                        synth_btn = gr.Button(
                            "Submit Synthesis", variant="stop"
                        )

                    # ── Tab 5: Skip ────────────────────────────────────────
                    with gr.TabItem("⏭ Skip"):
                        gr.Markdown(
                            "Skip this step (reward: **-0.02**). "
                            "Only use if you're stuck."
                        )
                        skip_btn = gr.Button("Skip Step", variant="secondary")

                gr.Markdown("---")
                reset_btn = gr.Button("🔄 Reset Episode", variant="secondary", size="lg")

        # ── Helper: refresh dropdowns after state change ───────────────────
        def _update_dropdowns(available: List[str], read: List[str]):
            return (
                gr.update(choices=available),
                gr.update(choices=read),
                gr.update(choices=read),
                gr.update(choices=available),
                gr.update(choices=available),
            )

        # ── Reset ──────────────────────────────────────────────────────────
        async def on_reset():
            try:
                data = await web_manager.reset_environment()
                obs = _parse_obs(data)
                available = obs["available_sources"]
                read = obs["sources_read"]
                return (
                    _progress_md(obs),
                    _sources_md(obs),
                    _result_md(obs),
                    gr.update(choices=available),
                    gr.update(choices=read),
                    gr.update(choices=read),
                    gr.update(choices=available),
                    gr.update(choices=available),
                )
            except Exception as e:
                err = f"❌ Reset failed: {e}"
                no_update = gr.update()
                return err, err, err, no_update, no_update, no_update, no_update, no_update

        reset_btn.click(
            fn=on_reset,
            outputs=[
                progress_display,
                sources_display,
                result_display,
                read_source_dd,
                xref_src_a,
                xref_src_b,
                gap_source_dd,
                synth_red_herring,
            ],
        )

        # ── Generic step handler ───────────────────────────────────────────
        async def _do_step(action_data: Dict[str, Any]):
            try:
                data = await web_manager.step_environment(action_data)
                obs = _parse_obs(data)
                available = obs["available_sources"]
                read = obs["sources_read"]
                return (
                    _progress_md(obs),
                    _sources_md(obs),
                    _result_md(obs),
                    gr.update(choices=available),
                    gr.update(choices=read),
                    gr.update(choices=read),
                    gr.update(choices=available),
                    gr.update(choices=available),
                )
            except Exception as e:
                err = f"❌ Step failed: {e}"
                no_update = gr.update()
                return err, err, err, no_update, no_update, no_update, no_update, no_update

        STEP_OUTPUTS = [
            progress_display,
            sources_display,
            result_display,
            read_source_dd,
            xref_src_a,
            xref_src_b,
            gap_source_dd,
            synth_red_herring,
        ]

        # ── Read Source ────────────────────────────────────────────────────
        async def on_read(source_id: str):
            if not source_id:
                err = "⚠️ Please select a source to read."
                no = gr.update()
                return err, err, err, no, no, no, no, no
            return await _do_step({"action_type": "read_source", "source_id": source_id})

        read_btn.click(fn=on_read, inputs=[read_source_dd], outputs=STEP_OUTPUTS)

        # ── Cross-Reference ────────────────────────────────────────────────
        async def on_xref(src_a: str, src_b: str):
            if not src_a or not src_b:
                err = "⚠️ Please select both sources."
                no = gr.update()
                return err, err, err, no, no, no, no, no
            return await _do_step(
                {"action_type": "cross_reference", "source_id": src_a, "source_id_b": src_b}
            )

        xref_btn.click(fn=on_xref, inputs=[xref_src_a, xref_src_b], outputs=STEP_OUTPUTS)

        # ── Identify Gap ───────────────────────────────────────────────────
        async def on_identify_gap(source_id: str, field: str):
            if not source_id or not field:
                err = "⚠️ Please select a source and a field."
                no = gr.update()
                return err, err, err, no, no, no, no, no
            return await _do_step(
                {
                    "action_type": "identify_gap",
                    "source_id": source_id,
                    "claimed_omitted_field": field,
                }
            )

        gap_btn.click(
            fn=on_identify_gap,
            inputs=[gap_source_dd, gap_field_dd],
            outputs=STEP_OUTPUTS,
        )

        # ── Synthesize ─────────────────────────────────────────────────────
        async def on_synthesize(
            responsible: str,
            omitted_fields: List[str],
            cover_up: str,
            red_herring: str,
        ):
            synthesis: Dict[str, Any] = {
                "responsible_party": responsible or "",
                "omitted_fields": omitted_fields or [],
                "cover_up_detail": cover_up or "",
            }
            if red_herring:
                synthesis["red_herring_source"] = red_herring
            return await _do_step({"action_type": "synthesize", "synthesis": synthesis})

        synth_btn.click(
            fn=on_synthesize,
            inputs=[synth_responsible, synth_omitted, synth_cover_up, synth_red_herring],
            outputs=STEP_OUTPUTS,
        )

        # ── Skip ───────────────────────────────────────────────────────────
        async def on_skip():
            return await _do_step({"action_type": "skip"})

        skip_btn.click(fn=on_skip, outputs=STEP_OUTPUTS)

    return demo
