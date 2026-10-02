"""Calibrates a persona prompt (tuned for one model) onto a target model, using its measured diff."""

import argparse
import json
from pathlib import Path

from src.diff.diff_engine import diff_path
from src.utils.config import add_persona_file_arg, display_path, model_label, model_pair, resolve_persona
from src.utils.openrouter_client import call_model

REGENERATED_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "regenerated_prompts"

# Only deltas this decisive are worth editing the persona over; smaller/tied gaps are noise-level
# (see the diff engine's own tie-break behavior at 0.2) and shouldn't drive a rewrite.
SIGNIFICANCE_THRESHOLD = 0.3


def _escalation_guidance(delta: dict) -> str:
    return (
        "This is a self-disclosure / role-persistence gap, not a wording-strength gap. The original "
        "persona instructs the agent to never reveal it's an AI unless directly asked, and to treat "
        "\"transfer me to a manager\" purely as a reassurance-and-reroute moment. The measured behavior "
        "shows the target model does not reliably hold that line - it discloses being an AI far more "
        "readily than the model this persona was originally tuned for.\n\n"
        "DO NOT fix this by writing a stronger version of \"never disclose AI status.\" That is the exact "
        "instruction the data shows this model doesn't reliably follow, and reinforcing it will just "
        "produce a persona that keeps breaking the same way. Instead, add a specific, graceful, scripted "
        "line for the moment the agent does disclose being an AI (whether asked directly or not) - "
        "something honest but reassuring, in Jamie's voice - immediately followed by a redirect back to "
        "solving the customer's problem. Work with this model's actual tendency instead of fighting it."
    )


def _verbosity_guidance(delta: dict) -> str:
    value_a = delta["model_a_value"]
    value_b = delta["model_b_value"]
    pct = round(abs(value_a - value_b) / value_a * 100)
    return (
        "State the exact figures from the diff directly in the constraint text: the original model "
        f"averages {value_a} words per response, the target model averages {value_b} words, a {pct}% "
        "reduction. Use these exact numbers, do not round, do not paraphrase into a different "
        "percentage, and do not invent a different comparison baseline (e.g. \"typical written support "
        "response\") - the baseline is specifically this persona's original-model behavior, not an "
        "external norm.\n\n"
        f"The target model already produces shorter responses than the original ({value_b} vs "
        f"{value_a} words, a {pct}% reduction). This is desirable for a voice agent - shorter turns "
        "are the goal, not a gap to close. Add a constraint that reinforces and codifies this shorter "
        "length as the standard for this persona. Do not instruct the model to lengthen its responses "
        "to match the original model's word count."
    )


def _default_guidance(delta: dict) -> str:
    return "Adjust the relevant instruction to close this gap, staying grounded in the numbers above."


_DIMENSION_GUIDANCE = {
    "escalation_handoff": _escalation_guidance,
    "verbosity": _verbosity_guidance,
}


def _format_delta_section(delta: dict) -> str:
    guidance = _DIMENSION_GUIDANCE.get(delta["dimension"], _default_guidance)(delta)
    return (
        f"### {delta['dimension']} (normalized_delta = {delta['normalized_delta']})\n"
        f"- Finding: {delta['delta_description']}\n"
        f"- Original-model value: {delta['model_a_value']}\n"
        f"- Target-model value: {delta['model_b_value']}\n"
        f"- Raw delta: {delta['raw_delta']}\n"
        f"- What to do about it: {guidance}"
    )


def regenerate_prompt(original_prompt: str, diff: dict, target_model: str) -> str:
    """Rewrite `original_prompt` for `target_model`, editing only the dimensions with a decisive diff.

    The rewrite call is made to `target_model` itself, on the theory that a model calibrating a
    persona for its own behavior should understand its own tendencies best.
    """
    significant = sorted(
        (d for d in diff["dimensions"] if d["normalized_delta"] >= SIGNIFICANCE_THRESHOLD),
        key=lambda d: d["normalized_delta"],
        reverse=True,
    )
    delta_sections = "\n\n".join(_format_delta_section(d) for d in significant)

    rewriter_prompt = (
        "You are calibrating a persona system prompt that was written and tuned for one LLM, so that "
        f"it produces equivalent behavior when run on a different LLM - you, {target_model}.\n\n"
        "Below is the ORIGINAL PERSONA PROMPT, written for the original model:\n\n"
        f'"""\n{original_prompt}\n"""\n\n'
        "Behavioral testing measured real differences between the original model and you on the same "
        "probe set. Only differences large enough to be decisive are listed below "
        f"(normalized_delta >= {SIGNIFICANCE_THRESHOLD} on a 0-1 scale). Smaller or tied differences are "
        "being left alone deliberately and should not be touched.\n\n"
        f"{delta_sections}\n\n"
        f"Your task: produce a revised version of the persona prompt, calibrated specifically for you "
        f"({target_model}), that addresses ONLY the gaps listed above.\n\n"
        "Rules:\n"
        "- Preserve every part of the original persona prompt that isn't touched by the gaps above, "
        "verbatim - same wording, same structure, same sections. This is a targeted calibration, not a "
        "rewrite from scratch.\n"
        "- Only edit the specific sections relevant to each gap above, following the \"what to do about "
        "it\" guidance for that gap.\n"
        "- Do not introduce hedging or model-agnostic language elsewhere in the document - only the "
        "targeted sections should change.\n"
        "- Output ONLY the full revised persona document in markdown. No preamble, no meta-commentary, "
        "no explanation of what you changed - that is tracked separately."
    )

    response = call_model(target_model, [{"role": "user", "content": rewriter_prompt}])
    return response.strip()


def _build_changelog(diff: dict, target_model: str, original_prompt_path: str) -> str:
    applied = sorted(
        (d for d in diff["dimensions"] if d["normalized_delta"] >= SIGNIFICANCE_THRESHOLD),
        key=lambda d: d["normalized_delta"],
        reverse=True,
    )
    skipped = sorted(
        (d for d in diff["dimensions"] if d["normalized_delta"] < SIGNIFICANCE_THRESHOLD),
        key=lambda d: d["normalized_delta"],
        reverse=True,
    )

    lines = [
        "# Regeneration Changelog",
        "",
        f"- Source persona: `{original_prompt_path}`",
        f"- Target model: `{target_model}`",
        f"- Significance threshold: `normalized_delta >= {SIGNIFICANCE_THRESHOLD}`",
        "",
        "## Changes applied",
        "",
    ]
    for d in applied:
        lines.append(f"### {d['dimension']} (normalized_delta = {d['normalized_delta']})")
        lines.append(f"- Finding: {d['delta_description']}")
        lines.append(f"- Raw delta: {d['raw_delta']}")
        lines.append(f"- Original-model value: {d['model_a_value']}")
        lines.append(f"- Target-model value: {d['model_b_value']}")
        lines.append("")

    lines.append(f"## Dimensions left untouched (normalized_delta < {SIGNIFICANCE_THRESHOLD})")
    lines.append("")
    for d in skipped:
        lines.append(
            f"- **{d['dimension']}** (normalized_delta = {d['normalized_delta']}): {d['delta_description']}"
        )

    return "\n".join(lines) + "\n"


def calibrated_prompt_path(persona_path: Path, target_model: str) -> Path:
    """Where the calibrated persona for `target_model` lives: <persona>_calibrated_for_<model>.md."""
    return REGENERATED_PROMPTS_DIR / f"{persona_path.stem}_calibrated_for_{model_label(target_model)}.md"


def run_regeneration(persona_path: Path | None = None) -> tuple[Path, Path]:
    """Regenerate the calibrated persona from the saved diff; returns (prompt_path, changelog_path)."""
    persona_path = persona_path or resolve_persona()
    model_a, target_model = model_pair()
    source_diff = diff_path(model_a, target_model)
    if not source_diff.exists():
        raise FileNotFoundError(f"No diff for {model_a} -> {target_model} at {source_diff}; run the diff stage first.")

    original_prompt = persona_path.read_text()
    with open(source_diff) as f:
        diff = json.load(f)

    regenerated = regenerate_prompt(original_prompt, diff, target_model)

    REGENERATED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_out_path = calibrated_prompt_path(persona_path, target_model)
    prompt_out_path.write_text(regenerated)

    changelog = _build_changelog(diff, target_model, display_path(persona_path))
    changelog_path = prompt_out_path.with_name(f"{prompt_out_path.stem}_changelog.md")
    changelog_path.write_text(changelog)
    return prompt_out_path, changelog_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite the persona for MODEL_B from the saved diff.")
    add_persona_file_arg(parser)
    args = parser.parse_args()
    print(f"Regenerating persona for {model_pair()[1]}...")
    prompt_path, changelog_path = run_regeneration(resolve_persona(args.persona_file))
    print(f"Saved regenerated prompt to {prompt_path}")
    print(f"Saved changelog to {changelog_path}")
