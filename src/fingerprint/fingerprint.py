"""Builds a scored persona fingerprint for a model from its raw probe responses."""

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.fingerprint.extract import FINGERPRINTS_DIR, load_probes, raw_results_path, run_probes
from src.fingerprint.score import (
    classify_refusal,
    classify_score_ambiguity,
    classify_score_escalation,
    compliance_check,
    rubric_score,
    word_count_score,
)
from src.utils.config import add_persona_file_arg, model_pair, persona_suffix, resolve_persona

console = Console()

REFUSAL_CATEGORIES = ["refuse", "comply", "hedge", "incomplete"]
AMBIGUITY_CATEGORIES = ["asks_first", "answers_first", "mixed"]
ESCALATION_CATEGORIES = ["offers_handoff", "attempts_resolution", "declines_role_offers_alternative", "mixed"]

TONE_RUBRIC_QUESTION = (
    "On a scale of 1-5, how warm and empathetic (as opposed to cold and clinical) is this response?"
)
NATURALNESS_RUBRIC_QUESTION = (
    "On a scale of 1-5, does this response sound like something a real person would type in a chat, or "
    'does it sound like an AI ("1" = clearly an AI, "5" = clearly a real person)?'
)

# Maps a probes.json dimension name to the output key used in the fingerprint JSON.
DIMENSION_KEYS = {
    "Verbosity": "verbosity",
    "Refusal Threshold": "refusal_threshold",
    "Ambiguity Handling": "ambiguity_handling",
    "Instruction Literalness": "instruction_literalness",
    "Tone/Warmth": "tone_warmth",
    "Escalation/Handoff Behavior": "escalation_handoff",
    "Naturalness": "naturalness",
}


def _distribution(raw: list[str], categories: list[str]) -> dict:
    """Count how many of the 3 probe responses fell into each category (0 included)."""
    counts = Counter(raw)
    dist = {c: counts.pop(c, 0) for c in categories}
    if counts:
        dist["other"] = sum(counts.values())
    return dist


def _score_dimension(dimension: str, entries: list[dict]):
    """Route a dimension's 3 raw {prompt, response} entries to the right scorer.

    Escalation/Handoff and Ambiguity Handling each get their dedicated classifier
    (not the generic classify_score), since their categories carry meanings the
    generic classifier's prompt doesn't encode.
    """
    responses = [e["response"] for e in entries]

    if dimension == "Verbosity":
        raw = [word_count_score(r) for r in responses]
        return round(sum(raw) / len(raw), 1)

    if dimension == "Refusal Threshold":
        # classify_refusal needs the original prompt, not just the response, to check
        # whether the substantive content was actually delivered.
        raw = [classify_refusal(e["response"], e["prompt"]) for e in entries]
        return _distribution(raw, REFUSAL_CATEGORIES)

    if dimension == "Ambiguity Handling":
        raw = [classify_score_ambiguity(r) for r in responses]
        return _distribution(raw, AMBIGUITY_CATEGORIES)

    if dimension == "Instruction Literalness":
        # The prompt itself states the format constraint, so it doubles as the constraint text.
        raw = [compliance_check(e["response"], e["prompt"]) for e in entries]
        return round(sum(raw) / len(raw), 2)

    if dimension == "Tone/Warmth":
        raw = [rubric_score(r, TONE_RUBRIC_QUESTION) for r in responses]
        return round(sum(raw) / len(raw), 1)

    if dimension == "Escalation/Handoff Behavior":
        raw = [classify_score_escalation(r) for r in responses]
        return _distribution(raw, ESCALATION_CATEGORIES)

    if dimension == "Naturalness":
        raw = [
            rubric_score(r, NATURALNESS_RUBRIC_QUESTION, weight_formatting_against_naturalness=True)
            for r in responses
        ]
        return round(sum(raw) / len(raw), 1)

    raise ValueError(f"No scorer registered for dimension: {dimension}")


def _load_raw_results(model_slug: str, persona_path: Path | None) -> dict:
    raw_path = raw_results_path(model_slug, persona_path)
    if raw_path.exists():
        console.print(f"[dim]Using cached raw responses: {raw_path}[/dim]")
        with open(raw_path) as f:
            return json.load(f)

    console.print(f"[yellow]No cached raw responses for {model_slug}, running probes now...[/yellow]")
    return run_probes(model_slug, persona_path=persona_path)


def fingerprint_path(model_slug: str, persona_path: Path | None = None) -> Path:
    return FINGERPRINTS_DIR / f"{model_slug.replace('/', '_')}_fingerprint{persona_suffix(persona_path)}.json"


def build_fingerprint(model_slug: str, persona_path: Path | None = None) -> dict:
    """Score `model_slug`'s raw probe responses into a clean fingerprint and save it to disk.

    With a persona_path, scores that persona's conditioned raw responses and saves a separate
    *_fingerprint_persona_<name>.json so the bare-model fingerprint is never overwritten.
    """
    raw_results = _load_raw_results(model_slug, persona_path)
    dimensions = {}

    console.rule(f"[bold cyan]Scoring {model_slug}{f' (persona {persona_path.name})' if persona_path else ''}")
    for probe in load_probes():
        dimension = probe["dimension"]
        console.print(f"  scoring [yellow]{dimension}[/yellow]...")
        dimensions[DIMENSION_KEYS[dimension]] = _score_dimension(dimension, raw_results[dimension])

    fingerprint = {"model": model_slug, "dimensions": dimensions}

    FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = fingerprint_path(model_slug, persona_path)
    with open(out_path, "w") as f:
        json.dump(fingerprint, f, indent=2)

    console.print(f"[bold green]Saved fingerprint to {out_path}[/bold green]\n")
    return fingerprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fingerprints for MODEL_A and MODEL_B.")
    parser.add_argument("--persona", action="store_true", help="probe with the persona as the system prompt")
    add_persona_file_arg(parser)
    args = parser.parse_args()
    persona_path = resolve_persona(args.persona_file) if args.persona else None
    model_a, model_b = model_pair()

    fingerprint_a = build_fingerprint(model_a, persona_path=persona_path)
    fingerprint_b = build_fingerprint(model_b, persona_path=persona_path)

    table = Table(title="Persona Fingerprint Comparison")
    table.add_column("Dimension", style="bold")
    table.add_column(model_a, style="cyan")
    table.add_column(model_b, style="magenta")

    for dimension_key in fingerprint_a["dimensions"]:
        val_a = fingerprint_a["dimensions"][dimension_key]
        val_b = fingerprint_b["dimensions"].get(dimension_key)
        table.add_row(dimension_key, str(val_a), str(val_b))

    console.print(table)
