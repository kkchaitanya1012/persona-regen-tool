"""Loads probe definitions and runs them against a model to fingerprint its persona."""

import json
import os
import sys
from pathlib import Path

from rich.console import Console

from src.utils.openrouter_client import call_model

PROBES_PATH = Path(__file__).resolve().parents[2] / "data" / "probes.json"
FINGERPRINTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "fingerprints"
PERSONA_PATH = Path(__file__).resolve().parents[2] / "data" / "personas" / "support_agent_v1.md"

console = Console()


def raw_results_path(model_slug: str, persona: bool = False) -> Path:
    """Where a model's raw probe responses live; persona-conditioned runs get their own file."""
    suffix = "_raw_persona.json" if persona else "_raw.json"
    return FINGERPRINTS_DIR / f"{model_slug.replace('/', '_')}{suffix}"


def load_probes() -> list[dict]:
    """Read and return the probe dimensions from data/probes.json."""
    with open(PROBES_PATH) as f:
        return json.load(f)


def run_probes(model_slug: str, persona: bool = False) -> dict:
    """Run every probe prompt against `model_slug` and save the raw responses.

    With persona=True, the support_agent_v1.md persona is sent as the system prompt before each
    probe (the real deployment condition); otherwise probes go to the bare model as before.
    """
    probes = load_probes()
    results = {}
    system_messages = [{"role": "system", "content": PERSONA_PATH.read_text()}] if persona else []

    condition = "with persona system prompt" if persona else "bare model"
    console.rule(f"[bold cyan]Running probes on {model_slug} ({condition})")

    for probe in probes:
        dimension = probe["dimension"]
        console.print(f"\n[bold yellow]{dimension}[/bold yellow]")
        results[dimension] = []

        for i, prompt in enumerate(probe["prompts"], start=1):
            console.print(f"  [{i}/{len(probe['prompts'])}] [dim]{prompt}[/dim]")
            response = call_model(model_slug, system_messages + [{"role": "user", "content": prompt}])
            console.print(f"        [green]-> {response[:120]}{'...' if len(response) > 120 else ''}[/green]")
            results[dimension].append({"prompt": prompt, "response": response})

    FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = raw_results_path(model_slug, persona)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    console.rule(f"[bold cyan]Saved raw results to {out_path}")

    return results


if __name__ == "__main__":
    model_b = os.getenv("MODEL_B")
    run_probes(model_b, persona="--persona" in sys.argv)
