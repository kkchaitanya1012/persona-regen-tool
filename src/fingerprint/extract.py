"""Loads probe definitions and runs them against a model to fingerprint its persona."""

import argparse
import json
import os
from pathlib import Path

from rich.console import Console

from src.utils.config import add_persona_file_arg, persona_suffix, resolve_persona
from src.utils.openrouter_client import call_model

PROBES_PATH = Path(__file__).resolve().parents[2] / "data" / "probes.json"
FINGERPRINTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "fingerprints"

console = Console()


def raw_results_path(model_slug: str, persona_path: Path | None = None) -> Path:
    """Where a model's raw probe responses live; each persona-conditioned run gets its own file."""
    return FINGERPRINTS_DIR / f"{model_slug.replace('/', '_')}_raw{persona_suffix(persona_path)}.json"


def load_probes() -> list[dict]:
    """Read and return the probe dimensions from data/probes.json."""
    with open(PROBES_PATH) as f:
        return json.load(f)


def run_probes(model_slug: str, persona_path: Path | None = None) -> dict:
    """Run every probe prompt against `model_slug` and save the raw responses.

    With a persona_path, that persona is sent as the system prompt before each probe (the real
    deployment condition); otherwise probes go to the bare model.
    """
    probes = load_probes()
    results = {}
    system_messages = [{"role": "system", "content": persona_path.read_text()}] if persona_path else []

    condition = f"with persona {persona_path.name}" if persona_path else "bare model"
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
    out_path = raw_results_path(model_slug, persona_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    console.rule(f"[bold cyan]Saved raw results to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the probe suite against MODEL_B.")
    parser.add_argument("--persona", action="store_true", help="send the persona as the system prompt")
    add_persona_file_arg(parser)
    args = parser.parse_args()
    run_probes(os.getenv("MODEL_B"), persona_path=resolve_persona(args.persona_file) if args.persona else None)
