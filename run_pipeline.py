"""Run the whole persona-calibration flow end to end: fingerprint -> diff -> regenerate -> validate.

    python run_pipeline.py            # reuses cached probe responses in outputs/fingerprints/ if present
    python run_pipeline.py --fresh    # re-runs the probes against both models first
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

console = Console()

# Rough latency per OpenRouter call, from one measured --fresh run (7m43s total); varies with provider load.
SECONDS_PER_PROBE_CALL = 4.5  # bare/persona probe generation (short-to-medium replies)
SECONDS_PER_JUDGE_CALL = 1.5  # temperature-0, one-word/JSON answers
SECONDS_PER_CONVO_TURN = 3.5  # multi-turn persona conversation reply (history grows each turn)
SECONDS_FOR_REGENERATION = 20.0  # one long rewrite of the full persona document

# classify_refusal makes two judge calls per probe; verbosity is scored without an LLM.
JUDGE_CALLS_PER_PROBE = {"Verbosity": 0, "Refusal Threshold": 2}


def fmt_duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def check_environment() -> None:
    missing = [name for name in ("OPENROUTER_API_KEY", "MODEL_A", "MODEL_B") if not os.getenv(name)]
    key = os.getenv("OPENROUTER_API_KEY", "")
    if missing or key.endswith("your-key-here"):
        problem = f"Missing in .env: {', '.join(missing)}" if missing else "OPENROUTER_API_KEY is still the placeholder."
        console.print(
            Panel(
                f"{problem}\nCopy .env.example to .env, add your OpenRouter key, and confirm the model slugs.",
                title="Setup incomplete",
                border_style="red",
            )
        )
        raise SystemExit(1)


def estimate(fresh: bool, models: list[str]) -> tuple[Table, float]:
    from src.demo.validate import CONVERSATION, MODELS, PERSONAS
    from src.fingerprint.extract import load_probes, raw_results_path

    probes = load_probes()
    probe_prompts = sum(len(p["prompts"]) for p in probes)
    judge_per_model = sum(len(p["prompts"]) * JUDGE_CALLS_PER_PROBE.get(p["dimension"], 1) for p in probes)
    models_to_probe = [m for m in models if fresh or not raw_results_path(m).exists()]

    probe_calls = probe_prompts * len(models_to_probe)
    judge_calls = judge_per_model * len(models)
    convo_turns = len(PERSONAS) * len(MODELS) * len(CONVERSATION)

    stages = [
        (
            "1. Fingerprints",
            f"{probe_calls} probe + {judge_calls} judge",
            probe_calls * SECONDS_PER_PROBE_CALL + judge_calls * SECONDS_PER_JUDGE_CALL,
            f"probing {len(models_to_probe)}/2 models" + ("" if fresh else " (cached responses reused for the rest)"),
        ),
        ("2. Diff", "0", 0.0, "local computation"),
        ("3. Regenerate persona", "1 rewrite", SECONDS_FOR_REGENERATION, ""),
        (
            "4. Validation",
            f"{convo_turns} chat + {convo_turns} judge",
            convo_turns * (SECONDS_PER_CONVO_TURN + SECONDS_PER_JUDGE_CALL),
            f"{len(PERSONAS)} personas x {len(MODELS)} models x {len(CONVERSATION)} turns",
        ),
    ]
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Stage")
    table.add_column("API calls")
    table.add_column("Est. time", justify="right")
    table.add_column("Notes", style="dim")
    for name, calls, seconds, note in stages:
        table.add_row(name, calls, fmt_duration(seconds) if seconds else "-", note)
    return table, sum(s[2] for s in stages)


def run_stage(number: int, title: str, timings: list, fn):
    console.print()
    console.rule(f"[bold cyan]Stage {number}/4 - {title}", style="cyan")
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        console.print(Panel(f"{type(exc).__name__}: {exc}", title=f"Stage {number} failed", border_style="red"))
        raise SystemExit(1)
    timings.append((f"{number}. {title}", time.perf_counter() - started))
    return result


def compact(value) -> str:
    if isinstance(value, dict):
        return "  ".join(f"{k}:{v}" for k, v in value.items() if v)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full persona fingerprint -> diff -> regenerate -> validate flow.")
    parser.add_argument("--fresh", action="store_true", help="re-run the probes even if cached responses exist")
    args = parser.parse_args()

    check_environment()

    from src.demo.validate import print_summary, run_validation
    from src.diff.diff_engine import compute_diff
    from src.fingerprint.extract import raw_results_path, run_probes
    from src.fingerprint.fingerprint import build_fingerprint
    from src.regenerate.regenerator import SIGNIFICANCE_THRESHOLD, run_regeneration

    model_a, model_b = os.getenv("MODEL_A"), os.getenv("MODEL_B")

    console.print(
        Panel.fit(
            f"[bold]persona-regen-tool[/bold]  -  full pipeline\n"
            f"Original model (MODEL_A): [cyan]{model_a}[/cyan]\n"
            f"Target model   (MODEL_B): [cyan]{model_b}[/cyan]\n"
            f"Persona: data/personas/support_agent_v1.md",
            border_style="cyan",
        )
    )
    table, total_seconds = estimate(args.fresh, [model_a, model_b])
    console.print(table)
    console.print(
        f"[bold]Estimated total: about {fmt_duration(total_seconds)}[/bold] "
        f"(plausible range {fmt_duration(total_seconds * 0.7)} to {fmt_duration(total_seconds * 1.5)}). "
        "[dim]Calls run sequentially; time depends on OpenRouter load. Real API calls are made and billed.[/dim]"
    )

    timings: list[tuple[str, float]] = []
    pipeline_started = time.perf_counter()

    def stage_fingerprints():
        fingerprints = []
        for model in (model_a, model_b):
            if args.fresh or not raw_results_path(model).exists():
                run_probes(model)
            fingerprints.append(build_fingerprint(model))
        table = Table(title="Bare-model fingerprints", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Dimension", style="bold")
        table.add_column(f"{model_a}", style="cyan")
        table.add_column(f"{model_b}", style="magenta")
        for key in fingerprints[0]["dimensions"]:
            table.add_row(key, compact(fingerprints[0]["dimensions"][key]), compact(fingerprints[1]["dimensions"][key]))
        console.print(table)
        return fingerprints

    fingerprint_a, fingerprint_b = run_stage(1, "Build fingerprints for both models", timings, stage_fingerprints)

    def stage_diff():
        diff = compute_diff(fingerprint_a, fingerprint_b)
        table = Table(title="Behavioral diff, ranked by normalized delta", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("#", justify="right")
        table.add_column("Dimension", style="bold")
        table.add_column("Delta", justify="right")
        table.add_column("What differs")
        table.add_column("Drives calibration?", justify="center")
        ranked = sorted(diff["dimensions"], key=lambda d: d["normalized_delta"], reverse=True)
        for i, d in enumerate(ranked, start=1):
            decisive = d["normalized_delta"] >= SIGNIFICANCE_THRESHOLD
            table.add_row(
                str(i), d["dimension"], f"{d['normalized_delta']:.2f}", d["delta_description"],
                "[green]yes[/green]" if decisive else "[dim]no[/dim]",
            )
        console.print(table)
        console.print(f"[dim]Only deltas >= {SIGNIFICANCE_THRESHOLD} are used to edit the persona.[/dim]")
        return diff

    run_stage(2, "Compute the behavioral diff", timings, stage_diff)

    def stage_regenerate():
        prompt_path, changelog_path = run_regeneration()
        console.print(f"Calibrated persona: [bold]{prompt_path.relative_to(ROOT)}[/bold]")
        console.print(f"Changelog:          [bold]{changelog_path.relative_to(ROOT)}[/bold]")

    run_stage(3, "Regenerate the calibrated persona", timings, stage_regenerate)

    def stage_validation():
        results = run_validation()
        print_summary(results)

    run_stage(4, "Before/after validation (7-turn pressure test, 4 conditions)", timings, stage_validation)

    total = time.perf_counter() - pipeline_started
    summary = Table(box=box.SIMPLE, show_header=False)
    for name, seconds in timings:
        summary.add_row(name, fmt_duration(seconds))
    summary.add_row("[bold]Total[/bold]", f"[bold]{fmt_duration(total)}[/bold]")
    console.print()
    console.rule("[bold green]Pipeline complete", style="green")
    console.print(summary)
    console.print(
        "Outputs in [bold]outputs/[/bold]: fingerprints/, diffs/, regenerated_prompts/, "
        "validation/{naive,calibrated}_{gpt4o,claude}_transcript.md"
    )


if __name__ == "__main__":
    main()
