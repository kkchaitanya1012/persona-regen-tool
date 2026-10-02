"""Names and paths derived from the configured model pair and persona, so outputs say what was actually run."""

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSONA_PATH = ROOT / "data" / "personas" / "support_agent_v1.md"


def model_label(model_slug: str) -> str:
    """Short, filename-safe name for a model: "openai/gpt-4o" -> "gpt-4o"."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model_slug.split("/")[-1])


def pair_tag(model_a: str, model_b: str) -> str:
    """Filename tag for a migration pair: "gpt-4o_to_claude-sonnet-4.6"."""
    return f"{model_label(model_a)}_to_{model_label(model_b)}"


def model_pair() -> tuple[str, str]:
    """(MODEL_A, MODEL_B) from the environment; both are required."""
    model_a, model_b = os.getenv("MODEL_A"), os.getenv("MODEL_B")
    missing = [name for name, value in (("MODEL_A", model_a), ("MODEL_B", model_b)) if not value]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not set in the environment (see .env.example).")
    return model_a, model_b


def resolve_persona(path: str | Path | None = None) -> Path:
    """The persona file to use: `path` if given (relative to the current directory), else the default."""
    persona_path = Path(path).expanduser().resolve() if path else DEFAULT_PERSONA_PATH
    if not persona_path.is_file():
        raise FileNotFoundError(f"Persona file not found: {persona_path}")
    return persona_path


def persona_suffix(persona_path: Path | None) -> str:
    """Filename suffix for persona-conditioned outputs, named after the persona file ("" for bare runs)."""
    return f"_persona_{persona_path.stem}" if persona_path else ""


def display_path(path: Path) -> str:
    """`path` relative to the repo root when it's inside it, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add_persona_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--persona-file",
        metavar="PATH",
        help=f"persona prompt to calibrate (default: {display_path(DEFAULT_PERSONA_PATH)})",
    )
