"""Diffs two model fingerprints into per-dimension deltas, ranked by normalized magnitude."""

import json
import sys
from pathlib import Path

DIFFS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "diffs"
FINGERPRINTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "fingerprints"

# Which normalization scheme applies to each dimension key from fingerprint.py's DIMENSION_KEYS.
BOUNDED_1_5 = {"tone_warmth", "naturalness"}
UNIT_INTERVAL = {"instruction_literalness"}
UNBOUNDED_COUNT = {"verbosity"}
DISTRIBUTION = {"refusal_threshold", "ambiguity_handling", "escalation_handoff"}

DIMENSION_LABELS = {
    "tone_warmth": "tone/warmth",
    "naturalness": "naturalness",
    "instruction_literalness": "instruction literalness",
}

# Human-readable phrasing for each category, used to build delta_description for
# distribution dimensions. Keys must match the category names produced by fingerprint.py.
CATEGORY_PHRASES = {
    "refusal_threshold": {
        "refuse": "explicitly refuse the request",
        "comply": "fully comply with the request",
        "hedge": "hedge rather than answer directly",
        "incomplete": "give an incomplete answer without an explicit refusal",
    },
    "ambiguity_handling": {
        "asks_first": "ask a clarifying question before giving any guidance",
        "answers_first": "jump straight to generic guidance without clarifying first",
        "mixed": "interleave clarifying questions with guidance",
    },
    "escalation_handoff": {
        "offers_handoff": "offer to hand off to a human/manager",
        "attempts_resolution": "attempt to resolve the issue itself without mentioning a handoff",
        "declines_role_offers_alternative": "explicitly decline the support-agent role and offer an alternative",
        "mixed": "mix more than one escalation behavior in a single response",
    },
}


def _distribution_proportions(distribution: dict) -> dict:
    total = sum(distribution.values())
    return {category: count / total for category, count in distribution.items()}


def _total_variation_distance(dist_a: dict, dist_b: dict) -> tuple[float, dict, dict]:
    """0.5 * sum of |proportion_a - proportion_b| across every category seen in either model."""
    categories = sorted(set(dist_a) | set(dist_b))
    prop_a = _distribution_proportions(dist_a)
    prop_b = _distribution_proportions(dist_b)
    tvd = 0.5 * sum(abs(prop_a.get(c, 0.0) - prop_b.get(c, 0.0)) for c in categories)
    return tvd, prop_a, prop_b


def _describe_distribution_shift(dimension, model_a, model_b, dist_a, dist_b, prop_a, prop_b) -> str:
    # Sorted (not a bare set) so a tie in the max() below resolves the same way on every run -
    # set iteration order for strings is randomized per-process otherwise.
    categories = sorted(set(dist_a) | set(dist_b))
    # The single category driving the largest proportion gap is what's worth describing.
    biggest_category = max(categories, key=lambda c: abs(prop_a.get(c, 0.0) - prop_b.get(c, 0.0)))
    signed_gap = prop_a.get(biggest_category, 0.0) - prop_b.get(biggest_category, 0.0)
    higher_model, lower_model = (model_a, model_b) if signed_gap > 0 else (model_b, model_a)
    higher_dist, lower_dist = (dist_a, dist_b) if signed_gap > 0 else (dist_b, dist_a)

    pct_points = abs(signed_gap) * 100
    phrase = CATEGORY_PHRASES.get(dimension, {}).get(biggest_category, biggest_category)
    count_higher, total_higher = higher_dist.get(biggest_category, 0), sum(higher_dist.values())
    count_lower, total_lower = lower_dist.get(biggest_category, 0), sum(lower_dist.values())

    return (
        f"{higher_model} is {pct_points:.0f} percentage points more likely to {phrase} than {lower_model} "
        f"({count_higher}/{total_higher} vs {count_lower}/{total_lower} probes)."
    )


def _describe_numeric_shift(dimension, model_a, model_b, value_a, value_b, kind) -> str:
    higher_model, lower_model = (model_a, model_b) if value_a >= value_b else (model_b, model_a)
    higher_value, lower_value = max(value_a, value_b), min(value_a, value_b)
    diff = higher_value - lower_value
    label = DIMENSION_LABELS.get(dimension, dimension)

    if kind == "bounded_1_5":
        return (
            f"{higher_model} scores {diff:.1f} points higher on {label} "
            f"({higher_value:.1f} vs {lower_value:.1f} on a 1-5 scale)."
        )
    if kind == "unit_interval":
        return (
            f"{higher_model} scores {diff * 100:.0f} percentage points higher on {label} "
            f"({higher_value * 100:.0f}% vs {lower_value * 100:.0f}%)."
        )
    if kind == "unbounded_count":
        pct = (diff / higher_value * 100) if higher_value else 0.0
        return (
            f"{higher_model} writes about {pct:.0f}% more words per response than {lower_model} "
            f"({higher_value:.1f} vs {lower_value:.1f} words on average)."
        )
    raise ValueError(f"Unknown numeric kind: {kind}")


def compute_diff(fingerprint_a: dict, fingerprint_b: dict, out_path: Path | None = None) -> dict:
    """Compare two model fingerprints dimension-by-dimension and rank the biggest behavioral shifts.

    Every dimension gets a raw_delta (human-readable, in the dimension's native units) and a
    normalized_delta in 0-1 (comparable across dimensions, used for ranking). The normalization
    scheme depends on the dimension's shape - see BOUNDED_1_5 / UNIT_INTERVAL / UNBOUNDED_COUNT /
    DISTRIBUTION above.
    """
    model_a, model_b = fingerprint_a["model"], fingerprint_b["model"]
    dims_a, dims_b = fingerprint_a["dimensions"], fingerprint_b["dimensions"]

    results = []
    for dimension in dims_a:
        value_a, value_b = dims_a[dimension], dims_b[dimension]

        if dimension in DISTRIBUTION:
            tvd, prop_a, prop_b = _total_variation_distance(value_a, value_b)
            categories = sorted(set(value_a) | set(value_b))
            raw_delta = {c: value_a.get(c, 0) - value_b.get(c, 0) for c in categories}
            normalized_delta = tvd
            description = _describe_distribution_shift(
                dimension, model_a, model_b, value_a, value_b, prop_a, prop_b
            )
        elif dimension in BOUNDED_1_5:
            raw_delta = round(value_a - value_b, 4)
            normalized_delta = abs(value_a - value_b) / 4
            description = _describe_numeric_shift(dimension, model_a, model_b, value_a, value_b, "bounded_1_5")
        elif dimension in UNIT_INTERVAL:
            raw_delta = round(value_a - value_b, 4)
            normalized_delta = abs(value_a - value_b)
            description = _describe_numeric_shift(dimension, model_a, model_b, value_a, value_b, "unit_interval")
        elif dimension in UNBOUNDED_COUNT:
            raw_delta = round(value_a - value_b, 4)
            denom = max(value_a, value_b)
            normalized_delta = abs(value_a - value_b) / denom if denom else 0.0
            description = _describe_numeric_shift(dimension, model_a, model_b, value_a, value_b, "unbounded_count")
        else:
            raise ValueError(f"No diff scheme registered for dimension: {dimension}")

        results.append(
            {
                "dimension": dimension,
                "model_a_value": value_a,
                "model_b_value": value_b,
                "raw_delta": raw_delta,
                "normalized_delta": round(normalized_delta, 4),
                "delta_description": description,
            }
        )

    top_deltas = sorted(results, key=lambda r: r["normalized_delta"], reverse=True)[:3]
    diff = {"dimensions": results, "top_deltas": top_deltas}

    DIFFS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or DIFFS_DIR / "gpt4o_to_claude_diff.json"
    with open(out_path, "w") as f:
        json.dump(diff, f, indent=2)

    return diff


if __name__ == "__main__":
    persona = "--persona" in sys.argv
    suffix = "_fingerprint_persona.json" if persona else "_fingerprint.json"
    with open(FINGERPRINTS_DIR / f"openai_gpt-4o{suffix}") as f:
        fingerprint_a = json.load(f)
    with open(FINGERPRINTS_DIR / f"anthropic_claude-sonnet-4.6{suffix}") as f:
        fingerprint_b = json.load(f)

    diff_out = DIFFS_DIR / ("gpt4o_to_claude_diff_persona.json" if persona else "gpt4o_to_claude_diff.json")
    diff = compute_diff(fingerprint_a, fingerprint_b, out_path=diff_out)

    print(f"Top 3 behavioral deltas: {fingerprint_a['model']} -> {fingerprint_b['model']}\n")
    for i, entry in enumerate(diff["top_deltas"], start=1):
        print(f"{i}. {entry['dimension']}  (normalized_delta = {entry['normalized_delta']})")
        print(f"   {entry['delta_description']}\n")

    all_ranked = sorted(diff["dimensions"], key=lambda r: r["normalized_delta"], reverse=True)
    print(f"All {len(all_ranked)} dimensions, ranked by normalized_delta:\n")
    for i, entry in enumerate(all_ranked, start=1):
        print(f"{i}. {entry['dimension']}  (normalized_delta = {entry['normalized_delta']})")
        print(f"   {entry['delta_description']}\n")
