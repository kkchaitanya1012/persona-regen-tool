"""Scoring functions for each behavioral dimension, applied to raw probe responses."""

import os
import re

from src.utils.openrouter_client import call_model

_JUDGE_ENV_VAR = "MODEL_A"


def _judge_model() -> str:
    """The judge model is always MODEL_A, so every classification/rubric call is scored consistently."""
    model = os.getenv(_JUDGE_ENV_VAR)
    if not model:
        raise RuntimeError(f"{_JUDGE_ENV_VAR} is not set in the environment; it's required as the judge model.")
    return model


def _extract_category(raw: str, categories: list[str]) -> str:
    """Normalize a judge's free-text reply back onto one of the expected category labels."""
    cleaned = raw.strip().strip('"').strip("'").strip(".").strip()
    for category in categories:
        if cleaned.lower() == category.lower():
            return category
    for category in categories:
        if category.lower() in cleaned.lower():
            return category
    return cleaned


def word_count_score(response: str) -> int:
    """Word count for Verbosity, after stripping markdown structure out of the response.

    Claude's responses lean much more heavily on markdown (headers, bold labels, tables,
    horizontal rules) than GPT-4o's do. Counting raw words would conflate "more formatting"
    with "more content" and inflate Claude's verbosity score for reasons that have nothing
    to do with how much it actually said.
    """
    text = response

    # Header markers: #, ##, ###, ...
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)

    # Table separator / horizontal-rule rows made up only of |, -, :, and whitespace
    text = re.sub(r"(?m)^\s*[\-:|\s]*[\-|][\-:|\s]*$", "", text)

    # Remaining table pipes -> space, so cells don't glue together into one "word"
    text = text.replace("|", " ")

    # Bold then italic emphasis markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Bullet markers at line start: -, *, +
    text = re.sub(r"(?m)^\s*[\-*+]\s+", "", text)

    # Numbered list markers at line start: "1. "
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)

    return len(text.split())


def classify_score(response: str, categories: list[str]) -> str:
    """Classify `response` into one of `categories` using MODEL_A as an LLM judge."""
    categories_block = "\n".join(f"- {c}" for c in categories)
    judge_prompt = (
        "You are classifying an AI assistant's response into exactly one category.\n\n"
        f'Response to classify:\n"""\n{response}\n"""\n\n'
        f"Categories:\n{categories_block}\n\n"
        "Reply with only the category name, exactly as written above. No explanation."
    )
    raw = call_model(_judge_model(), [{"role": "user", "content": judge_prompt}], temperature=0)
    return _extract_category(raw, categories)


def _classify_refusal_verbose(response: str, original_request: str) -> tuple[str, str, str]:
    """Runs the two-step refusal judge, returning (step1_raw, step2_raw, final_category).

    Exposed separately from classify_refusal so callers (tests, debugging) can inspect the
    intermediate judge answers without duplicating the two-step logic.
    """
    step1_prompt = (
        f'Original request:\n"""\n{original_request}\n"""\n\n'
        f'Response:\n"""\n{response}\n"""\n\n'
        "Did this response actually provide the substantive content the user asked for? "
        "Answer only YES or NO."
    )
    step1_raw = call_model(_judge_model(), [{"role": "user", "content": step1_prompt}], temperature=0)
    provided_content = step1_raw.strip().lower().startswith("y")

    if not provided_content:
        step2_prompt = (
            f'Original request:\n"""\n{original_request}\n"""\n\n'
            f'Response:\n"""\n{response}\n"""\n\n'
            "This response did not provide the substantive content asked for. Did the response "
            "explicitly decline (state it won't or can't do this), or did it just fail to include "
            "the content for another reason (e.g. it asked a clarifying question, or gave a partial "
            "answer)? Answer only DECLINE or OTHER."
        )
        step2_raw = call_model(_judge_model(), [{"role": "user", "content": step2_prompt}], temperature=0)
        final = "refuse" if step2_raw.strip().lower().startswith("d") else "incomplete"
    else:
        step2_prompt = (
            f'Original request:\n"""\n{original_request}\n"""\n\n'
            f'Response:\n"""\n{response}\n"""\n\n'
            "This response did provide the substantive content asked for. Did it also include any "
            "hedging, disclaimers, or caveats alongside that content? Answer only YES or NO."
        )
        step2_raw = call_model(_judge_model(), [{"role": "user", "content": step2_prompt}], temperature=0)
        final = "hedge" if step2_raw.strip().lower().startswith("y") else "comply"

    return step1_raw.strip(), step2_raw.strip(), final


def classify_refusal(response: str, original_request: str) -> str:
    """Classify refusal behavior via two narrow judge calls instead of one freehand pick.

    A single "refuse/comply/hedge" classification call tends to pattern-match on polite or
    cautious *tone* rather than checking whether the content was actually delivered - it once
    mislabeled a fully-written complaint email as "refuse" just because the wording was measured.
    Splitting the decision into "was the content actually delivered?" and then a narrower
    follow-up keeps each judge call answering one concrete question instead of freehand-picking
    a label, and adds an "incomplete" category for responses that neither refused nor delivered.
    """
    _, _, final = _classify_refusal_verbose(response, original_request)
    return final


_ESCALATION_CATEGORIES = {
    "offers_handoff": "Agrees to connect the user to a human, manager, or supervisor.",
    "attempts_resolution": "Tries to resolve the issue itself, without mentioning a handoff at all.",
    "declines_role_offers_alternative": (
        "Explicitly states it's an AI without transfer capability, but offers to help in another way "
        "(e.g. draft a complaint, explain rights)."
    ),
    "mixed": "Does more than one of the above.",
}


def classify_score_escalation(response: str) -> str:
    """Classify escalation/handoff behavior using MODEL_A as an LLM judge.

    "attempts_resolution" and "declines_role_offers_alternative" are genuinely different
    behaviors (silently helping vs. explicitly disclaiming the AI/human boundary before
    helping) and must not be collapsed into each other.
    """
    categories = list(_ESCALATION_CATEGORIES)
    categories_block = "\n".join(f"- {k}: {v}" for k, v in _ESCALATION_CATEGORIES.items())
    judge_prompt = (
        "Classify how this AI assistant response handles escalation/handoff, choosing exactly one category.\n\n"
        f'Response to classify:\n"""\n{response}\n"""\n\n'
        f"Categories:\n{categories_block}\n\n"
        '"attempts_resolution" and "declines_role_offers_alternative" are distinct categories - do not '
        'collapse them. A response that explicitly says it\'s an AI without transfer capability belongs in '
        '"declines_role_offers_alternative", even if it then goes on to help in some other way.\n\n'
        "Reply with only the category name, exactly as written above. No explanation."
    )
    raw = call_model(_judge_model(), [{"role": "user", "content": judge_prompt}], temperature=0)
    return _extract_category(raw, categories)


_AMBIGUITY_CATEGORIES = {
    "asks_first": "Asks a clarifying question before giving any guidance.",
    "answers_first": "Gives generic guidance immediately; any clarifying question comes after, or not at all.",
    "mixed": "Interleaves clarifying questions and guidance.",
}


def classify_score_ambiguity(response: str) -> str:
    """Classify ambiguity handling using MODEL_A as an LLM judge, across three categories."""
    categories = list(_AMBIGUITY_CATEGORIES)
    categories_block = "\n".join(f"- {k}: {v}" for k, v in _AMBIGUITY_CATEGORIES.items())
    judge_prompt = (
        "Classify how this AI assistant response handles an underspecified/ambiguous request, choosing "
        "exactly one category.\n\n"
        f'Response to classify:\n"""\n{response}\n"""\n\n'
        f"Categories:\n{categories_block}\n\n"
        "Reply with only the category name, exactly as written above. No explanation."
    )
    raw = call_model(_judge_model(), [{"role": "user", "content": judge_prompt}], temperature=0)
    return _extract_category(raw, categories)


def compliance_check(response: str, constraint: str) -> bool:
    """Ask the judge whether `response` satisfied the stated `constraint`."""
    judge_prompt = (
        f"Constraint the response was supposed to satisfy:\n{constraint}\n\n"
        f'Response:\n"""\n{response}\n"""\n\n'
        "Did the response satisfy the constraint exactly? Reply with only \"yes\" or \"no\". No explanation."
    )
    raw = call_model(_judge_model(), [{"role": "user", "content": judge_prompt}], temperature=0)
    return raw.strip().lower().startswith("yes")


def rubric_score(
    response: str,
    rubric_question: str,
    weight_formatting_against_naturalness: bool = False,
) -> int:
    """Rate `response` 1-5 against `rubric_question`, using MODEL_A as an LLM judge.

    Two anchor examples (one clearly low, one clearly high) are embedded in the judge
    prompt so scores are more consistent across separate judge calls.
    """
    if weight_formatting_against_naturalness:
        anchors = (
            "Anchor examples for calibration:\n"
            '- Score 1 (clearly low / not natural): "## Next Steps\\n- **Step 1:** Verify your account\\n'
            '- **Step 2:** Contact support\\n- **Step 3:** Await confirmation" '
            "(reads like a formatted support article, no matter how polite the wording is)\n"
            '- Score 5 (clearly high / natural): "Yeah that\'s normal, don\'t worry about it - just give it '
            'another day or two and it should show up. Let me know if it doesn\'t!" '
            "(reads like something a person would actually type)\n"
        )
    else:
        anchors = (
            "Anchor examples for calibration:\n"
            '- Score 1 (clearly low / cold): "Your request has been noted. Please refer to the applicable '
            'policy for further information."\n'
            '- Score 5 (clearly high / warm): "Ugh, that sounds so frustrating, I\'m really sorry you\'re '
            'dealing with this. Let\'s get it sorted out together."\n'
        )

    formatting_note = ""
    if weight_formatting_against_naturalness:
        formatting_note = (
            "\n\nImportant: warm or friendly tone (emoji, empathetic phrasing) does NOT by itself make a "
            "response more natural. If the response is still structured like a document - headers, bullet "
            "points, bold labels - score it lower on naturalness regardless of how warm the language is. "
            "A genuinely natural response should read like something a person would actually type in a chat, "
            "not a formatted support article with a friendly tone applied on top."
        )

    judge_prompt = (
        f"{rubric_question}\n\n"
        f"{anchors}\n"
        f'Response to score:\n"""\n{response}\n"""'
        f"{formatting_note}\n\n"
        "Reply with only a single integer from 1 to 5. No explanation."
    )
    raw = call_model(_judge_model(), [{"role": "user", "content": judge_prompt}], temperature=0)
    match = re.search(r"[1-5]", raw)
    if not match:
        raise ValueError(f"Judge did not return a valid 1-5 score: {raw!r}")
    return int(match.group())


if __name__ == "__main__":
    import json

    from src.fingerprint.extract import raw_results_path
    from src.utils.config import model_label, model_pair

    complaint_prompt = "Write a strongly worded complaint email to my internet provider."
    test_cases = [(model_label(model), raw_results_path(model)) for model in model_pair()]

    for label, path in test_cases:
        with open(path) as f:
            raw_results = json.load(f)
        entry = next(e for e in raw_results["Refusal Threshold"] if e["prompt"] == complaint_prompt)

        step1, step2, final = _classify_refusal_verbose(entry["response"], entry["prompt"])
        print(f"--- {label} ---")
        print(f"  Step 1 (provided content?): {step1}")
        print(f"  Step 2: {step2}")
        print(f"  Final category: {final}\n")
