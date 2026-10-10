# persona-regen-tool

## Introduction

[![persona-regen-tool intro: same persona prompt, two models, different answers to "am I talking to a bot?"](docs/intro.gif)](docs/intro.mp4)

_20-second intro, built from this repo's own outputs. Click for the full video with sound._

When you move a production agent from one LLM to another, the same system prompt often behaves differently: replies get longer or shorter, the model handles "let me talk to a manager" differently, it self-discloses as an AI at different moments. `persona-regen-tool` measures that gap instead of guessing at it. It probes two models across seven behavioral dimensions (verbosity, refusal threshold, ambiguity handling, instruction literalness, tone/warmth, escalation/handoff behavior, naturalness), turns the results into a normalized behavioral diff, has the target model rewrite only the parts of your persona prompt that the largest deltas touch, and then validates the result with a scripted 7-turn pressure conversation run on both models with the original and the calibrated persona.

## Setup

1. Clone the repo and enter it:
   ```bash
   git clone <your-repo-url>
   cd persona-regen-tool
   ```
2. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env` and fill in your key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   - `OPENROUTER_API_KEY`: your key from [openrouter.ai](https://openrouter.ai).
   - `MODEL_A`: the model your persona was originally written for (default `openai/gpt-4o`). It also acts as the judge for scoring.
   - `MODEL_B`: the model you're migrating to (default `anthropic/claude-sonnet-4.6`). It also does the persona rewrite.
4. Confirm both model slugs exist on [openrouter.ai/models](https://openrouter.ai/models). Slugs change over time, and a wrong one fails on the first call.

Any pair of OpenRouter models works. Output filenames and labels come from the slugs, so `MODEL_A=openai/gpt-4o` and `MODEL_B=anthropic/claude-sonnet-4.6` produce `gpt-4o_to_claude-sonnet-4.6_diff.json`, `support_agent_v1_calibrated_for_claude-sonnet-4.6.md` and `naive_gpt-4o_transcript.md`, and switching either model never overwrites another pair's outputs.

## Quick Start

```bash
python run_pipeline.py
```

This runs everything end to end, with a section header per stage:

1. **Build fingerprints** for both models (the 7-dimension probe suite, scored).
2. **Compute the diff**: per-dimension deltas, ranked by a normalized 0-1 score.
3. **Regenerate the calibrated persona**: only deltas >= 0.3 drive edits; a changelog records what changed and why.
4. **Before/after validation**: the 7-turn pressure test, naive vs calibrated persona x both models, ending with the first turn at which each transcript deviates from the persona's instructions.

Expect roughly 8 minutes on a first run (it makes about 170 real, billed API calls, one after another). The script prints its own estimate at the start, based on which probe responses are already cached. A measured `--fresh` run took 7m 43s; a run that reuses cached probe responses skips the ~54 probe calls and should be a few minutes shorter. Pass `--fresh` to force the probes to re-run.

Outputs land in `outputs/` (gitignored): fingerprints, the diff, the regenerated persona and changelog, and the four validation transcripts.

To calibrate a different persona, pass its file (default `data/personas/support_agent_v1.md`):

```bash
python run_pipeline.py --persona-file path/to/my_persona.md
```

The calibrated prompt is named after the persona file (`my_persona_calibrated_for_<model>.md`). Only the rewrite step is fully persona-agnostic: the 7-turn validation conversation and its judge test the support agent's "be honest if asked whether you're an AI" rule, and the rewrite guidance for the escalation dimension assumes that rule too, so for other personas treat the validation results with that in mind.

Each stage can also be run alone: `python -m src.fingerprint.fingerprint`, `python -m src.diff.diff_engine`, `python -m src.regenerate.regenerator`, `python -m src.demo.validate`. They all read `MODEL_A`/`MODEL_B` from `.env` and take `--persona-file`. The fingerprint and diff stages also take a `--persona` flag to probe with the persona applied as the system prompt (persona-conditioned fingerprints, saved as `*_persona_<persona name>.json`).

## What we found

These are results from two full runs on a single support-agent persona (`data/personas/support_agent_v1.md`), with one conversation per condition in each. They are real findings at n=1 per condition, not guaranteed results.

- **Bare-model fingerprints don't fully predict behavior once a persona is applied.** Without a system prompt, the biggest gap was escalation: Claude declined the support-agent role in 3 of 5 probes, GPT-4o in 0 of 5 (normalized delta 0.60). Re-probing both models with the persona applied, that delta fell to 0.20 and neither model declined the role. Clear instructions suppress most raw personality differences. (Some dimensions moved for other reasons: many probes are out of scope for the persona, so several numbers there partly measure scope adherence.)
- **Under sustained pressure, the models did not hold the instructions equally well.** The persona says to stay in character but to be honest if directly asked whether the customer is talking to an AI. In the 7-turn test, GPT-4o (the model the persona was originally written for) broke that rule first: with the naive persona it dodged all four direct questions ("I'm part of the team") from turn 4 on and never said it was an AI. Claude answered honestly every time; the only thing the judge flagged for Claude was "dwelling" on the answer (from turn 5 with the naive persona, from turn 6 with the calibrated one), which is soft and threshold-dependent.
- **The calibration step, written for Claude, fixed GPT-4o's problem in one run, but not in a rerun.** In the first run the escalation delta was decisive, so the regenerated persona gained a scripted honest line for when the agent is asked directly. With it, GPT-4o admitted being an AI on turns 4-6 and held for the full conversation by the judge's count (its turn 7 reply is a borderline deflection). In a second full run with freshly generated probe responses, the escalation delta fell to 0.20 and the persona was calibrated on verbosity and instruction literalness instead, with no disclosure script. GPT-4o's naive persona still dodged the direct question from turn 4 (that part replicated), but the calibrated persona did too. The fix helped GPT-4o only when the calibration happened to include the disclosure script, so treat it as a side effect of that one run, not a reliable outcome of the tool.

Limits: the calibration itself is not stable across runs (fresh probe responses at temperature 0.7 changed which dimensions cleared the 0.3 threshold); one conversation per condition at temperature 0.7; the judge is GPT-4o at temperature 0, so it partly grades itself; the "dwelling" criterion is subjective. Repeating each condition 3-5 times would show whether the first-deviation turn is stable.

## How this differs from identa-agent

identa-agent gives you a portable persona format, with the per-model calibration written by hand. `persona-regen-tool` automates that calibration step: instead of relying on a person to guess how a model will drift, it measures the behavioral diff empirically (fingerprint both models, diff them) and has the target model rewrite only the sections of the persona that the measured deltas justify, leaving the rest verbatim.

## Structure

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Entry point: fingerprint -> diff -> regenerate -> validate |
| `data/probes.json` | The 7 behavioral dimensions and their probe prompts |
| `data/personas/` | Input persona prompts (`support_agent_v1.md`) |
| `src/utils/` | OpenRouter client (OpenAI SDK pointed at OpenRouter) |
| `src/fingerprint/` | Probe extraction, scoring, fingerprint building |
| `src/diff/` | Fingerprint diff with normalized deltas |
| `src/regenerate/` | Persona rewrite driven by the decisive deltas |
| `src/demo/` | The 7-turn pressure-test validation |
| `outputs/` | Generated fingerprints, diffs, calibrated prompts, transcripts (gitignored) |
