# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`persona-regen-tool` measures how two LLMs (via OpenRouter) behave differently under the same system prompt, then has the target model rewrite a persona prompt to close the largest gaps. See `README.md` for findings and limits. `../Persona_Tool_Claude_Code_Build_Guide.md` is the original build spec (the parent `pAG/` dir is not a git repo; this dir is).

## Commands

Python 3.10+. No test suite and no linter configured.

```bash
cp .env.example .env          # set OPENROUTER_API_KEY, MODEL_A, MODEL_B
python run_pipeline.py                         # full run; reuses cached probe responses
python run_pipeline.py --fresh                 # re-run probes
python run_pipeline.py --persona-file PATH     # calibrate another persona
python -m src.fingerprint.fingerprint          # single stages (also: src.diff.diff_engine, src.regenerate.regenerator, src.demo.validate)
python -m src.utils.openrouter_client          # smoke-test both model slugs
```

A full run makes ~170 real, billed API calls (~8 min, sequential). Run stages as modules from the repo root (`python -m ...`), not as scripts, so `src.*` imports resolve.

## Architecture

Pipeline in `run_pipeline.py`: fingerprint -> diff -> regenerate -> validate. Each stage is also a standalone module with its own `__main__`, reads `MODEL_A`/`MODEL_B` from env, and communicates with the next stage **only via files in `outputs/`** (gitignored).

- `src/utils/openrouter_client.py`: only LLM call path (`call_model`, OpenAI SDK pointed at OpenRouter, one retry).
- `src/utils/config.py`: derives all output filenames from model slugs and persona filename (`pair_tag`, `model_label`, `persona_suffix`). Add new output paths here/in stage modules via these helpers so different model pairs/personas never overwrite each other.
- `src/fingerprint/`: `extract.py` runs `data/probes.json` (7 dimensions x prompts) and caches raw responses; `score.py` scores each dimension (verbosity by word count, no LLM; others via LLM judge, MODEL_A at temperature 0); `fingerprint.py` orchestrates. `--persona` mode probes with the persona as system prompt.
- `src/diff/diff_engine.py`: per-dimension deltas normalized to 0-1, ranked.
- `src/regenerate/regenerator.py`: only deltas >= `SIGNIFICANCE_THRESHOLD` (0.3) drive edits; per-dimension rewrite guidance lives in `_DIMENSION_GUIDANCE` (escalation and verbosity have custom guidance, others fall back to default). Writes calibrated persona plus changelog; unchanged sections stay verbatim.
- `src/demo/validate.py`: scripted 7-turn pressure conversation, naive vs calibrated persona x both models, judge finds first turn deviating from persona.

## Gotchas

- Persona-agnostic only in the regenerate stage. `validate.py` and the escalation guidance hardcode the support agent's "be honest if asked whether you're an AI" rule (persona name "Jamie").
- Calibration is nondeterministic (probes at temperature 0.7); which dimensions clear 0.3 can change between runs.
- Judge is MODEL_A, so it partly grades itself.
- `outputs/` holds some older files with legacy names (e.g. `gpt4o_to_claude_diff.json`) alongside current slug-derived names; do not assume a filename without checking the helpers in `config.py`.
- `.env` holds a real key; never print or commit it.
