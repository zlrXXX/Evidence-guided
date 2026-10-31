# Counselling evaluation

The English pipeline uses the six original CACTUS CTRS rubrics. The Chinese pipeline uses the project’s Chinese translations of those rubrics. Each dialogue produces six independent requests: understanding, interpersonal effectiveness, collaboration, guided discovery, focus, and strategy for change. The judge is `gpt-5.6-terra`, with temperature 0, reasoning disabled, explicit prompt-cache mode, and a maximum of 1,024 output tokens per criterion. Each response must contain an integer score from 0 to 6 and an explanation.

## Setup

Use Python 3.10 or later on Linux or macOS:

```sh
python -m pip install -r evaluation/requirements.txt
```

`numpy>=1.26,<3` follows the experiment dependency specification. The historical CTRS launcher requires `certifi` without a recorded version pin. HTTPS requests use certificate verification and do not follow redirects or retry automatically.

## Input

Provide a JSON configuration with one or more cohorts. Each cohort contains one dialogue per profile. Profile identifiers must be stable across methods to support paired comparisons. Dialogue paths are relative to the configuration file.

```json
{
  "run_id": "cactus_trace_en",
  "cohorts": [
    {
      "cohort": "cactus_qwen_trace",
      "dataset": "cactus",
      "backbone": "qwen3",
      "sessions": [
        {
          "session_id": "CACTUS-E220V2-001",
          "profile_id": "CACTUS-E220V2-001",
          "dialogue": "dialogues/CACTUS-E220V2-001.json"
        }
      ]
    }
  ]
}
```

Each dialogue is a JSON array of objects with exactly `speaker` and `utterance`. Speakers alternate between `counsellor` and `patient`, starting with the counsellor. Include complete exchanges, with the opening counted as exchange 1. The input must be a dialogue-only export; controller records and hidden profiles are not sent to the judge.

Preparation preserves the complete source file under `full/` and creates the judge view from its first 40 complete exchanges under `inputs/`. It rejects incomplete pairs and blank utterances. Session/profile mappings, source hashes, transcript hashes, rubrics, request settings, and runner files are frozen in the manifest. Existing runs are never overwritten.

For translated evaluation, supply the translated dialogue-only files with the same session/profile identifiers. The pipeline scores the supplied language; it does not translate dialogues. Keep translated evaluation separate from direct generation in that language.

## Prepare and validate

Supply `pricing.json` with `model` set to `gpt-5.6-terra`, an ISO-format UTC `checked_utc` date, `long_context_threshold` in tokens, and the current Batch rates per million tokens in `batch_input_usd_per_million`, `batch_cached_input_usd_per_million`, `batch_cache_write_usd_per_million` and `batch_output_usd_per_million`. Verify model availability, endpoint support and prices before submission. The verification date must be within seven days. Cost bounds use UTF-8 bytes, full output allowances and a 25% reserve.

```sh
python evaluation/ctrs.py prepare --config cohorts.json --language en --pricing pricing.json --run runs/cactus_trace_en
```

Use `--language zh` with Chinese inputs for the Chinese pipeline. The prepared `requests.jsonl` contains six requests per session. Preparation and validation run locally and do not access credentials or the API.

## Submit, check, and collect

Set `OPENAI_API_KEY` in your own environment. Submission sends the prepared dialogues and rubrics to `https://api.openai.com` and incurs the provider’s API charges. Supply a maximum cost covering the bound reported by preparation.

```sh
python evaluation/ctrs.py submit --run runs/cactus_trace_en --maximum-cost-usd 25
python evaluation/ctrs.py status --run runs/cactus_trace_en
python evaluation/ctrs.py collect --run runs/cactus_trace_en
```

The Batch window is 24 hours. Upload and Batch creation states are saved before each network operation. An ambiguous submission blocks further submission until `reconcile` finds the matching remote object and verifies its identity and bytes:

```sh
python evaluation/ctrs.py reconcile --run runs/cactus_trace_en
```

Collection retains raw output and error files. The parser checks request identity, unique coverage, finish status, refusal status, judge identity, score range, explanation text, and usage fields. Chinese parsing also accepts `分` after the score, as in the original Chinese pipeline. Successful scores remain in `results/scores.json`; failures and missing responses remain in `results/failures.json` and `results/failed-requests.jsonl`. No recovery requests are submitted automatically.

```sh
python evaluation/ctrs.py summarize --run runs/cactus_trace_en
```

The summary reports criterion means and sample standard deviations, and Overall as the mean of the six criterion scores across sessions. It reports a cohort mean only when every expected session has all six scores. Token-based costs use the prepared pricing record and separate ordinary input, cached input, cache-write, and output usage.

## Confidence intervals and paired tests

`analyse.py` uses 10,000 profile bootstrap resamples for percentile 95% confidence intervals and 100,000 two-sided paired sign-flip permutations of six-criterion session sums. Monte Carlo p-values use `(extreme + 1) / (permutations + 1)`. Each declared family receives Holm correction. The main paper uses 13 baseline comparisons separately per dataset and 12 ablation comparisons jointly across datasets and backbones. Seed values are deterministic and written to the result. Set `bootstrap_seed` per group and `permutation_seed` per comparison to reuse recorded seeds.

Supply named groups pointing to prepared runs and their cohort identifiers. Declare every comparison in each family and its expected count. For example, a separate one-comparison analysis is:

```json
{
  "groups": [
    {"name": "trace", "run": "runs/cactus_trace_en", "cohort": "cactus_qwen_trace"},
    {"name": "baseline", "run": "runs/cactus_baseline_en", "cohort": "cactus_qwen_baseline"}
  ],
  "comparisons": [
    {"reference": "trace", "variant": "baseline", "family": "single_comparison"}
  ],
  "families": {"single_comparison": 1}
}
```

```sh
python evaluation/analyse.py --config comparisons.json --output statistics.json
```

Pairing uses profile identifiers within the same dataset, backbone, and evaluation language. The output records matched and unmatched profiles. Incomplete scoring blocks analysis by default; `--allow-incomplete` explicitly limits analysis to complete six-score profiles and reports every excluded profile. Planned generation exclusions should be reflected in the input cohort instead of represented as zero scores. Do not use a smaller comparison family to report the paper’s corrected p-values.

## Rubrics

English source: https://github.com/coding-groot/cactus, revision `da8cafc3d7dacbbf6073bc52a92dc09e46a398a3`. `prompts/en/` preserves the official rubric bytes; `prompts/zh/` preserves the Chinese translations used in the experiments. The Chinese translations retain the criterion definitions, scoring anchors, interpolation rules, deficiency limits, and score-plus-explanation format.
