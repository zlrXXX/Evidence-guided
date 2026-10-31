# Training

The entry points cover CACTUS and EEYORE, Qwen3-8B and Llama-3.1-8B-Instruct, and separate counsellor and patient SFT and RLOO. The selected source implementations retain their prompts, reward computations, sampling, checkpoint selection and training settings.

## Environment and data

Use Python 3.12 and a BF16-capable NVIDIA A100 with 80 GB GPU memory. From the repository root:

```bash
python -m pip install -r training/requirements.txt
python training/_inputs.py
```

Prepare the base-model snapshots listed in `external_assets.json`. Supply their local directories with `--model-dir`; gated model access requires your own Hugging Face permission. `HF_TOKEN` may be supplied through the environment. Model weights and trained checkpoint binaries are not included in this code-and-data package.

SFT uses the released train/development/test files under `datasets/<dataset>/sft/<role>/`. The test split is retained as a separate input and is not used for checkpoint selection. RLOO reads `datasets/<dataset>/rloo/<role>/<backbone>/`. Dataset bytes, counts and hashes are checked against `datasets/data_manifest.json` before training. `--data-dir` can point to another extraction of the same released files.

## Supervised fine-tuning

Run both roles separately for each dataset and backbone:

```bash
python training/run_sft.py --dataset cactus --backbone qwen3 --role patient --model-dir assets/qwen3 --output outputs/cactus/qwen3/patient_sft
python training/run_sft.py --dataset cactus --backbone qwen3 --role counsellor --model-dir assets/qwen3 --output outputs/cactus/qwen3/counsellor_sft
```

Substitute `eeyore`, `llama31` and the matching model directory for the other conditions. `sft_settings.json` records the per-condition configuration. The common settings are BF16, LoRA rank 32, alpha 64, dropout 0.05, learning rate 2e-5, batch size 1 with 12 accumulation steps, and a 16,384-token sequence limit. Counsellor SFT runs for one epoch and patient SFT for two. Only target tokens contribute to the loss; the selected adapter is determined by development loss. Qwen cores write selected weights to `adapter/`; Llama cores write `best_adapter/` inside their role-specific run directory. Completion records identify the selected output.

Append `--validate-only` to inspect the exact dataset hashes and settings without loading models, using a GPU or contacting an API.

## Reinforcement learning

For each dataset/backbone condition, train the patient first and then train the counsellor with that selected patient frozen. RLOO needs the frozen resistance forecaster; patient training also needs the patient-state expression checker, and counsellor training needs the historical CBT evidence scorer. The required files and checkpoint hashes are in `external_assets.json`. The later cumulative CBT tracker is a different component and cannot replace this training scorer.

Set `OPENAI_API_KEY` in the process environment when you intend to run the semantic reward judge. The code never retrieves credentials from a personal key store. Supply an explicit `--max-api-cost` for the judge's existing budget guard. The original judge uses GPT-5.6 Luna with high reasoning; its prompts and selected-judge contracts are included in each condition.

```bash
python training/run_rloo.py --dataset cactus --backbone qwen3 --role patient --model-dir assets/qwen3 --sft-adapter outputs/cactus/qwen3/patient_sft/adapter --resistance-tracker assets/resistance_forecaster --expression-checker assets/patient_state_expression --context-checkpoint-sha256 9C5B69EEEC97C890021FB7AF564BD8B9F0E55BD81ABB84E11A552EAAB14FABAA --output outputs/cactus/qwen3/patient_rloo --max-api-cost 50
```

`--context-checkpoint-sha256` is the recorded binding for the frozen counselling contexts, not the hash of the newly trained patient. Use the condition's value in `external_assets.json`. The displayed cost ceiling is an example, not an estimated cost.

After patient training completes, set `--patient-rloo-run` to the directory containing `COMPLETED.json`, `RUN_CONFIG.json`, `training_summary.json` and `selected_patient_rl_adapter/`:

```bash
python training/run_rloo.py --dataset cactus --backbone qwen3 --role counsellor --model-dir assets/qwen3 --sft-adapter outputs/cactus/qwen3/counsellor_sft/adapter --patient-rloo-run outputs/cactus/qwen3/patient_rloo/runs/PATIENT_RUN_DIRECTORY --resistance-tracker assets/resistance_forecaster --evidence-tracker assets/historical_cbt_evidence --output outputs/cactus/qwen3/counsellor_rloo --max-api-cost 50
```

Use the exact selected SFT adapter directories written by the preceding runs. The patient completion record must designate an accepted RLOO checkpoint before counsellor training starts. RLOO uses four responses per context, accumulation over four contexts, KL coefficient 0.05 and development-based selection. Patient naturalness combines the semantic score and SFT alignment at 80/20; counsellor naturalness uses 70/30. The source modules retain the role-specific rewards and adaptive weighting.

`--validate-only` checks the supplied data and adapter bindings without submitting API requests. Resume a run with the same command and output directory. Each output is bound to its data, code, adapters and cost setting; incompatible reuse is rejected. Checkpoint staging stays inside that output directory. Training writes its configuration, logs, development evaluations, selection summary and adapter completion records.
