# Llama 3 Korean MCQ fine-tuning experiment

This experiment fine-tunes `beomi/Llama-3-Open-Ko-8B` for the Korean
general-knowledge multiple-choice task in this repository.

## Training setup

- Supervised fine-tuning with TRL `SFTTrainer`
- LoRA adapters on all linear layers
- 4-process FSDP training
- FP16 mixed precision
- Sequence length: 512
- One training epoch

The shared competition data lives in `../../baseline/data`. To run this
experiment, either copy or link `train_data.csv` and `test_data.csv` into a
local `data/` directory, then run:

```bash
sh run.sh
```

Model checkpoints, virtual environments, W&B logs, and local datasets are
excluded from version control. Add `--resume_from_checkpoint` to `run.sh`
only when resuming from a local checkpoint.
