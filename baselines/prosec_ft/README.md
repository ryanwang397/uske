# ProSec-FT baseline (Xu et al., 2025, "ProSec: Fortifying Code LLMs with Proactive Security Alignment")

ProSec-FT is the training-based comparison in Table 3.  We do not retrain it: we evaluate
the CodeLlama-7B-Instruct checkpoint released by the ProSec authors,
`prosecalign/codellama-7b-inst-step-400` (Hugging Face), under our protocol.  The checkpoint
is decoded with the base `codellama/CodeLlama-7b-Instruct-hf` tokenizer and prompts, greedy,
with the same token budgets and scoring as every other row.

```bash
M=codellama-7b; L=python; CKPT=prosecalign/codellama-7b-inst-step-400
python scripts/07_generate.py --model $M --lang $L --dataset cyberseceval --model_path $CKPT --tag prosec_ft
python scripts/07_generate.py --model $M --lang $L --dataset humaneval    --model_path $CKPT --tag prosec_ft
python scripts/07_generate.py --model $M --lang $L --dataset mbxp         --model_path $CKPT --tag prosec_ft
```

ProSec's own training data are drawn from the same ProSec instruction pool, so the checkpoint
is reported on CyberSecEval and the utility benchmarks (cross-benchmark setting) rather than on
the ProSec test split.  Use the same filtered CyberSecEval file (`data/cyberseceval_filtered.json`)
as for every other row so that all methods are scored on identical prompts.  To train a ProSec model from scratch, follow the official repository
(https://github.com/PurCL/ProSec).
