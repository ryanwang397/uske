# USKE: Utility-Regularized Security Knowledge Editing for Code LLMs

Simplified reference implementation for the paper
**"Eliciting Latent Security Knowledge in LLMs for Secure Code Generation"** (under review).

USKE is a training-free method that turns a security concept vector into sparse,
rank-one weight edits on attention heads and MLP neurons.  Each component is scored by
its *security gain* (alignment with the security direction) and its *utility risk*
(how strongly an edit would perturb ordinary code generation, estimated from benign
coding activations), and edit magnitudes are allocated in closed form:

```
lambda_e = sign(g_e) * [ |g_e| - beta * r_e - rho * alpha ]_+ / ( rho * (1 - alpha) )
```

A component is edited only if its security gain exceeds a sparsity threshold raised by
its utility risk.  `beta = 0` recovers target-only allocation (the Steer2Edit rule).
The edits are merged into the weights, so the edited model runs with a standard forward
pass and no inference-time intervention.

This release contains (i) the main method, (ii) the baselines used for comparison
(knowledge-injection controls, target-only editing, SCS-Code, MoC, ProSec-FT) and
(iii) the ablations of design choices.  It is intentionally compact: it reproduces the
pipeline end-to-end for one model/language pair at a time and leaves out our internal
sweep, logging and analysis tooling (see *Scope* at the end).

## Repository layout

```
uske/                    core library
  languages.py           language settings, CWE lists, prompt templates (Appendix C)
  prompting.py           chat formatting, code-block extraction, model aliases
  data.py                ProSec / CyberSecEval / MBPP loading and the 40%/60% ProSec split
  activations.py         forward hooks on attention-head and MLP-neuron outputs
  directions.py          security concept vectors from contrast pairs (Sec. 2.1)
  benign_stats.py        benign activation statistics: per-head PCA, MLP second moments (Sec. 3.2)
  allocation.py          security gain, utility risk, closed-form allocation (Eq. 4-8)
  edits.py               build / apply / export sparse rank-one edits (Sec. 3.1, 3.3)
  steering.py            RSS and CLS steering hooks (knowledge-injection controls)
  generation.py          greedy batched generation (vLLM / transformers)
scripts/                 numbered end-to-end pipeline (see below)
baselines/               SCS-Code, MoC and ProSec-FT baselines
ablations/               ablations of design choices (Table 4)
configs/                 model aliases and the hyper-parameter grids of Appendix B.3
```


## Installation

```bash
conda create -n uske python=3.10 -y && conda activate uske
pip install -r requirements.txt
```

External tools (not bundled, used through their own CLIs):

* **Insecure Code Detector (ICD)** from Meta's PurpleLlama `CybersecurityBenchmarks`
  (security metric): `git clone https://github.com/meta-llama/PurpleLlama` and install
  its `CybersecurityBenchmarks` requirements (needs `weggli` and `semgrep`).
* **bigcode-evaluation-harness** (HumanEval / MultiPL-E pass@1):
  `git clone https://github.com/bigcode-project/bigcode-evaluation-harness`.
* **mxeval** (MBXP pass@1): `git clone https://github.com/amazon-science/mxeval` and
  run its `language_setup` script for the languages you evaluate.

Set `PURPLELLAMA_DIR`, `BIGCODE_HARNESS_DIR` and `MXEVAL_DIR` to the clone locations.

## Data

| Role | Dataset | Source | How we use it |
|---|---|---|---|
| Security (primary) | **ProSec** vulnerability-inducing instructions | HF `prosecalign/prosec-mixed-clm7b-inst` (train split, `benign == False`) | per language, `train_test_split(train_size=0.4, seed=20, shuffle=True)`: the **40 %** part is the adaptation split (contrast pairs + development), the **60 %** part is the held-out test set |
| Security (cross-benchmark) | **CyberSecEval** `instruct` test cases | `CybersecurityBenchmarks/datasets/instruct/instruct-v2.json` in Meta's PurpleLlama repo (commit 31a82e49) | `instruct-v2.json` restricted to the five languages and the CWE categories of the table below (`scripts/prepare_cyberseceval.py`); 388 test cases (C 32, C++ 53, Java 66, JavaScript 84, Python 153) |
| Benign activations | **MBPP** training tasks (task ids 601-974) | HF `Muennighoff/mbpp` (config `full`) | the base model answers each task with the baseline prompt (task text rewritten to the target language); prompt + own completion gives the benign activation statistics |
| Utility | **HumanEval-Multi** | OpenAI HumanEval (Python) and MultiPL-E `humaneval-{cpp,java,js}` via bigcode-evaluation-harness | pass@1, greedy, 512 new tokens; C is scored with the C++ tasks |
| Utility | **MXEval / MBXP** | https://github.com/amazon-science/mxeval (`mbxp` data) | pass@1, greedy, 512 new tokens |

ProSec split sizes (adaptation / test): Python 3583 / 5376, C 1297 / 1946, C++ 1575 / 2364,
Java 3079 / 4620, JavaScript 2014 / 3021.

Evaluated CWE categories (they follow the benchmarks and therefore differ across languages):

| Language | CWEs |
|---|---|
| Python | CWE-89, CWE-78, CWE-502, CWE-338 |
| C / C++ | CWE-377, CWE-119, CWE-78, CWE-676, CWE-338 |
| Java | CWE-502, CWE-78, CWE-352, CWE-79, CWE-200, CWE-295, CWE-611 |
| JavaScript | CWE-79, CWE-119, CWE-22, CWE-338 |

Security is measured with the rule-based Insecure Code Detector (ICD) of CybersecurityBenchmarks
(`scripts/02_detect.py`): the code block of a generation is scanned with the detector rules of its
language, and the program counts as *vulnerable* if at least one insecure pattern is reported
(`--target_cwe_only` restricts this to the prompt's target CWE); the vulnerable-code ratio is the
fraction of vulnerable generations.  Programs on which the detector times out are excluded.  All datasets are
downloaded by their owners' tooling; nothing is redistributed here.


## Pipeline (one model, one language)

All scripts take `--model` (alias from `configs/models.yaml` or any HF id), `--lang`
(`python | c | cpp | java | js`) and write under `--work_dir` (default `work/`).

```bash
M=codellama-7b; L=python
# 1. completions on the ProSec 40% adaptation split with the baseline and the secure template
python scripts/01_generate_contrast_responses.py --model $M --lang $L --template_mode baseline
python scripts/01_generate_contrast_responses.py --model $M --lang $L --template_mode secure
# 2. label them with the ICD
python scripts/02_detect.py --input work/responses/${M}_${L}_baseline.jsonl --output work/detected/${M}_${L}_baseline.jsonl
python scripts/02_detect.py --input work/responses/${M}_${L}_secure.jsonl   --output work/detected/${M}_${L}_secure.jsonl
# 3. contrast pairs: baseline vulnerable AND secure clean
python scripts/03_build_contrast_pairs.py --model $M --lang $L
# 4. security concept vectors (attention / MLP blocks of every layer) + mean block inputs
python scripts/04_extract_directions.py --model $M --lang $L
# 5. benign statistics for utility risk (MBPP completions -> per-head PCA, MLP second moments)
python scripts/05_benign_stats.py --stage generate --model $M --lang $L
python scripts/05_benign_stats.py --stage stats    --model $M --lang $L --top_k 16
# 6. utility-regularized edit allocation (attention-only example); --beta 0 = target-only allocation
python scripts/06_build_edits.py --model $M --lang $L --rho_attn 0.30 --alpha 0.80 --beta 0.15
# 7. generate with the edited model on the four benchmarks (merges the edits into the weights)
python scripts/07_generate.py --model $M --lang $L --dataset prosec_test   --edit_pack work/edits/${M}_${L}/<pack>.pt
python scripts/07_generate.py --model $M --lang $L --dataset cyberseceval  --edit_pack work/edits/${M}_${L}/<pack>.pt
python scripts/07_generate.py --model $M --lang $L --dataset humaneval     --edit_pack work/edits/${M}_${L}/<pack>.pt
python scripts/07_generate.py --model $M --lang $L --dataset mbxp          --edit_pack work/edits/${M}_${L}/<pack>.pt
#    security: scripts/02_detect.py on the ProSec / CyberSecEval outputs (--target_cwe_only)
#    utility : bigcode-evaluation-harness --load_generations_path / mxeval evaluate_functional_correctness
# 8. pick the configuration with the paper's selection rule
python scripts/08_select_config.py --csv dev_results.csv --base_pass1 <base pass@1>
```

The edited model is a standard checkpoint: `scripts/06_build_edits.py --export_dir <dir>`
writes it with `save_pretrained`, so it can be served with transformers or vLLM directly.


## Baselines (Section 4.1; details in `baselines/README.md`)

| Group | Baseline | How to run |
|---|---|---|
| Knowledge-injection controls (same vectors, different injection point) | **PKI** prompt injection | `scripts/07_generate.py --template_mode secure` |
|  | **RSS** residual-stream steering | `baselines/extract_residual_directions.py`, then `scripts/07_generate.py --steer residual --gamma <g>` |
|  | **CLS** component-level steering | `scripts/07_generate.py --steer component --directions work/directions/... --gamma <g>` |
| Target-only editing | **Steer2Edit allocation** (`beta = 0`) | `scripts/06_build_edits.py ... --beta 0` |
| Existing secure-code methods | **SCS-Code** (Wendlinger et al., 2026) | `baselines/scs_code/build_scs_vector.py`, then `--steer residual --layers <L> --gamma 1` |
|  | **MoC** (Yu et al., 2025), official code | `baselines/moc/README.md` |
|  | **ProSec-FT** (Xu et al., 2025), training-based | `baselines/prosec_ft/README.md`, then `scripts/07_generate.py --model_path <checkpoint>` |

The steering controls use the strength grid `gamma in {0.0, 0.1, ..., 2.0}`; every baseline
is scored with the same detector, harnesses and decoding settings as USKE.  Steering is
implemented with transformers forward hooks (`uske/steering.py`), so the vectors are added at
every position of the prompt and of every generated token.


## Ablations of design choices (Section 4.6, Table 4)

`ablations/run_ablations.sh` builds the edit packs of Table 4 on CodeLlama-7B-Instruct /
Python, starting from the selected Python configuration (attention-only edits,
`rho_attn = 0.20`, `alpha = 0.90`, `beta = 0.15`, `K = 16`, conditional input direction) and
changing one design choice at a time:

* **input direction** `k_hat`: `--key_mode conditional` (ours, `W_e^T s_hat`) vs. `random`
  (Gaussian, normalized) vs. `mean` (mean component input);
* **sparsity penalty**: near-L1 `--alpha 0.99`, L2-only `--alpha 0`, Elastic-Net `--alpha 0.90` (ours);
* **PCA directions** for the attention utility risk: `--pca_top_k 4 | 8 | 16`;
* **target-only allocation** (Steer2Edit rule): `--beta 0`;
* sensitivity to `beta` and `alpha` around the selected configuration (Appendix I.2).

Each pack is evaluated exactly like the main method (CyberSecEval-Python vulnerable ratio,
HumanEval-Python pass@1).


## Hyper-parameter search and selection (Appendix B.3)

`configs/search_grids.yaml` lists the grids used in the paper.  For every model and
language we search three editing modes with `rho_attn` / `rho_mlp` as in Table 7
(`-1.0` disables a component family), combined with `beta in {0.05, 0.15, 0.25}` and
`alpha in {0.70, 0.75, 0.80, 0.85, 0.90}`; attention utility risk uses `K = 16`
principal directions.  The steering controls (RSS, CLS) search the strength
`gamma in {0.0, 0.1, ..., 2.0}`.

| Model | Mode | `rho_attn` | `rho_mlp` |
|---|---|---|---|
| CodeLlama-7B-Instruct | attention-only | 0.10 : 0.05 : 0.60 | -1.0 |
|  | MLP-only | -1.0 | 0.10 : 0.05 : 0.80 |
|  | combined | 0.16 : 0.02 : 0.24 | 0.35 : 0.05 : 0.55 |
| Qwen2.5-Coder-3B-Instruct | attention-only | 0.10 : 0.05 : 0.60 | -1.0 |
|  | MLP-only | -1.0 | 0.10 : 0.05 : 0.80 |
|  | combined | 0.16 : 0.02 : 0.36 | 0.35 : 0.05 : 0.55 |

**Selection rule.** Candidate configurations are compared on development data (security
measured with the detector, utility with pass@1).  Among the configurations that retain at
least **90 %** of the base model's pass@1, we select the one with the **lowest vulnerable-code
ratio**; ties are broken by the higher pass@1 (`scripts/08_select_config.py`).  The constraint
rules out degenerate edits that lower the detector count by producing non-functional code
(e.g. the near-L1 ablation).

Decoding is greedy everywhere (2048 new tokens for the security benchmarks, 512 for the
utility benchmarks, seed 20).


## Scope of this release

This is a deliberately compact version of our research code.  It contains the complete
method and everything needed to reproduce the protocol (data splits, prompts, scoring,
baselines, ablations, search grids), but not:

* our Slurm launchers, result files, per-run logs and the bookkeeping scripts that merge
  sweeps into tables;
* the preliminary component-overlap analysis of Section 2.2 (built on the public
  Safety-Neuron code base) and the visualization scripts of Section 4.4;
* the LLM-as-a-judge, manual-audit and joint-evaluation protocols of Appendix H, which
  depend on external services and human annotation.

Scoring is uniform for all methods: generated programs are passed to the detector and to the
utility harnesses as produced (only markdown fences are stripped for the detector), with no
method-specific post-processing.  Numbers can differ slightly from the paper across GPU types
and inference engines (greedy decoding is not bit-exact across kernels); relative comparisons
are stable.


## Citation

```bibtex
@article{uske2026,
  title   = {Eliciting Latent Security Knowledge in LLMs for Secure Code Generation},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```
