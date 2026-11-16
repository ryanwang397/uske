# Baselines

All baselines are evaluated with the protocol of the main method (same prompts, data
splits, greedy decoding, token budgets, detector and utility harnesses).  Three groups are
compared in the paper (Section 4.1).

## 1. Knowledge-injection controls (Table 2)

They use the **same security concept vectors** as USKE and differ only in how the
knowledge is injected.

| Baseline | What it does | Command |
|---|---|---|
| **PKI** (prompt knowledge injection) | the secure prompt template (system + user turn with the language's CWE guidance) at inference time | `scripts/07_generate.py --template_mode secure` |
| **RSS** (residual-stream steering) | adds `gamma * d_l` to the residual stream after every decoder layer; `d_l` = secure-minus-vulnerable mean residual activation, extracted from the same contrast pairs (`baselines/extract_residual_directions.py`) | `scripts/07_generate.py --steer residual --directions <residual dirs> --gamma <g>` |
| **CLS** (component-level steering) | adds `gamma * s_{l,b}` to the outputs of the attention and MLP blocks of every layer, using the component-level vectors of step 4 (vectors added as extracted; `--normalize_steering` for unit vectors) | `scripts/07_generate.py --steer component --directions work/directions/<model>_<lang>_directions.pt --gamma <g>` |

`gamma` is searched over `{0.0, 0.1, ..., 2.0}` with the selection rule of Appendix B.3.

## 2. Target-only editing (Steer2Edit allocation, Table 3)

Same vectors, same editing pipeline, utility term removed:
`scripts/06_build_edits.py ... --beta 0`.  This isolates the effect of utility
regularization from the move from activation steering to weight editing (Appendix B.4).

## 3. Existing secure-code generation methods (Table 3)

* **SCS-Code** (Wendlinger et al., 2026) – single-layer difference-in-means steering;
  see `scs_code/build_scs_vector.py` (vector + layer selection), then
  `scripts/07_generate.py --steer residual --directions <scs dirs> --layers <selected> --gamma 1.0`.
  Default data are our ProSec contrast pairs (data-matched); `--source cybernative` uses the
  data of the original paper.
* **MoC** (Yu et al., 2025) – official implementation with probe-gated linear corrections;
  see `moc/README.md`.
* **ProSec-FT** (Xu et al., 2025) – training-based security alignment; see `prosec_ft/README.md`.
  The resulting checkpoint is evaluated with `scripts/07_generate.py --model_path <checkpoint>`.
