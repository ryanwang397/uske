#!/usr/bin/env bash
# Ablation of design choices (Section 4.6, Table 4; settings in Appendix I.2).
# Every run changes exactly one design choice relative to the reference configuration;
# edits are built with scripts/06_build_edits.py, then evaluated like any other edit pack
# (scripts/07_generate.py --edit_pack ... on CyberSecEval-Python / HumanEval-Python).
#
# Reference configuration (the selected Python edit of Table 2): CodeLlama-7B-Instruct, Python,
# attention-only edits with rho_attn=0.20, Elastic-Net alpha=0.90, utility beta=0.15, K=16
# principal directions, conditional input direction (73 edited heads).
set -euo pipefail
MODEL=${MODEL:-codellama-7b}
LANG_=${LANG_:-python}
RHO_ATTN=${RHO_ATTN:-0.20}      # reference edit; other values: attention-only grid in configs/search_grids.yaml
WORK=${WORK:-work}
run() { python scripts/06_build_edits.py --model "$MODEL" --lang "$LANG_" --work_dir "$WORK" --rho_attn "$RHO_ATTN" "$@"; }

# (1) input direction k_hat: conditional (ours) vs. random vs. mean activation
run --alpha 0.90 --beta 0.15 --key_mode conditional
run --alpha 0.90 --beta 0.15 --key_mode random
run --alpha 0.90 --beta 0.15 --key_mode mean

# (2) sparsity penalty: near-L1 (alpha=0.99), L2-only (alpha=0), Elastic-Net (alpha=0.90, ours)
run --alpha 0.99 --beta 0.15
run --alpha 0.00 --beta 0.15

# (3) number of principal directions for the attention utility risk
run --alpha 0.90 --beta 0.15 --pca_top_k 4
run --alpha 0.90 --beta 0.15 --pca_top_k 8

# (4) target-only allocation (Steer2Edit rule) = no utility term
run --alpha 0.90 --beta 0.0

# (5) sensitivity to beta (at alpha=0.90) and alpha (at beta=0.15); Appendix I.2 reports the
#     neighbourhood beta in [0.15, 0.30] and alpha in [0.80, 0.90] of the selected configuration
for b in 0.0 0.05 0.10 0.20 0.25 0.30; do run --alpha 0.90 --beta "$b"; done
for a in 0.60 0.65 0.70 0.75 0.80 0.85 0.95; do run --alpha "$a" --beta 0.15; done
