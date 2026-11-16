# MoC baseline (Yu et al., 2025, "A Mixture of Linear Corrections Generates Secure Code")

We run the **official implementation** (https://github.com/viviable/MoC) with its default
recipe and only add thin glue code:

* `build_moc_data.py` – converts our ProSec contrast pairs into the SVEN-style per-CWE
  paired files the official code expects (data-matched variant used in the paper's
  Table 3).  The original SVEN data can be used instead by pointing `--data_root` at it.
* `get_representation_blocks.py` – official step 1 with block activations read from the
  decoder-layer outputs (see the docstring); everything else is imported unchanged.
* `gen_moc.py` – batched greedy generation on our benchmarks with the official
  `MoCSteerer` hook (probe-gated per-token corrections, `alpha(t) = exp(-0.05 t)`).

```bash
git clone https://github.com/viviable/MoC && export MOC_SRC=$PWD/MoC/code/src
python baselines/moc/build_moc_data.py --model codellama-7b --lang python          # -> work/moc_data/python
CWES=$(python -c "import json;print(' '.join(json.load(open('work/moc_data/python/cwes.json'))))")
export MOC_CWE_DESC_JSON=$PWD/work/moc_data/cwe_descriptions.json
python baselines/moc/get_representation_blocks.py --model_name codellama/CodeLlama-7b-Instruct-hf \
    --data_root work/moc_data/python --save_path work/moc/repr --few_shot_pairs 2 --splits train val \
    --cwes $CWES --pool last_token
WORK=$PWD/work
(cd $MOC_SRC && python train_probe.py --repr_dir $WORK/moc/repr --save_path $WORK/moc/probes --kind linear \
    --layers -1 --val_size 0.2 --epochs 100 --lr 1e-3 --wd 1e-4 --optimizer adam --cwes $CWES)
(cd $MOC_SRC && python corrections.py --repr_dir $WORK/moc/repr --probe_dir $WORK/moc/probes \
    --save_path $WORK/moc/corrections --method n --cwes $CWES)   # `n` = official default; `g` (group mean) also available
python baselines/moc/gen_moc.py --model codellama-7b --lang python --dataset prosec_test \
    --probe_dir work/moc/probes --corrections work/moc/corrections/static_n.pt --cwes $CWES
```

Settings follow the official `run_all.sh` defaults: linear probes trained on the last
block (also the steered block), 100 epochs, lr 1e-3, weight decay 1e-4, Adam, 20% internal
validation; static corrections `n` (probe normal vector, default) or `g` (group mean);
decay rate 0.05.  Generated files have the same format as `scripts/07_generate.py` outputs
and are scored with `scripts/02_detect.py` and the utility harnesses.
