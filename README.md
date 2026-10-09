# PHM few-shot fault diagnosis benchmark

A reproducible PyTorch benchmark for **MAML, Reptile, Prototypical Networks and Relation Networks**, each with **1D and 2D CNN encoders**, evaluated at **1, 5 and 10 shots** on one CWRU setup.

This repository contains the code and measured results used in the PHM lecture's common-benchmark example. These are **new experiments**, not the accuracy numbers reported by the cited papers. The source/target class partition follows MetaFD's CWRU T2 task. Data handling, architecture variants and training budget are specified below.

## What is included

- Complete preprocessing, episode sampling, meta-training, adaptation and evaluation code.
- Download script for exactly ten original CWRU files, with SHA-256 verification.
- Fixed source/target splits and paired, nested 1/5/10-shot target episodes.
- All **24 trained checkpoints**, training histories, source-validation selections and target predictions.
- Result reconstruction from predictions, source/data fingerprints, CSV tables and optional plots.
- CPU tests and a GitHub Actions workflow. GPU retraining is an explicit command, not part of CI.

Raw CWRU data are downloaded from the original host and remain outside Git. This repository covers the benchmark and its result plots; it does not require the slide authoring software.

## Install

Use **Python 3.10**. The reference runs used Python 3.10.11, PyTorch 2.5.1+cu124, NumPy 1.26.4 and SciPy 1.11.4 on Windows with an NVIDIA RTX A4000. See [reproducibility details](docs/reproducibility.md).

```bash
git clone https://github.com/MainakMallick/phm-few-shot-benchmark.git
cd phm-few-shot-benchmark
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell.

For the CUDA 12.4 build used in the reference experiment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For CPU-only execution, substitute `https://download.pytorch.org/whl/cpu` in the first command. Use `--device cpu` in the commands below. Training and gradient-based target adaptation are slower on CPU.

Optional scientific plots:

```bash
python -m pip install -r requirements-plots.txt
```

## Three ways to reproduce

### 1. Verify the reported numbers without data or training

```bash
python audit.py --reference
python report.py --output outputs/reference-report
python -m unittest discover -s tests -v
```

The audit checks reference-file hashes, every saved prediction, episode accuracy, checkpoint selection, class/split separation and all 24 aggregate result rows. It does not run model inference. The report reconstructs the table in Markdown and CSV. Add `--plots` to produce PNG/SVG accuracy plots with seed-SD error bars.

### 2. Evaluate the included checkpoints on freshly prepared data

```bash
python reproduce.py --mode checkpoints --download --device cuda:0 --output outputs/checkpoint-evaluation
```

This downloads and verifies the original files, rebuilds normalized windows and episodes, runs implementation checks, reselects adaptation rates using source validation, evaluates every checkpoint and compares all predictions with the reference. It saves a detailed comparison in `outputs/checkpoint-evaluation/audit.json`.

### 3. Train every model from scratch

```bash
python reproduce.py --mode train --download --device cuda:0 --output outputs/retraining
```

This runs all four algorithms, both backbones and all three seeds, then evaluates 1/5/10 shots, aggregates results and audits them. Add `--plots` if the optional plotting dependency is installed. The pipeline records software/device metadata per run and resumes completed models. Interrupted models restart from their seed; there is no mid-training optimizer resume.

To use existing data, omit `--download` and pass `--data-dir /path/to/cwru`. See [data layout](docs/reproducibility.md#data-layout). The same checksums are required.

Use different output directories for training, checkpoint evaluation and smoke tests. The bundled `reference/` is protected from benchmark writes. `prepare` refuses to overwrite a directory containing runs. Code, configuration or prepared-data changes invalidate an existing experiment fingerprint.

## Experimental setup

| Item | Fixed setting |
|---|---|
| Dataset | CWRU drive-end vibration, 0 HP, 12 kHz fault recordings |
| Source classes | IR007, IR014, IR021, OR007@6, OR014@6, OR021@6 |
| Target classes | Normal, B007, B014, B021 |
| Class meaning | Inner-race, outer-race and ball faults at 0.007, 0.014 and 0.021 inch diameters |
| Source split | First 60% of each recording for training, last 35% for validation, 5% gap |
| Target split | First 30% for support, last 60% for query, 10% gap |
| Healthy signal | Resample native 48 kHz data to 12 kHz with `resample_poly(1, 4)` before windowing |
| Windows | 1,024 samples, no overlap, mean removal and standardization per window |
| 1D input | 1 × 1,024 waveform |
| 2D input | The same samples reshaped row-major to 1 × 32 × 32; no STFT or scalogram |
| Source episode | 4 of 6 classes, 5 support and 5 query windows per class |
| Training | 1,200 episodes per model per seed; seeds 17, 29 and 43 |
| Selection | Best of six source-validation checkpoints; first tie wins |
| Target episode | All 4 held-out classes; 1/5/10 support and 15 query windows per class |
| Evaluation | 200 fixed episodes; identical query indices and nested support subsets across models/shots/seeds |
| Reporting | Mean accuracy and sample SD across the three training-seed means |

Healthy and ball-fault classes never enter meta-training or source-validation selection. Meta-testing includes labeled target support windows; the separate query windows provide the test accuracy.

## Models

All encoders use four blocks of 64-filter convolution (kernel 3, padding 1), GroupNorm (8 groups), ReLU and factor-two max-pooling. Operation dimensionality matches the input. GroupNorm prevents normalization statistics from depending on the other query examples.

| Method | Meta-training | Target classification |
|---|---|---|
| MAML | Full second-order gradients through 3 support SGD steps at 0.05; Adam outer rate 0.001 | 5 SGD support steps; rate chosen from 0.01, 0.05, 0.1 using source validation |
| Reptile | 5 Adam support-minibatch steps at 0.001, betas (0, 0.999), batch size 10; interpolate initialization with epsilon 1.0 to 0.1 | 5 Adam steps on all support; rate chosen from 0.0005, 0.001, 0.005 using source validation |
| ProtoNet | Episodic query cross-entropy using negative mean squared Euclidean distances | Average support embeddings by class; classify by nearest prototype; no target gradient updates |
| RelationNet | Joint encoder/comparison training with query cross-entropy over four learned relation logits | Average support maps by class and score each query/class pair; no target gradient updates |

MAML/Reptile/ProtoNet use 256-dimensional pooled embeddings. MAML/Reptile add a four-class linear classifier. RelationNet keeps spatial maps and concatenates support/query maps before two extra 64-channel convolution/GroupNorm/ReLU/pooling blocks, global average pooling and FC64→8→1 with LeakyReLU. See the clearly separated classes/functions in [benchmark.py](benchmark.py).

`configs/protocol.json` documents this fixed benchmark. This is not a general configuration-driven hyperparameter-sweep framework: architecture sizes and several protocol constants are deliberately fixed in the implementation. Changes require a new experiment directory and corresponding code/config updates.

## Results

The complete table is in [docs/results.md](docs/results.md), with full-precision values in [reference/results.json](reference/results.json). The slide tables and charts round values to two decimals. The optional plots also show sample SD across training seeds.

## Interpretation and limits

- Support and query windows come from separated portions of the **same recording** per class. This does not measure generalization to independent bearings, loads or datasets.
- The common source episodes and training count do not match parameter counts or gradient computation across algorithms. Reptile's outer update does not use the sampled source queries.
- This is a fixed-budget teaching benchmark, not a claim of converged training or best possible performance for any method.
- Three-seed SD describes sensitivity to training initialization, not uncertainty across independent assets. Episodes reuse a limited pool of recorded windows.
- Seeds, hashes and saved predictions make results auditable. Retraining on another device, OS or PyTorch build can differ numerically. Reference inference and fresh training are compared explicitly rather than assumed to match.

## References

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file), including the [12 kHz fault files](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) and [healthy files](https://engineering.case.edu/bearingdatacenter/normal-baseline-data).
- [MetaFD](https://github.com/fyancy/MetaFD), for the CWRU T2 class partition. The code here is a standalone PyTorch implementation and does not import MetaFD or learn2learn.
- [Finn et al., MAML, ICML 2017](https://proceedings.mlr.press/v70/finn17a.html).
- [Nichol et al., On First-Order Meta-Learning Algorithms, 2018](https://arxiv.org/abs/1803.02999).
- [Snell et al., Prototypical Networks, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html).
- [Sung et al., Learning to Compare, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html).

Model operations are controlled variants of these algorithms, not exact reproductions of their original image experiments. CWRU data remain subject to the original provider's terms.
