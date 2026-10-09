# Reproduction details

## Data layout

The downloader creates this directory tree. An existing copy can be supplied with `--data-dir`.

```text
data/raw/cwru/
  Normal Data/Normal_0.mat
  12k Drive End Bearing Fault Data/
    IR007_0.mat
    IR014_0.mat
    IR021_0.mat
    OR007@6_0.mat
    OR014@6_0.mat
    OR021@6_0.mat
    B007_0.mat
    B014_0.mat
    B021_0.mat
```

`configs/data_files.json` binds each semantic name to the original download URL and exact byte hash. Downloaded files are checked before use. Existing files with wrong hashes are preserved and rejected, not silently replaced. HTTPS verification remains enabled.

Preprocessing chooses the unique MATLAB variable ending in `_DE_time`. It resamples the healthy waveform first, selects temporal split boundaries by integer truncation, and then forms nonoverlapping 1,024-sample windows. Each window is standardized independently with a standard-deviation floor of 1e-6. Remainders at the end of each split are discarded. All transformed data are float32.

The result is 71 source-training and 41 source-validation windows per class. Target support/query counts are 17/35 for healthy and 35/71 for each ball-fault class. Saved manifests contain precise sample boundaries and file hashes. No global scaler is fit to target data.

## Software and determinism

Reference platform: Windows, Python 3.10.11, PyTorch 2.5.1+cu124, CUDA runtime 12.4, cuDNN 9.1.0, NumPy 1.26.4, SciPy 1.11.4 and NVIDIA RTX A4000. Each new run writes its actual environment to `runs/<name>/environment.json`.

Python, NumPy and PyTorch random generators are seeded at the beginning of each run. CPU thread count is two. cuDNN benchmarking and TF32 are disabled, and deterministic cuDNN convolution selection is enabled, matching the original experiment. Global `torch.use_deterministic_algorithms` was not enabled in the original experiment and is not added during packaging. Bitwise agreement across machines is not guaranteed.

Source episode sampling uses an independent NumPy generator seeded by the training seed. Source-validation sampling uses seed 74019. Target episode seed is 20261008. Query indices are frozen once, and the first K entries of the same ten-example support list define the three shot settings.

## Advanced commands

Prepare once, then run independent algorithm/backbone combinations. The following two training commands may be launched in separate terminals with two GPUs. They write to different run directories under the same prepared output; do not run `prepare` or edit frozen source/configuration while either is running.

```bash
python benchmark.py prepare --output outputs/manual
python benchmark.py check --output outputs/manual --device cuda:0
python benchmark.py run --output outputs/manual --rep 1d --device cuda:0
python benchmark.py run --output outputs/manual --rep 2d --device cuda:1
```

For one final run, add `--method maml --rep 1d --seed 17`. Aggregation expects all 24 final runs, so use it after both full representation commands finish:

```bash
python benchmark.py aggregate --output outputs/manual
python audit.py --output outputs/manual --data-dir data/raw/cwru --compare reference
python report.py --results outputs/manual/results.json --output outputs/manual/report --plots
```

For a strict same-environment comparison, add `--require-exact` to the audit command. It fails unless all 864,000 predicted query labels match the reference: 24 trained models × 3 shot settings × 200 episodes × 60 queries. The normal audit records mismatches without disguising them as exact reproduction.

Source-validation smoke tests use a separate prepared output directory:

```bash
python benchmark.py prepare --output outputs/smoke
python benchmark.py smoke --output outputs/smoke --method protonet --rep 1d --device cuda:0
```

Smoke uses seed 999 and 600 source episodes, and performs no target evaluation. It is excluded from final aggregation. Use a fresh output directory for each independent smoke command.

## Evidence files

| File | Purpose |
|---|---|
| `reference/results.json` | Full-precision aggregate values and protocol |
| `reference/runs/*/best.pt` | Selected weights and checkpoint metadata |
| `reference/runs/*/training.json` | Six source-validation checkpoints and selection evidence |
| `reference/runs/*/adaptation_selection.json` | Source-only rate search and selected adaptation rate |
| `reference/runs/*/predictions.json` | Every target prediction at every shot count |
| `reference/test_episodes.json` | Paired query and nested support indices |
| `reference/data_manifest.json` | Portable paths, raw-file hashes and temporal split boundaries |
| `reference/checksums.json` | SHA-256 checksums of the bundled evidence |
| `reference/provenance.json` | Original environment, frozen-source fingerprints and revision history |
| `outputs/*/frozen.json` | Fingerprint of code, configuration and prepared data for a new run |

The original experiment contained machine-specific paths. Packaging replaces those paths with portable relative paths in the protocol and manifests, leaving result arrays, checkpoints and episode indices intact. Original protocol/code hashes are retained as historical identifiers in `reference/provenance.json`; they are not presented as hashes of the refactored repository source.

MAML and ProtoNet completed before the final original Reptile/RelationNet revision and were retained unchanged. The original SGD Reptile and MLP/MSE RelationNet performed poorly on source validation. Reptile was revised to Adam minibatches, and RelationNet to a convolutional comparison head with cross-entropy, using source validation before evaluating their final revision. The reference includes only final seeds 17, 29 and 43.

## CI versus experiment verification

CI checks full MAML meta-gradients and parameter isolation, Reptile's Adam update, metric-method class permutation, paired episode construction, corruption detection and all saved reference predictions. It runs on CPU and does not download CWRU or retrain the GPU benchmark. The fresh-data and full-training verification performed when packaging this repository is recorded separately in `docs/verification.json`.
