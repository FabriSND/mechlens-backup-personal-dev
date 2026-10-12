# Debugging SAE Cache Parity

A BF16-only difference in SAE feature projections does not by itself establish a cache bug. First compare raw cached and live-hook activations from the **same forward pass**, then compare feature projections using identical arithmetic.

This guide documents a controlled GPT-2 experiment prompted by the June 10, 2026 GitHub notification for TransformerLens issue #1108. The notification reported a difference at `blocks.8.mlp.hook_post` under BF16 but not FP32. It mentioned an attached reproducer, but the retrieved message contained no downloadable attachment. The original model, inputs, trained SAE, and runtime were unavailable.

## Fixed experimental inputs

The code baseline is commit `33387b7def1ead3a8cba532a729da5599e875873`. The model and random SAE below are authorized experimental assumptions, not facts supplied by the original report.

| Input | Value |
|---|---|
| Model | Stock GPT-2 small, loaded with `from_pretrained_no_processing` |
| Model revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Hook | `blocks.8.mlp.hook_post` |
| Seed | 42 |
| Forward dtypes | FP32 and BF16 |
| Device used for measurements | CPU |
| Prompt | `The quick brown fox jumps over the lazy dog.` |
| Tokenization | BOS prepended; 11 tokens |
| Activation shape | `[1, 11, 3072]` |
| Encoder matrix | `W_enc[3072, 4096]` |
| Decoder bias | `b_dec[3072]` |
| Parameter generation | Standard-normal draws from one seed-42 CPU generator: matrix first, bias second |
| Parameter dtype | Cast to the forward dtype before feature calculation |
| Feature calculation | FP32 linear projection; no encoder bias or nonlinearity |
| Execution mode | Evaluation and inference mode; feature autocast disabled |

The exact token IDs are:

```text
[50256, 464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
```

The numerical results below were measured with Python 3.12.14, PyTorch 2.14.1+cpu, and Transformers 5.19.0. These runtime versions were experimental choices, not manifest pins. A normal repository installation uses its own lockfile; do not assume that installation reproduces this exact runtime. The script records the versions actually used.

## Reproduce the experiment

Use the investigation branch, which includes the runnable script and integration test; the baseline commit alone does not include these additions.

```bash
git clone --branch fix/sae-bf16-cache-parity https://github.com/FabriSND/mechlens-backup-personal-dev.git
cd mechlens-backup-personal-dev
uv sync
uv run python repros/sae_cache_parity.py --device cpu --output sae_cache_parity_results.json
uv run pytest tests/integration/test_sae_cache_parity.py -v -s --tb=short
```

The first execution requires access to the pinned public GPT-2 checkpoint. If using an existing configured checkout, source its `.env` before Hugging Face operations. Once model files are cached, `HF_HUB_OFFLINE=1` can be used to run offline.

For the CPU-only device and the small GPT-2 model used here, the script loads FP32 weights without folding or centering and casts the model to each tested dtype. It registers a live observer, clones the observed activation into separate storage, and calculates live features inside that observer. It then calls `run_with_cache` for the same forward and computes features from the returned cache.

The script writes a JSON array with one record per dtype. It exits successfully only when both records satisfy its exact parity, finiteness, dtype, single-hook-call, and independent-storage checks. A mismatch produces a nonzero exit after writing the records; setup or model-loading errors must not be interpreted as numerical results.

| JSON field | Meaning |
|---|---|
| `dtype`, `device`, `shape` | Raw activation dtype, device, and shape |
| `max_abs_error`, `mean_abs_error`, `allclose` | Cached-path versus live-hook feature projection errors; zero tolerances |
| `raw_cache_matches_live` | Exact raw activation equality |
| `normalized_feature_path_matches_float32` | Exact equality with an FP32 reference computed from the same forward's live activation |
| `raw_cache_errors` | Separate raw activation error measurements |
| `subtract_before_cast_errors` | Non-gating diagnostic using BF16 subtraction before promotion |
| `assumption_based` | Identifies the experiment as using substituted inputs |

The FP32 reference field does **not** compare a BF16 model forward against a separate FP32 model forward.

## Measured results

| Forward dtype | Raw max / mean absolute error | Feature max / mean absolute error | Unequal raw / feature elements |
|---|---|---|---|
| FP32 | 0.0 / 0.0 | 0.0 / 0.0 | 0 / 0 |
| BF16 | 0.0 / 0.0 | 0.0 / 0.0 | 0 / 0 |

Each raw comparison covered 33,792 elements; each feature comparison covered 45,056 elements. All tensors were finite and each live observer fired exactly once. The integration test run completed with `4 passed, 1 warning in 15.17s`; the warning concerned type-checking instrumentation in `SVDInterpreter.py`.

## Diagnosis: promotion order can create a BF16-only difference

Compare these expressions using the same activation and parameter values:

```python
# Promote every operand before doing arithmetic.
reference = (activation.float() - b_dec.float()) @ W_enc.float()

# BF16 operands round the subtraction result before it reaches FP32.
late_cast = (activation - b_dec).float() @ W_enc.float()
```

The controlled diagnostic measured:

| Operand dtype | Max absolute feature difference | Mean absolute feature difference | Unequal elements |
|---|---:|---:|---:|
| FP32 | 0.0 | 0.0 | 0 / 45,056 |
| BF16 | 0.4064788818359375 | 0.0772847831249237 | 45,056 / 45,056 |

Subtracting two BF16 operands first produces a rounded BF16 result. Converting that result to FP32 cannot recover the discarded information. Matrix multiplication propagates the difference into the feature projections.

This demonstrates a mechanism consistent with the reported FP32/BF16 pattern. It does not establish the original report's cause, and these magnitudes apply only to the random SAE used here.

The pinned cache-saving path in `HookedRootModule.get_caching_hooks` uses `tensor.detach().to(device)`, with optional slicing. It requests no dtype conversion and performs no SAE arithmetic. `ActivationCache.__getitem__` returns the stored tensor directly. No BF16-specific cache rounding was observed.

## Safe promotion order

Use exactly the same parameter values and arithmetic in both paths:

```python
with torch.autocast(device_type=activation.device.type, enabled=False):
    features = ((activation.float() - b_dec.float()) @ W_enc.float())
```

Promote **before subtraction**, and promote the encoder matrix **before matrix multiplication**. Disable autocast for this expression so the requested FP32 path is explicit. Casting the subtraction result or the completed projection afterward is not equivalent.

If SAE parameters have already been cast to BF16, their promotion to FP32 preserves those rounded values; it does not restore the original FP32 checkpoint. Both paths must use the same parameter version and precision history.

## If a mismatch remains

1. Compare raw activations first, using the same input tokens and the same forward. Check dtype, shape, device, positions, and hook-call count.
2. Clone the live observation. Cache tensors can share storage because `detach()` is not a clone; later in-place changes can otherwise obscure what was observed.
3. Inspect hook registration order and any hook that modifies or replaces activations. Two observers separated by a modifying hook need not see the same value.
4. If raw activations match, compare SAE parameter values, promotion order, autocast context, and any additional nonlinearities or preprocessing.
5. If comparing separate forwards, also examine model mode, stochastic operations, backend settings, and input alignment.

A passing synthetic regression establishes parity for these fixed experimental inputs. It does not close issue #1108 or provide release clearance without the original reproducer and acceptance evidence. No cache-internal change is justified by the measured results here.
