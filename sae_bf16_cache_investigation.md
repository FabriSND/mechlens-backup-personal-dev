# SAE BF16 cache-parity investigation

## Conclusion

The tested same-forward raw cache and live-hook activations match exactly in FP32 and BF16. Their SAE feature projections also match exactly when both paths promote every operand to FP32 before subtraction and matrix multiplication.

A controlled diagnostic produced a BF16-only feature difference by subtracting in BF16 before casting to FP32. This establishes a viable mechanism, not the confirmed cause of the original report. The original reproducer was unavailable, so this investigation does not establish a release PASS or resolve issue #1108.

## Report provenance and scope

The source is a GitHub notification from notifications@github.com dated 2026-06-10 at 08:30 UTC, with subject:

> [TransformerLensOrg/TransformerLens] bug: SAE feature dashboard mismatch on cached run vs live run (#1108)

The notification says that nkechi-of opened issue #1108 and reported slightly different SAE feature activations between a cached run and a live run at `blocks.8.mlp.hook_post` under BF16, with FP32 working correctly. It suggests dtype-promotion order as a possibility. It does not establish that the compared activations came from the same forward pass.

The accessible Gmail message ID is `19eb0a713038c8fc`. The supplied manifest instead identifies `19eb0a712af9b0b5`, which returned 404. The accessible notification matches the manifest's subject and date. No current issue-tracker content was used as evidence.

The manifest pins repository `jesuswork60-create/mechlens-backup-personal-dev` at commit `33387b7def1ead3a8cba532a729da5599e875873` and lists:

- `transformer_lens/hook_points.py`
- `transformer_lens/ActivationCache.py`
- `transformer_lens/HookedTransformer.py`
- `transformer_lens/components/mlps/mlp.py`
- `tests/acceptance/model_bridge/compatibility/test_run_with_cache.py`
- `tests/unit/test_hook_points.py`

All six files were accessible at that commit. The executable experiment used the complete code snapshot at that commit, including the cache implementation in `HookedRootModule.py`. The investigation branch in `FabriSND/mechlens-backup-personal-dev` starts from the same commit.

## Reproducer availability and explicit assumptions

Although the notification says “Reproducer attached,” its retrieved MIME payload contains no attachment. The manifest explicitly warns that the staged notification has no downloadable reproducer. No original model specification, input tokens, trained SAE weights, runtime pins, or locked output templates were supplied.

The user authorized an assumption-based substitute using stock GPT-2 small and seed-42 random SAE parameters. The following are experimental choices, not bundle facts:

- GPT-2 checkpoint revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- `HookedTransformer.from_pretrained_no_processing`: no weight folding or centering.
- CPU execution in evaluation/inference mode; FP32 model weights cast to each forward dtype.
- Prompt: “The quick brown fox jumps over the lazy dog.” with BOS, producing 11 tokens.
- Seed 42; `d_mlp = 3072`; 4096 SAE features.
- Standard-normal `W_enc[3072, 4096]` drawn first, then `b_dec[3072]`, using one CPU generator seeded at 42.
- Parameters cast to the tested forward dtype before the feature calculation.
- Feature projections use the prescribed linear expression, without an added encoder bias, ReLU, or other SAE nonlinearity.
- Python 3.12.14, PyTorch 2.14.1+cpu, and Transformers 5.19.0 for the numerical runs.

## Raw cache versus live-hook findings

The live observer and cache hook execute during the same forward pass. The observer clones the activation into independent storage and computes the feature projection inside the live hook. Exactly one live-hook call was observed per forward. All compared tensors were finite.

| Forward dtype | Comparison | Shape | Max absolute error | Mean absolute error | Unequal elements |
|---|---|---|---:|---:|---:|
| FP32 | Raw cached versus live activation | [1, 11, 3072] | 0.0 | 0.0 | 0 / 33,792 |
| BF16 | Raw cached versus live activation | [1, 11, 3072] | 0.0 | 0.0 | 0 / 33,792 |
| FP32 | Cached versus live FP32 feature projection | [1, 11, 4096] | 0.0 | 0.0 | 0 / 45,056 |
| BF16 | Cached versus live FP32 feature projection | [1, 11, 4096] | 0.0 | 0.0 | 0 / 45,056 |

These are exact same-forward comparisons with zero tolerances. They do not assert equality between activations from separate FP32 and BF16 model forwards.

The pinned cache-saving path in `HookedRootModule.get_caching_hooks` uses `tensor.detach().to(device)`, optionally selects positions/removes a batch dimension, and stores the tensor. It requests no dtype conversion and performs no SAE arithmetic. `ActivationCache.__getitem__` returns the stored tensor directly.

There is no observed BF16-specific rounding introduced by this cache path. Two general caveats remain: a detached tensor can share storage when no copy is made, and modifying hooks can cause observers at different hook-order positions to see different values. Neither mechanism was implicated by this experiment; neither inherently requires BF16.

## Demonstrated cause of the diagnostic difference

The diagnostic compared the same activation and parameters using:

```python
# Promote first.
reference = (activation.float() - b_dec.float()) @ W_enc.float()

# Subtract first; if both operands are BF16, rounding happens before promotion.
late_cast = (activation - b_dec).float() @ W_enc.float()
```

| Operand dtype before promotion | Max absolute feature difference | Mean absolute feature difference | Unequal elements |
|---|---:|---:|---:|
| FP32 | 0.0 | 0.0 | 0 / 45,056 |
| BF16 | 0.4064788818359375 | 0.0772847831249237 | 45,056 / 45,056 |

With BF16 operands, the second expression materializes a rounded BF16 subtraction result. Converting that result to FP32 cannot recover the discarded information. The matrix multiplication then propagates the difference into the feature projections.

This demonstrates a BF16-only discrepancy without a raw cache mismatch. Its error magnitudes apply to the random parameters in this experiment, not to the original trained SAE or report. Inconsistent promotion order is therefore a supported hypothesis for #1108, not an established diagnosis. Different autocast contexts, parameter precision, or separate-forward execution remain untested alternatives for the original report.

## Recommended promotion order

Use the same expression on both cached and live paths, promoting all operands before any subtraction or matrix multiplication:

```python
with torch.autocast(device_type=activation.device.type, enabled=False):
    features = ((activation.float() - b_dec.float()) @ W_enc.float())
```

Disabling autocast makes the intended FP32 arithmetic explicit. Casting only after subtraction or only after the matrix multiplication is not equivalent. Both paths must use the same parameter values; promoting already-rounded BF16 parameters does not restore an original FP32 checkpoint.

The JSON field `normalized_feature_path_matches_float32` refers to an FP32 reference calculated from the same forward's live activation. It does not compare BF16 model output against a separate FP32 model output.

## Artifacts and validation

- `repros/sae_cache_parity.py`: runnable assumption-based check.
- `sae_cache_parity_results.json`: one measured record per dtype, including raw parity, normalized feature parity, and the non-gating subtraction-order diagnostic.
- `tests/integration/test_sae_cache_parity.py`: exact raw-cache and promoted-feature checks for both FP32 and BF16.

The integration test run reported:

```text
======================== 4 passed, 1 warning in 15.17s =========================
```

The warning concerned type-checking instrumentation in `SVDInterpreter.py`.

On branch commit `d20813a5f551c1cc36e7bdc4e4106f78c5eb5558`, a separate checkout using the repository's locked environment returned exit 0 for `make format`, `make check-format`, and `uv run mypy .`. Mypy reported no issues in 301 source files; its configuration excludes tests. Formatting left the working tree unchanged, but isort reported “Broken 1 paths” for `isort format .`, pycln emitted an invalid-escape SyntaxWarning, and Black skipped notebooks because its Jupyter support was unavailable. Those checks were not a rerun of numerical parity under the locked environment.

Cache internals were not changed. No PR was opened and no email was sent. The synthetic regression passes; release clearance and resolution of the original report remain unverified.
