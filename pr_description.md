# Investigate SAE BF16 cache parity at blocks.8.mlp.hook_post

## Issue and scope

Related to TransformerLensOrg/TransformerLens#1108: the June 10, 2026 notification reports SAE feature differences between cached and live runs at `blocks.8.mlp.hook_post` under BF16, with FP32 unaffected.

The notification's referenced reproducer was unavailable. This change adds a controlled, assumption-based regression and investigation; it does not establish the original report's cause or close the issue.

The branch starts from pinned code commit `33387b7def1ead3a8cba532a729da5599e875873`.

## Diagnosis and changes

Same-forward cached and independently cloned live activations matched exactly in FP32 and BF16. Feature projections also matched exactly when both paths used:

```python
with torch.autocast(device_type=activation.device.type, enabled=False):
    features = ((activation.float() - b_dec.float()) @ W_enc.float())
```

A controlled diagnostic changed only the promotion order to `(activation - b_dec).float() @ W_enc.float()`. With BF16 operands, subtraction rounds before promotion and produces a feature difference; the FP32 diagnostic remains equal. This demonstrates a plausible mechanism for the report, not a confirmed diagnosis of the original setup.

The inspected cache-saving path uses `tensor.detach().to(device)` and optional slicing, without requesting a dtype cast or performing SAE arithmetic. No cache internals or model behavior are changed.

Added artifacts:

- `repros/sae_cache_parity.py`: same-forward parity check with per-dtype JSON output.
- `sae_cache_parity_results.json`: recorded measurements and runtime information.
- `tests/integration/test_sae_cache_parity.py`: raw-cache and promoted-feature assertions for FP32 and BF16.
- `sae_bf16_cache_investigation.md`: provenance, assumptions, evidence, and diagnostic limits.
- `docs/source/content/debugging_sae_cache_parity.md`: reproduction and promotion-order guidance.
- `issue_1108_reply.md`: unposted issue-reply draft.

## Fixed experimental inputs

The substitute inputs were authorized explicitly and are not bundle facts: stock GPT-2 small without weight folding or centering; checkpoint revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; CPU; seed 42; prompt “The quick brown fox jumps over the lazy dog.” plus BOS; 11 tokens; standard-normal random `W_enc[3072, 4096]` and `b_dec[3072]`, drawn in that order from one seed-42 CPU generator and cast to each forward dtype.

The prescribed feature expression is a linear projection without additional SAE bias or nonlinearity. Its FP32 reference uses the same forward's activation, not a separate FP32 model forward.

## Numerical and integration-test evidence

Numerical runtime: Python 3.12.14, PyTorch 2.14.1+cpu, Transformers 5.19.0. The live observer fired once per forward, used independent activation storage, and all compared tensors were finite. Equality tolerances were zero.

| Forward dtype | Comparison | Shape | Max absolute error | Mean absolute error | Unequal elements |
|---|---|---|---:|---:|---:|
| FP32 | Raw cache versus live | [1, 11, 3072] | 0.0 | 0.0 | 0 / 33,792 |
| BF16 | Raw cache versus live | [1, 11, 3072] | 0.0 | 0.0 | 0 / 33,792 |
| FP32 | Promoted SAE features | [1, 11, 4096] | 0.0 | 0.0 | 0 / 45,056 |
| BF16 | Promoted SAE features | [1, 11, 4096] | 0.0 | 0.0 | 0 / 45,056 |

The non-gating subtract-before-cast diagnostic measured:

| Dtype | Max absolute feature difference | Mean absolute feature difference | Unequal elements |
|---|---:|---:|---:|
| FP32 | 0.0 | 0.0 | 0 / 45,056 |
| BF16 | 0.4064788818359375 | 0.0772847831249237 | 45,056 / 45,056 |

Actual integration command:

```bash
HF_HUB_OFFLINE=1 uv run --no-project --python ../parity-env/bin/python python -m pytest tests/integration/test_sae_cache_parity.py -v -s --tb=short
```

This used the isolated numerical environment and the repository's pytest configuration. Result: exit 0.

```text
======================== 4 passed, 1 warning in 15.17s =========================
```

All four cases passed: `test_raw_cache_matches_live[fp32]`, `test_sae_features_match_with_fp32_promotion[fp32]`, `test_raw_cache_matches_live[bf16]`, and `test_sae_features_match_with_fp32_promotion[bf16]`.

The warning was an `InstrumentationWarning` in `SVDInterpreter.py:30`: `@typechecked` could not instrument `get_singular_vectors` with its existing wrapper.

## Formatting and type-check evidence

These commands ran sequentially in a separate checkout of branch commit `d20813a5f551c1cc36e7bdc4e4106f78c5eb5558`, using the repository's locked environment. They were run before the subsequent investigation, guide, and reply Markdown additions. They do not constitute numerical testing under the locked runtime or validation of every later branch commit.

Tool versions: pycln 2.5.0, isort 5.8.0, Black 23.12.1, mypy 1.17.0.

### make format — exit 0

```text
uv run pycln --all . --exclude "__init__.py"
Looks good! ✨ 🍰 ✨
556 files left unchanged.
<unknown>:1410: SyntaxWarning: invalid escape sequence '\s'
uv run isort format .
Skipped 35 files
Broken 1 paths
uv run black .
Skipping .ipynb files as Jupyter dependencies are not installed.
You can fix this by running ``pip install "black[jupyter]"``
All done! ✨ 🍰 ✨
590 files left unchanged.
```

### make check-format — exit 0

```text
uv run pycln --check --all . --exclude "__init__.py"
Looks good! ✨ 🍰 ✨
556 files would be left unchanged.
<unknown>:1410: SyntaxWarning: invalid escape sequence '\s'
uv run isort --check-only .
Skipped 35 files
uv run black --check .
Skipping .ipynb files as Jupyter dependencies are not installed.
You can fix this by running ``pip install "black[jupyter]"``
All done! ✨ 🍰 ✨
590 files would be left unchanged.
```

### uv run mypy . — exit 0

```text
Success: no issues found in 301 source files
```

Formatting left the working tree unchanged. The successful exits retain these limitations: isort reported a broken path for `isort format .`; Black skipped notebooks; and the mypy configuration excludes tests, so this result does not type-check the new integration test. The full test suite was not run.

### Documentation build — exit 1

After adding the debugging guide locally, `uv run build-docs` failed during model-table generation following Hugging Face token/metadata retrieval warnings. The terminal error was:

```text
KeyError: "['tokenizer.name', 'tokenizer.class', 'tokenizer.vocab_size', 'tokenizer.vocab_hash'] not in index"
```

A successful complete documentation build has not been established.

## Risk and limitations

Production behavior risk is low because the changes add a reproducer, tests, recorded results, and documentation without modifying the cache or model implementation.

The integration tests require the public GPT-2 checkpoint or an existing cache and add model-loading time and memory use. Measurements cover CPU, one prompt, one model revision, and synthetic SAE parameters; they do not establish CUDA parity or trained-SAE behavior. Dependency versions can affect results, and the stored JSON is a record of the measured environment rather than a universal expected output.

The main interpretation risk is treating the demonstrated promotion-order mechanism as proof of the original cause. The original model, inputs, SAE checkpoint, runtime, and reproducer remain unavailable. This change provides no release clearance.

Storage aliasing from non-cloning cache capture and modifying-hook ordering remain general considerations. Neither was implicated by this experiment.

## Rollback

Revert the additive commits on `fix/sae-bf16-cache-parity`, or remove the reproducer, recorded JSON, integration test, investigation note, debugging guide, and draft Markdown files. No cache-internal rollback, model-weight change, schema migration, or production configuration change is required.

No PR has been opened, the issue reply has not been posted, and no email has been sent.
