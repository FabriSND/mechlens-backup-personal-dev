"""Run an assumption-based same-forward SAE/cache check; this is not the #1108 reproducer.

Run with the repository environment: uv run python repros/sae_cache_parity.py
The default output is sae_cache_parity_results.json in the working directory.

Assumptions: stock GPT-2 small without weight processing, the prompt below plus
BOS, CPU by default, and 4096 standard-normal random SAE features. A CPU generator
seeded at 42 draws W_enc first and b_dec second; both are cast to the forward dtype.
The code baseline is commit 33387b7def1ead3a8cba532a729da5599e875873.

max_abs_error, mean_abs_error and allclose compare cached-path features with
features calculated inside the live hook. shape describes the raw activation.
normalized_feature_path_matches_float32 compares the normalized cached path with
an explicit FP32 reference from the SAME forward's live activation. It does not
compare a BF16 model forward against a separate FP32 model forward. Autocast is
disabled for every feature calculation. Equality tolerances are zero.

The subtract-before-cast diagnostic is non-gating. No trained SAE, original
reproducer, or locked release templates are supplied by this experiment.
"""

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers

from transformer_lens import HookedTransformer

SEED = 42
HOOK = "blocks.8.mlp.hook_post"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PROMPT = "The quick brown fox jumps over the lazy dog."


def errors(actual, expected):
    """Measure exact parity after promoting operands to FP32."""
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "allclose": torch.allclose(actual, expected, rtol=0, atol=0),
        "mismatched_elements": torch.count_nonzero(actual != expected).item(),
    }


def features(activation, b_dec, W_enc):
    """Promote before subtraction and matrix multiplication."""
    with torch.autocast(device_type=activation.device.type, enabled=False):
        return (activation.float() - b_dec.float()) @ W_enc.float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("sae_cache_parity_results.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model = HookedTransformer.from_pretrained_no_processing(
        "gpt2", device=args.device, dtype=torch.float32, revision=MODEL_REVISION
    )
    model.eval()
    tokens = model.to_tokens(PROMPT, prepend_bos=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    W_base = torch.randn(model.cfg.d_mlp, 4096, generator=generator)
    b_base = torch.randn(model.cfg.d_mlp, generator=generator)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(SEED)
        model.to(dtype)
        W_enc = W_base.to(device=args.device, dtype=dtype)
        b_dec = b_base.to(device=args.device, dtype=dtype)
        live = {}
        calls = []

        def capture(activation, hook):
            calls.append(hook.name)
            live["activation"] = activation.detach().clone()
            live["features"] = features(activation, b_dec, W_enc).detach().clone()

        with torch.inference_mode(), model.hooks(fwd_hooks=[(HOOK, capture)]):
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK], return_type=None)
            cached = cache[HOOK]
            normalized = features(cached, b_dec, W_enc)
            with torch.autocast(device_type=cached.device.type, enabled=False):
                reference = (live["activation"].float() - b_dec.float()) @ W_enc.float()
                late_cast = (cached - b_dec).float() @ W_enc.float()
            raw = errors(cached, live["activation"])
            finite = all(
                torch.isfinite(tensor).all().item()
                for tensor in (cached, live["activation"], normalized, live["features"], reference)
            )
            record = {
                "dtype": str(dtype).removeprefix("torch."),
                "device": str(cached.device),
                "shape": list(cached.shape),
                **errors(normalized, live["features"]),
                "raw_cache_matches_live": torch.equal(cached, live["activation"]),
                "normalized_feature_path_matches_float32": torch.equal(normalized, reference),
                "raw_cache_errors": raw,
                "subtract_before_cast_errors": errors(late_cast, reference),
                "feature_shape": list(normalized.shape),
                "hook_calls": len(calls),
                "independent_storage": cached.data_ptr() != live["activation"].data_ptr(),
                "finite": finite,
                "seed": SEED,
                "hook": HOOK,
                "prompt": PROMPT,
                "tokens": tokens.tolist(),
                "model_revision": MODEL_REVISION,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "assumption_based": True,
            }
            record["passed"] = all(
                (
                    record["allclose"],
                    record["raw_cache_matches_live"],
                    record["normalized_feature_path_matches_float32"],
                    record["hook_calls"] == 1,
                    record["independent_storage"],
                    record["finite"],
                    cached.dtype == live["activation"].dtype == dtype,
                )
            )
            records.append(record)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(json.dumps(records, indent=2, allow_nan=False))
    raise SystemExit(0 if all(record["passed"] for record in records) else 1)


if __name__ == "__main__":
    main()
