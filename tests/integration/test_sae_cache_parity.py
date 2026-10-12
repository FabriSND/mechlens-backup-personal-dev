"""Same-forward parity with stock GPT-2 and seeded random SAE parameters."""

import pytest
import torch

HOOK = "blocks.8.mlp.hook_post"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


@pytest.fixture(scope="module", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def same_forward(request):
    """Capture an independent live snapshot and cache from one CPU forward."""
    from transformer_lens import HookedTransformer

    dtype = request.param
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = HookedTransformer.from_pretrained_no_processing(
                "gpt2", device="cpu", dtype=torch.float32, revision=REVISION
            )
            model.eval()
            model.to(dtype)
            tokens = model.to_tokens(
                "The quick brown fox jumps over the lazy dog.", prepend_bos=True
            )
            generator = torch.Generator(device="cpu").manual_seed(42)
            W_enc = torch.randn(model.cfg.d_mlp, 4096, generator=generator).to(dtype)
            b_dec = torch.randn(model.cfg.d_mlp, generator=generator).to(dtype)
            live = {}
            calls = []

            def capture(activation, hook):
                calls.append(hook.name)
                live["activation"] = activation.detach().clone()
                with torch.autocast(device_type="cpu", enabled=False):
                    live["features"] = (
                        ((activation.float() - b_dec.float()) @ W_enc.float()).detach().clone()
                    )

            with torch.inference_mode(), model.hooks(fwd_hooks=[(HOOK, capture)]):
                _, cache = model.run_with_cache(tokens, names_filter=[HOOK], return_type=None)
                cached = cache[HOOK]
                with torch.autocast(device_type="cpu", enabled=False):
                    cached_features = (cached.float() - b_dec.float()) @ W_enc.float()
            yield {
                "dtype": dtype,
                "calls": calls,
                "cached": cached,
                "live": live["activation"],
                "cached_features": cached_features,
                "live_features": live["features"],
            }
    finally:
        torch.set_num_threads(previous_threads)


def assert_exact_parity(actual, expected):
    """Require finite, exactly equal values and print measured errors."""
    difference = (actual.float() - expected.float()).abs()
    print(
        f"shape={list(actual.shape)} max_abs_error={difference.max().item()} "
        f"mean_abs_error={difference.mean().item()} "
        f"mismatched_elements={torch.count_nonzero(actual != expected).item()}"
    )
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_raw_cache_matches_live(same_forward):
    assert same_forward["calls"] == [HOOK]
    cached, live = same_forward["cached"], same_forward["live"]
    assert cached.dtype == live.dtype == same_forward["dtype"]
    assert cached.shape == (1, 11, 3072)
    assert cached.data_ptr() != live.data_ptr()
    assert_exact_parity(cached, live)


def test_sae_features_match_with_fp32_promotion(same_forward):
    cached, live = same_forward["cached_features"], same_forward["live_features"]
    assert same_forward["calls"] == [HOOK]
    assert cached.dtype == live.dtype == torch.float32
    assert cached.shape == (1, 11, 4096)
    assert cached.data_ptr() != live.data_ptr()
    assert_exact_parity(cached, live)
