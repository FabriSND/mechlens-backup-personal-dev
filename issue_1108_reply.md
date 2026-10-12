Thanks for reporting this, @nkechi-of.

We investigated the BF16-only SAE feature mismatch at `blocks.8.mlp.hook_post`. The notification mentions an attached reproducer, but no downloadable attachment was available in the copy we received, so we have not reproduced your original setup.

Using code snapshot `33387b7def1ead3a8cba532a729da5599e875873`, we ran a controlled CPU check with stock GPT-2 small and seed-42 random SAE parameters. These are substitute inputs, not your original model or SAE. We captured the live activation and populated the cache during the same forward pass.

| Forward dtype | Raw cache versus live: max / mean absolute error | SAE features: max / mean absolute error |
|---|---|---|
| FP32 | 0.0 / 0.0 | 0.0 / 0.0 |
| BF16 | 0.0 / 0.0 | 0.0 / 0.0 |

The feature comparisons used FP32 arithmetic on both paths:

```python
with torch.autocast(device_type=activation.device.type, enabled=False):
    features = ((activation.float() - b_dec.float()) @ W_enc.float())
```

We also tested a different promotion order:

```python
features = (activation - b_dec).float() @ W_enc.float()
```

With BF16 activation and bias operands, this subtracts and rounds in BF16 before promotion. In our random-SAE diagnostic, it produced a maximum feature difference of `0.4064788818359375` and a mean difference of `0.0772847831249237` from the promote-first expression. The FP32 diagnostic remained exactly equal. Those magnitudes apply only to our synthetic parameters.

The cache-saving path we inspected detaches the activation, optionally moves it to a device or slices it, and stores it without requesting a dtype conversion. We found no raw-cache discrepancy in this experiment. Promotion order is therefore a demonstrated way to produce the reported FP32/BF16 pattern, but we cannot yet confirm that it explains your result. Autocast context, parameter precision, hook ordering or in-place mutation, and differences between separate forwards are also worth checking in the original setup.

We added a runnable check and four integration cases covering raw-cache and promoted-feature parity in FP32 and BF16. The integration run completed with `4 passed, 1 warning in 15.17s`; the warning concerned type-checking instrumentation in `SVDInterpreter.py`. The numerical environment was Python 3.12.14, PyTorch 2.14.1+cpu, and Transformers 5.19.0. No cache internals were changed.

Could you reattach the minimal reproducer and include the model/checkpoint revision, input tokens, SAE checkpoint or parameter-generation code, package versions, device, and autocast settings? It would also help to know whether the cached and live values come from the same forward, and to see the exact feature expression on each path.

For now, we recommend promoting all three operands before subtraction and matrix multiplication, with autocast disabled for that expression. This synthetic test is not sufficient to close the report or declare release clearance.
