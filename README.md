# KernelAgent Artifacts

Curated, minimal artifacts showcasing KernelAgent/Fuser outputs across L1/L2/L3.

Source repo: https://github.com/meta-pytorch/KernelAgent
Blog post: https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/


## Layout

- `L1/` — selected Level 1 problems (original + generated Triton kernel).

- `L2/` — selected Level 2 problems (original + generated Triton kernel).

- `L3/` — selected Level 3 problems:

  - Fuser route: original + fused.py + subgraphs.json + per-subgraph kernels + composed_kernel.py + verify logs.

  - KernelAgent route: original + final_kernel.py + test.py + result.json.

- `kernelfalcon-artifacts/` — tiny examples for quick browsing.


## Minimal Examples

- `L1-examples/matmul/` — original + generated Triton kernel + short PASS log.

- `L2-examples/conv-bn-relu/` — original + generated Triton kernel (KernelAgent route).

- `L3-examples/resnet-block/` — input_model + subgraphs.json + composed_kernel + verify log (Fuser route).


## Reproduction

Artifacts were produced using the KernelAgent repo:
https://github.com/meta-pytorch/KernelAgent

- Auto-router: `python -m Fuser.auto_agent --problem <abs/path/to/problem.py> --verify`

- Full pipeline: `python -m Fuser.pipeline --problem <abs/path> --extract-model gpt-5 --dispatch-model o4-mini --compose-model o4-mini --dispatch-jobs auto --verify`

- KernelAgent direct (Python): see `triton_kernel_agent` in the main repo.


## Provenance

Each example folder contains a `manifest.json` with route, files, and verification info.

We trimmed logs for readability; full run artifacts live under `.fuse/run_*` and agent sessions in the main workspace and can be reproduced with the commands above.


## Notes

- Kernels here are Triton implementations (no PyTorch compute helpers in wrappers).

- Verification gates success on execution-based checks with tolerances (default rtol=1e-3, atol=1e-3; caps for fp16/bf16 at 1e-2).
