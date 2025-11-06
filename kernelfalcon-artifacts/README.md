# KernelAgent Minimal Examples

This folder contains small, curated examples that showcase the pipeline outputs.

Source repo: https://github.com/meta-pytorch/KernelAgent
Blog post: https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/

- L1-examples/matmul: original model + generated Triton kernel + a short test log.
- L2-examples/conv-bn-relu: original model + generated Triton kernel (kernelagent route).
- L3-examples/resnet-block: original input model + subgraphs JSON + composed kernel + verification log (fuser route).

For full run artifacts (subgraphs, per-subgraph kernels, composed programs, and logs), see sibling L1/L2/L3 folders.
