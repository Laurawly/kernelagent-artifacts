import sys
import torch
import triton
import triton.language as tl

# Triton kernel implementing elementwise ReLU
@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # program id along the flattened array
    pid = tl.program_id(0)
    # compute offsets for this program
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask off out-of-bounds lanes
    mask = offs < n_elements
    # load inputs (out-of-bounds lanes get 0.0 but won't be stored)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # apply ReLU
    y = tl.maximum(x, 0.0)
    # store outputs
    tl.store(y_ptr + offs, y, mask=mask)

def kernel_function(x: torch.Tensor,
                    out: torch.Tensor = None,
                    inplace: bool = True,
                    block_size: int = 4096):
    """
    Runs the Triton-based ReLU on tensor x.
    Args:
      x: Input CUDA tensor (contiguous), dtype float16/bfloat16/float32.
      out: Optional output tensor. If None and inplace=True, operates in-place.
      inplace: If True and out is None, mutates x in-place. Otherwise, allocates new output.
      block_size: Tile size for Triton kernel (constexpr).
    Returns:
      Tensor with ReLU applied, same shape/dtype as x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        "Unsupported dtype; supported: float16, bfloat16, float32"
    n_elements = x.numel()

    # determine output buffer
    if out is None:
        if inplace:
            out = x
        else:
            out = torch.empty_like(x)
    else:
        assert out.is_cuda and out.is_contiguous(), "out must be a CUDA contiguous tensor"
        assert out.shape == x.shape and out.dtype == x.dtype, "out must match x in shape and dtype"

    # launch grid: one program per block_size elements
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    _relu_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )
    return out

# Model and input generators (for testing/integration)
class Model(torch.nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    # returns a CUDA tensor for testing
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]

def get_init_inputs():
    return []  # no extra initialization inputs

# Self-test to compare Triton kernel vs. PyTorch reference
def run_tests():
    inputs = get_inputs()
    x = inputs[0]
    out_triton = kernel_function(x, inplace=False)
    with torch.no_grad():
        out_ref = torch.relu(x)
    if not torch.allclose(out_triton, out_ref, rtol=1e-3, atol=1e-3):
        max_diff = (out_triton - out_ref).abs().max().item()
        print(f"Test FAILED: max diff = {max_diff}")
        sys.exit(1)
    print("PASS")
    sys.exit(0)

if __name__ == "__main__":
    run_tests()
