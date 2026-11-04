"""Scoped framework precision policy for independent numerical comparisons."""
from contextlib import contextmanager


@contextmanager
def ieee_float32(torch):
    precision = torch.get_float32_matmul_precision()
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        # The boolean flag and named mode share backend state. Restore the
        # named mode last so "medium" retains its BF16 policy on PyTorch 2.9.
        torch.set_float32_matmul_precision(precision)
