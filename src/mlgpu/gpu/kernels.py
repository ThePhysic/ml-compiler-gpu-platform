"""Original kernels for the bounded mlgpu IR.

Only imported after explicit backend selection. No framework arithmetic is used
here. Fused kernels retain the IR's observable dtype barriers.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


@triton.jit
def elementwise(Ptrs, Out, N: tl.constexpr, SHAPE: tl.constexpr,
                OUT_STRIDES: tl.constexpr, IN_SHAPES: tl.constexpr,
                IN_STRIDES: tl.constexpr, OPS: tl.constexpr,
                ARG0: tl.constexpr, ARG1: tl.constexpr, BLOCK: tl.constexpr):
    """Compile a bounded straight-line SSA region to one memory pass.

    External tensors can broadcast; intermediates have the final shape.
    Integer operand indices bind the validated SSA graph to a register tuple.
    """
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = ()
    for p in tl.static_range(len(IN_SHAPES)):
        offset = tl.full((BLOCK,), 0, tl.int32)
        for axis in tl.static_range(len(SHAPE)):
            if IN_SHAPES[p][axis] != 1:
                coordinate = (index // OUT_STRIDES[axis]) % SHAPE[axis]
                offset += coordinate * IN_STRIDES[p][axis]
        value = tl.load(Ptrs[p] + offset, index < N, other=0).to(tl.float32)
        values += (value,)
    for op in tl.static_range(len(OPS)):
        a = values[ARG0[op]]
        if OPS[op] == 0:
            value = a + values[ARG1[op]]
        elif OPS[op] == 1:
            value = a * values[ARG1[op]]
        elif OPS[op] == 2:
            value = tl.maximum(a, 0.0, propagate_nan=tl.PropagateNan.ALL)
        elif OPS[op] == 3:
            value = _gelu(a)
        else:
            value = a
        value = value.to(Out.dtype.element_ty).to(tl.float32)
        values += (value,)
    tl.store(Out + index, values[-1], index < N)


@triton.jit
def copy_flat(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + idx, tl.load(X + idx, idx < N, other=0), idx < N)


@triton.jit
def transpose_copy(X, Y, N: tl.constexpr, SHAPE: tl.constexpr,
                   OUT_STRIDES: tl.constexpr, SRC_STRIDES: tl.constexpr,
                   BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = tl.full((BLOCK,), 0, tl.int32)
    for axis in tl.static_range(len(SHAPE)):
        source += ((idx // OUT_STRIDES[axis]) % SHAPE[axis]) * SRC_STRIDES[axis]
    tl.store(Y + idx, tl.load(X + source, idx < N, other=0), idx < N)


@triton.jit
def softmax_rows(X, Y, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=-float("inf")).to(tl.float32)
    shifted = x - tl.max(x, axis=0)
    numerator = tl.exp(shifted)
    probability = numerator / tl.sum(numerator, axis=0)
    tl.store(Y + row * WIDTH + col, probability, col < WIDTH)


@triton.jit
def layernorm_rows(X, W, B, Y, WIDTH: tl.constexpr, EPS: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(col < WIDTH, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    normalized = centered * tl.rsqrt(variance + EPS)
    weight = tl.load(W + col, col < WIDTH, other=0).to(tl.float32)
    bias = tl.load(B + col, col < WIDTH, other=0).to(tl.float32)
    value = normalized * weight + bias
    tl.store(Y + row * WIDTH + col, value, col < WIDTH)


@triton.jit
def sum_axis(X, Y, WIDTH: tl.constexpr, INNER: tl.constexpr,
             BLOCK: tl.constexpr):
    out = tl.program_id(0)
    axis = tl.arange(0, BLOCK)
    source = (out // INNER) * WIDTH * INNER + axis * INNER + out % INNER
    value = tl.load(X + source, axis < WIDTH, other=0).to(tl.float32)
    tl.store(Y + out, tl.sum(value, axis=0))


@triton.jit
def matmul_epilogue(A, B, Bias, Y, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, HAS_BIAS: tl.constexpr,
                    ACTIVATION: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    """Rectangular row-major tiles with all three dimension tails masked."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + rk
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(B + k[:, None] * N + cols[None, :],
                    (k[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    # Preserve materialized matmul/add rounding in a register epilogue.
    value = acc.to(Y.dtype.element_ty).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + cols, cols < N, other=0).to(tl.float32)
        value = (value + bias[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    if ACTIVATION == 1:
        value = tl.maximum(value, 0.0, propagate_nan=tl.PropagateNan.ALL)
    elif ACTIVATION == 2:
        value = _gelu(value)
    tl.store(Y + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))
