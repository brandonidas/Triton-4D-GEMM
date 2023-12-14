import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_4d_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B, C, # batch and channel
        M, N, K,
        # naive navigation of batch and channel. slight wastage on last group but that's fine.
        batch_stride_a, channel_stride_a, # from A.stride(0),A.stride(1)
        batch_stride_b, channel_stride_b,
        batch_stride_c, channel_stride_c,

        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,

        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m, pid_n = init_m_n_pid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    
    a_outer_dim_offset = calculate_outer_dim_offset(batch_stride_a, channel_stride_a)
    b_outer_dim_offset = calculate_outer_dim_offset(batch_stride_b, channel_stride_b)
    c_outer_dim_offset = calculate_outer_dim_offset(batch_stride_c, channel_stride_c)

    # Pointers for the first blocks of A and B.
    # Advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am, offs_bn, offs_k = \
        calculate_block_offsets(M, N, 
                                BLOCK_SIZE_M, BLOCK_SIZE_N, 
                                BLOCK_SIZE_K, 
                                pid_m, pid_n)

    a_ptrs, b_ptrs = init_A_B_pointers(a_ptr, b_ptr, 
                                       stride_am, stride_ak, 
                                       stride_bk, stride_bn, 
                                       a_outer_dim_offset, 
                                       b_outer_dim_offset, 
                                       offs_am, offs_bn, offs_k)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    write_back(c, c_ptr, 
               M, N, 
               stride_cm, stride_cn, 
               BLOCK_SIZE_M, BLOCK_SIZE_N, 
               c_outer_dim_offset, 
               pid_m, pid_n)
@triton.jit
def calculate_block_offsets(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, pid_m, pid_n):
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    return offs_am,offs_bn,offs_k

@triton.jit
def init_m_n_pid(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m,pid_n

@triton.jit
def calculate_outer_dim_offset(batch_stride, channel_stride):
    channel_offset = tl.program_id(axis=1) * channel_stride
    batch_offset   = tl.program_id(axis=2) * batch_stride 
    outer_dim_offset = channel_offset + batch_offset
    return outer_dim_offset

@triton.jit
def init_A_B_pointers(a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn, a_outer_dim_offset, b_outer_dim_offset, offs_am, offs_bn, offs_k):
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) 
    a_ptrs += a_outer_dim_offset

    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs += b_outer_dim_offset
    return a_ptrs,b_ptrs
@triton.jit
def write_back(c, c_ptr, M, N, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N, c_outer_dim_offset, pid_m, pid_n):
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_ptrs += c_outer_dim_offset

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def batched_4d_matmul(a, b):
    assert shape_constraints_true(a, b) , "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    B, C, M, K = a.shape
    _, _ , K, N = b.shape
    # Allocates output.
    c = torch.empty((B,C,M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: \
        (triton.cdiv(M, META['BLOCK_SIZE_M']) 
        * triton.cdiv(N, META['BLOCK_SIZE_N']),N,B, )
    batched_4d_matmul_kernel[grid](
        a, b, c,  #
        B, C, M, N, K,  #
        
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #

        a.stride(2), a.stride(3),  #
        b.stride(2), b.stride(3),  #
        c.stride(2), c.stride(3),  #
    )
    return c

def shape_constraints_true(a, b):
    shape_constraints = a.shape[0] == b.shape[0] and \
                        a.shape[1] == b.shape[1] and \
                        a.shape[3] == b.shape[2]
                        
    return shape_constraints