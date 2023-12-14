import torch

import triton
import triton.language as tl

from triton_4d_gemm import batched_4d_matmul

device = torch.device('cuda:0')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[i for i in range(1, 32)],  # Different possible values for `x_name`
        # TODO tweak above

        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    K = 128 # standard LLAMA 2 embedding size
    B, C = 6, 32 # standard from LLAMA 2

    # print("in bench")
    a = torch.randn((B,C,M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((B,C,K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    warmup = 25
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), 
                                                     quantiles=quantiles,
                                                     warmup = warmup)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: batched_4d_matmul(a, b), 
                                                     quantiles=quantiles,
                                                     warmup = warmup)
    perf = lambda ms: 2 * B * C *M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)