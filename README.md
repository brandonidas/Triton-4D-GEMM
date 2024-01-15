# Intro
This repository presents a novel approach to optimize 4D matrix multiplications, particularly focused on enhancing Input/Output (IO) efficiency in Large Language Models (LLMs) by leveraging OpenAI's Triton. We introduce a block-based programming model that surpasses industry-standard cuBLAS, especially for non-square matrices. Our evaluation on an Nvidia A100 GPU demonstrates Triton's superiority in handling irregular matrix shapes and sizes. 

The following are from the report submitted for the graduate course CPSC 538G at UBC, Fall Term 2023, instructed by Arpan Gujarati. The report is also included in this repository for further reading.

<img width="413" alt="square_matrices" src="https://github.com/brandonidas/Triton-4D-GEMM/assets/4231029/02369ae3-a3c8-49e9-b5a7-36603b495244">

<img width="428" alt="LLAMA2_batch_sizes" src="https://github.com/brandonidas/Triton-4D-GEMM/assets/4231029/a6ba2e23-c0ff-43f6-9490-a048da931fc4">


# Implementation
The heart of our implementation is the 4D matrix multiplication kernel in Triton. We extended a documented 2D matrix multiplier to the 4D case by calculating strides for navigating matrices and advancing block pointers.

We employ an autotuner to select the optimal configuration for different matrix dimensions, enhancing Triton's adaptability to varying input sizes. The launch grid divides work by arranging blocks in a 1 to 3-dimensional grid, dynamically adjusting block sizes and warps based on the autotuner's recommendations.

# Discussion
We conducted several experiments to evaluate Triton's performance, including perfectly square matrices, LLAMA-2 dimensions, and start sequence dimensions. Triton consistently outperformed Pytorch in handling irregularly shaped matrices, demonstrating its effectiveness in optimizing 4D matrix operations in LLMs.

However, we acknowledge that these assessments are based on runtime comparisons and should be interpreted with caution, as other factors can influence runtime. Future work could focus on more refined experimental setups to explore Triton's performance in more detail.

[BRANDON_TONG___RTDS_Research_Project.pdf](https://github.com/brandonidas/Triton-4D-GEMM/files/13942336/BRANDON_TONG___RTDS_Research_Project.pdf)
