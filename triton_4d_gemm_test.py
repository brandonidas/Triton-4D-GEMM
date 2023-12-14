import torch

import triton
import triton.language as tl

from triton_4d_gemm import batched_4d_matmul

device = torch.device('cuda:0')

def random_matrix_test():
   # Testing the function
   B, C, M, K, N = 2, 2, 2, 2, 2
   A = torch.randn(B, C, M, K).to(device)
   B = torch.randn(B, C, K, N).to(device)

   # Using the custom function
   C_custom = batched_4d_matmul(A, B)

   # Using torch.bmm directly for comparison
   C_bmm = torch.matmul(A,B)

   # Check if results are the same
   print("shapes, right?", C_custom.shape, C_bmm.shape)
   print("row vs row:") 
   print(C_custom)
   print(C_bmm)
   print("Are the results identical? absolute tolerance = 1e-2", torch.allclose(C_custom, C_bmm,atol=1e-2))
random_matrix_test()
def identity_matrix_test():
  # Create an identity matrix for each 2x2 matrix in the 4D tensor
  identity_4d = torch.eye(2).repeat(2, 2, 1, 1).float().to(device)
  # Generate a 2 x 2 x 2 x 2 tensor with increasing numbers
  tensor_4d = torch.arange(2*2*2*2).view(2, 2, 2, 2).float().to(device)
  C_custom = batched_4d_matmul(tensor_4d,identity_4d)
  C_bmm = torch.matmul(tensor_4d, identity_4d)
  assert C_custom.is_contiguous()
  print(C_custom)
  print(C_bmm)

identity_matrix_test()

def big_matrix_test():
  torch.manual_seed(0)
  import time
  a = torch.randn((12,13,170,320), device='cuda', dtype=torch.float16).contiguous()
  b = torch.randn((12,13,320,170), device='cuda', dtype=torch.float16).contiguous()

  
  torch_start = time.time()
  torch_output = torch.matmul(a, b)
  torch_end = time.time()
  torch_time = torch_end - torch_start
  
  triton_start = time.time()
  triton_output = batched_4d_matmul(a, b)
  triton_end = time.time()
  triton_time = triton_end - triton_start

  print("triton time: ", triton_time)
  print("torch  time: ", torch_time)
  print("big matrix test: all close?", torch.allclose(triton_output, torch_output, atol=1e-2))
big_matrix_test()