# Stage one

5 epochs, 10 batches per epoch, median value:

CPU: 0.4464s
CUDA: 27.3880s

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)   Count   Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)            Operation           
 --------  ---------------  -------  --------  --------  --------  ----------  -----------  ------------------------------
     46.0      389,583,372  940,588     414.2     384.0       224      24,479        440.8  [CUDA memcpy Device-to-Host]  
     28.0      237,308,475  252,014     941.6     960.0       704     202,421        559.8  [CUDA memcpy Device-to-Device]
     26.1      220,868,986  470,300     469.6     288.0       191  27,703,465     40,726.8  [CUDA memcpy Host-to-Device]

### Solution

Here we found that some of the operations were not implemented in CUDA. 
This lead to many memcopies with just a single element, stalling the CUDA
pipeline heavily. Solutoin is to bring those operations onto the GPU as well.

### After fix times

CPU: 0.4464s
GPU: 0.5023s

# Stage two

### Before fix times

5 epochs, 50 batches per epoch, median values: 

CPU: 1.7937s
GPU: 0.5023s

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  -------  --------  --------  --------  --------  -----------  ------------------------------
    612.337  378,210     0.002     0.003     0.000     0.201        0.003  [CUDA memcpy Device-to-Device]
    222.979       36     6.194     0.000     0.000   197.568       33.011  [CUDA memcpy Host-to-Device] 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Med (ns)  Min (ns)   Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  --------  --------  --------  ----------  -----------  ----------------------
     85.1    1,102,538,229    378,246   2,914.9   2,778.0     2,481  21,411,034     35,366.4  cudaMemcpy            
      8.8      114,277,057      1,251  91,348.6   2,331.0     1,045  99,851,235  2,823,252.0  cudaMalloc            
      4.7       60,950,710      2,019  30,188.6   3,956.0       622   2,169,833    143,562.3  cudaDeviceSynchronize


### Solution 

Looking at the timeline we see that there happens a lot of device-to-device memcopies, then a few sparse kernel 
executions, then the memcopies again. It appears that the slicing operations in the shuffling for training absorb
lots of run-time before anything can happen. We will try a separate kernel that first aligns all data before copying
fractured memory.

### After fix times

CPU: 0.4464s
*CUDA*: 0.1635s

# Stage three

### Before fix times

5 epochs, 50 batches per epoch, median values: 

CPU: 1.7937s
GPU: 0.1635s

### Solution

We do see that a lot of time is not spent on reordering the data in the 
makeContiguousCopy CUDA kernel. The only function calling this one right now is the backward matmul pass. To bypass the kernel we implement a CUDA kernel that implicitely indexes based on a transposed flag.

To be consisten with the CPU we do the same with the CPU implementation of matmul.

### After fix times

*CPU*: 1.9847s
*CUDA*: 0.0704s

**Important**: This has slowed the CPU version down. We will optimize that later after the CUDA optimization.

### Additionally: Fixed out-of-bounds bug

Used compute sanitizer -> memcheck to find out-of-bounds error in one kernel.

# Stage four

5 epochs, 10 batches per epoch, median value:

*CPU*: 1.9847s
*CUDA*: 0.0704s

### Solution

We see that the memcopies once again dominate the overall picture. We will tackle them two-fold: 

1. Initialize tensors directly on the GPU, rather than on the CPU and then copying them to the GPU.
2. Use CUDA streams for async copies.
3. Other optimizations, e.g. pinned memory perhaps?

