# Stage one

5 epoch, 10 batches per epoch, median value:

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

# Stage two

