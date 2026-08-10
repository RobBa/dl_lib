# Step 1

### Times before

5 epochs, 500 batches per epoch.

CPU: 17.9229s
GPU: 0.2058s

### Perf output

Perf tells us 

+   40.06%    40.06%  python3  [.] Tensor::matMulImpl(Tensor const&, Tensor const&, bool, bool)
+    6.97%     0.00%  python3  [.] 0x8348535554415541
+    6.97%     0.00%  python3  [.] 0x00007a5e1326ebe8
+    6.97%     0.00%  python3  [.] 0x0000000038d1b717
+    3.48%     2.97%  python3  [.] Tensor::tensorValues_t::operator[](unsigned int) const   


We see two multiple main loops:

388:   lea        (%r8,%rcx,1),%esi                                                                                                                                           ▒
   0.05 │       mov        %ecx,%eax
        │       add        $0x1,%ecx
  13.33 │       movss      (%r11,%rsi,4),%xmm0
        │       mulss      0x0(%r13,%rax,4),%xmm0
   0.12 │       addss      %xmm0,%xmm1
        │       cmp        %ecx,%ebp
  13.97 │     ↑ jne        388

---------------------------------------

        │490:   mov        %edi,%eax
        │       mov        %ecx,%r8d 
        │       add        $0x1,%ecx
  14.06 │       movss      (%r14,%r8,4),%xmm0
   1.01 │       mulss      (%rbx,%rax,4),%xmm0
        │       mov        0x8(%rsp),%eax
        │       add        %eax,%edi
  13.83 │       addss      %xmm0,%xmm1
        │       cmp        %ebp,%ecx
  12.56 │     ↑ jne        490

--------------------------------------- 

570:   mov        %edi,%r11d
        │       mov        %r8d,%r10d
        │       add        $0x1,%esi
  10.12 │       add        %eax,%edi
   0.07 │       movss      (%r14,%r11,4),%xmm0
   0.15 │       mulss      0x0(%r13,%r10,4),%xmm0
        │       add        %edx,%r8d
   8.92 │       addss      %xmm0,%xmm1
   0.01 │       cmp        %ecx,%esi
  10.76 │     ↑ jne        570

### Analysis

ss -> no SIMD instructions -> no loop unrolling happening (would have suffix ps)

### Try fix

Let cmake know it can optimize, and that it can run the native hardware as well via 

```
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
```

My CPU is an Intel i5 210Hx12. Checking for AVX via 

```
cat /proc/cpuinfo | grep flags | head -1 | tr ' ' '\n' | grep -i avx
```

We find 

- avx
- avx2
- avx_vnni

So we should be able to let it compile for AVX. Interestingly, the --march=native made things slower, 
to from ~18s per 500 batches to ~22s. We find that it finds -march=native -> alderlake, we have raptorlake.
Setting flag by hand we get the same result. Inspect: 

Inspection does not reveal directly what it is, but we do want to use AVX anyway. We omit the --march and only use -O3. 
Speedup: 

Minimal. No AVX used.

### Times after

- ~17.07s
- Minimal improvement...

# Step 2: Optimize matmul for non transposed case

### Analysis and profiling

We find that the matmul is the lowest hanging fruit still. We want to make it AVX friendly, therefore we modify the memory access pattern. We want to go transposition version dependent, therefore we start with the first version: no transpose happening in both cases. Write benchmarks, profile with 

```
perf record -g -e cycles:pp,cache-misses:pp ./bm_matmul --benchmark_filter='BM_MatMul_CPU/'
```

, or more granular with 

```
perf record -g -e cpu_core/cycles/pp ./bm_matmul --benchmark_filter='BM_MatMul_CPU/'
perf record -g -e cpu_core/cache-misses/pp ./bm_matmul --benchmark_filter='BM_MatMul_CPU/'
```
and we annotate with

```
sudo perf annotate --stdio -l --source
```

to get the screenshots step_2_*.png

Looking at the cache misses we do see the two instructions 

```
6.83 :   6dd30:  movss  0x0(%r13,%r10,4),%xmm1 // tensor.h:341
75.98 :   6dd37:  mulss  (%r9,%rax,4),%xmm1
```

Indexing in x86: displacement(base_addr, index, scale) results in address resolution

displacement + base_addr + index * scale. I.e. we have memfetches baked into the mulss, which get multiplied by value in xmm1, then store back in xmm1.
xmm1 = floating-point register.

Also, we see 

```
6dd3d:  mov    0x8(%rsp),%eax
```

, likely a register spill.\

For better comparison, we use 

```
taskset -c 0 sudo perf stat -e cpu_core/cycles/,cpu_core/instructions/,cpu_core/cache-misses/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/ -- ./bm_matmul --benchmark_filter="BM_MatMul_CPU/"
```

to get an overview. 

```
---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64              142 us          142 us         4921 GFLOP/s=3.68479/s
BM_MatMul_CPU/256/256/256         10459 us        10448 us           67 GFLOP/s=3.21153/s
BM_MatMul_CPU/512/512/512        122607 us       121918 us            6 GFLOP/s=2.20177/s
BM_MatMul_CPU/1024/1024/1024    3441785 us      3436093 us            1 GFLOP/s=0.624978/s
BM_MatMul_CPU/64/784/256          13191 us        13173 us           53 GFLOP/s=1.95014/s
BM_MatMul_CPU/64/256/128           1282 us         1281 us          547 GFLOP/s=3.27549/s
BM_MatMul_CPU/64/128/10            45.2 us         45.1 us        15522 GFLOP/s=3.62976/s

 Performance counter stats for './bm_matmul --benchmark_filter=BM_MatMul_CPU/':

    34,739,206,009      cpu_core/cycles/                                                      
    95,697,238,606      cpu_core/instructions/           #    2.75  insn per cycle            
        60,146,208      cpu_core/cache-misses/                                                
    28,555,334,711      cpu_core/L1-dcache-loads/                                             
     5,320,403,187      cpu_core/L1-dcache-load-misses/  #   18.63% of all L1-dcache accesses 
     1,159,881,849      cpu_core/LLC-loads/                                                   
        20,760,502      cpu_core/LLC-load-misses/        #    1.79% of all LL-cache accesses  

       9.288843453 seconds time elapsed

       8.921408000 seconds user
       0.341977000 seconds sys
```

## Step 2-1: More cache friendly version

```
  if constexpr (!transposeLeft && !transposeRight) {
    for (tensorSize_t i = 0; i < nRowsLeft; i++) {
      const tensorSize_t leftOffset = i * nColsLeft;
      const tensorSize_t resOffset = i * nColsRight;

      {
        // tensors not zero initialized, therefore need one dry run pre-filling
        const auto leftVal = left[leftOffset];
        for(tensorSize_t j = 0; j < nColsRight; j++) {
          res.values->data()[resOffset + j] = leftVal * right[j];
        }
      }

      // compute the entire column of the left matrix
      for (tensorSize_t k = 1; k < nRowsRight; k++) {
        const tensorSize_t rightOffset = k * nColsRight;

        const ftype leftVal = left[leftOffset + k];
        for(tensorSize_t j = 0; j < nColsRight; j++) {
          res.values->data()[resOffset + j] += leftVal * right[rightOffset + j];
        }
      }
    }
  }
  else {
    const tensorSize_t M = transposeLeft ? nColsLeft : nRowsLeft;
    const tensorSize_t K = transposeLeft ? nRowsLeft : nColsLeft;
    const tensorSize_t N = transposeRight ? nRowsRight : nColsRight;

    for (tensorSize_t i = 0; i < M; i++) {
      for (tensorSize_t j = 0; j < N; j++) {
        ftype sum = 0;

        for (tensorSize_t k = 0; k < K; k++) {
          tensorSize_t leftIdx = transposeLeft ? leftOffset + k * nColsLeft + i
                                              : leftOffset + i * nColsLeft + k;
          
          tensorSize_t rightIdx = transposeRight ? rightOffset + j * nColsRight + k
                                                : rightOffset + k * nColsRight + j;

          sum += left.values->data()[leftIdx] * right.values->data()[rightIdx];
        }

        res.values->data()[resOffset + i * N + j] = sum;
      }
    }
  }
```

### Times after step 2-1: 

Before: ~17.0s
After: 29.3046s
Slower...

### Analysis

```
---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64              521 us          521 us         1346 GFLOP/s=1.00686/s
BM_MatMul_CPU/256/256/256         30242 us        30143 us           23 GFLOP/s=1.11317/s
BM_MatMul_CPU/512/512/512        240924 us       239886 us            3 GFLOP/s=1.11901/s
BM_MatMul_CPU/1024/1024/1024    1899346 us      1896614 us            1 GFLOP/s=1.13227/s
BM_MatMul_CPU/64/784/256          23110 us        23081 us           30 GFLOP/s=1.11306/s
BM_MatMul_CPU/64/256/128           4008 us         4003 us          175 GFLOP/s=1.04782/s
BM_MatMul_CPU/64/128/10             237 us          236 us         2960 GFLOP/s=0.693071/s

 Performance counter stats for './bm_matmul --benchmark_filter=BM_MatMul_CPU/':

    30,493,695,954      cpu_core/cycles/                                                      
   172,518,480,863      cpu_core/instructions/           #    5.66  insn per cycle            
        11,217,468      cpu_core/cache-misses/                                                
    69,372,480,963      cpu_core/L1-dcache-loads/                                             
        27,245,452      cpu_core/L1-dcache-load-misses/  #    0.04% of all L1-dcache accesses 
         1,986,866      cpu_core/LLC-loads/                                                   
           676,631      cpu_core/LLC-load-misses/        #   34.06% of all LL-cache accesses  

       8.144915641 seconds time elapsed

       7.771304000 seconds user
       0.343924000 seconds sys

```

```
    0.04 :   6dde2:  mov    %r14,%rdi
   22.88 :   6dde5:  call   6d230 <Tensor::operator[](unsigned int) const> // tensor.h:320
         : 593   ftype* data() noexcept { return values; }
    0.02 :   6ddea:  mov    0x8(%rsp),%rax
         : 595   res.values->data()[resOffset + j] += leftVal * right[rightOffset + j];
    0.00 :   6ddef:  lea    (%rbx,%r15,1),%edx
    0.01 :   6ddf3:  mulss  0x10(%rsp),%xmm0
         : 598   for(tensorSize_t j = 0; j < nColsRight; j++) {
   24.15 :   6ddf9:  add    $0x1,%ebx // tensor.h:319
         : 600   ftype* data() noexcept { return values; }
    0.00 :   6ddfc:  mov    0x68(%rax),%rax
         : 602   res.values->data()[resOffset + j] += leftVal * right[rightOffset + j];
    0.07 :   6de00:  mov    0x8(%rax),%rax
    0.05 :   6de04:  lea    (%rax,%rdx,4),%rax
   27.04 :   6de08:  addss  (%rax),%xmm0 // tensor.h:320
    0.04 :   6de0c:  movss  %xmm0,(%rax)
         : 607   for(tensorSize_t j = 0; j < nColsRight; j++) {
    0.00 :   6de10:  mov    0x1c(%rsp),%eax
    0.00 :   6de14:  mov    %eax,%r12d
    0.00 :   6de17:  cmp    %eax,%ebx
   24.46 :   6de19:  jne    6dde0 <Tensor::matMulImpl(Tensor const&, Tensor const&, bool, bool)+0x5d0> // tensor.h:319
```

Likely culprit: []-operator not inlined.

## Step 2-2: Make [] operator inlinable, use the array directly in CPU kernel

Replace instructions like ```right[rightOffset + j];``` with accesses to the array directly.

### Times after step 2-2

Before: 29.3046s
After: 15.1904s

## Step 2-3: More inlining of class methods and other small functions

### Analysis

---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64              159 us          158 us         4433 GFLOP/s=3.30962/s
BM_MatMul_CPU/256/256/256          8533 us         8523 us           82 GFLOP/s=3.93702/s
BM_MatMul_CPU/512/512/512         69071 us        68942 us           10 GFLOP/s=3.89366/s
BM_MatMul_CPU/1024/1024/1024     526542 us       525977 us            1 GFLOP/s=4.08285/s
BM_MatMul_CPU/64/784/256           6553 us         6545 us          107 GFLOP/s=3.92498/s
BM_MatMul_CPU/64/256/128           1128 us         1126 us          621 GFLOP/s=3.72468/s
BM_MatMul_CPU/64/128/10            42.1 us         42.0 us        16639 GFLOP/s=3.9011/s

 Performance counter stats for './bm_matmul --benchmark_filter=BM_MatMul_CPU/':

    22,987,630,818      cpu_core/cycles/                                                      
   113,088,940,741      cpu_core/instructions/           #    4.92  insn per cycle            
        11,798,362      cpu_core/cache-misses/                                                
    22,697,746,817      cpu_core/L1-dcache-loads/                                             
        32,255,795      cpu_core/L1-dcache-load-misses/  #    0.14% of all L1-dcache accesses 
         1,641,856      cpu_core/LLC-loads/                                                   
           697,471      cpu_core/LLC-load-misses/        #   42.48% of all LL-cache accesses

```
   15.35 :   6ca46:  sub    $0x18,%rsp // tensor.cpp:212
    0.00 :   6ca4a:  mov    %fs:0x28,%rax
    0.00 :   6ca53:  mov    %rax,0x8(%rsp)
    0.00 :   6ca58:  xor    %eax,%eax
         : 19    if(idx >= size)
   16.86 :   6ca5a:  cmp    (%rdi),%esi // tensor.cpp:213
    0.00 :   6ca5c:  jae    1175a <Tensor::tensorValues_t::operator[](unsigned int) const [clone .cold]>
         : 22    throw std::out_of_range("Out of range for tensor");
         :
         : 24    switch(device){
    0.00 :   6ca62:  mov    0x10(%rdi),%eax
    0.00 :   6ca65:  test   %eax,%eax
   17.87 :   6ca67:  je     6cac8 <Tensor::tensorValues_t::operator[](unsigned int) const+0x88> // tensor.cpp:216
```

We still see some functions that are being called. To reduce this noise from future profiling we go through the main classes tensor and dimension and inline what can be inlined. Changes made here can be found in git commit with tag b320be5b15f0af3791c3e7e89557e1e03e039e30

### Times after step 2-3

Before: 15.1904s
After: 14.5321s

## Step 2-4: Tiled matmul implementation of forward non-transposed case

### Analysis

```
---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64              146 us          146 us         4679 GFLOP/s=3.59966/s
BM_MatMul_CPU/256/256/256          8786 us         8777 us           80 GFLOP/s=3.8228/s
BM_MatMul_CPU/512/512/512         67861 us        67745 us           10 GFLOP/s=3.96243/s
BM_MatMul_CPU/1024/1024/1024     534708 us       533912 us            1 GFLOP/s=4.02216/s
BM_MatMul_CPU/64/784/256           6916 us         6904 us          102 GFLOP/s=3.72125/s
BM_MatMul_CPU/64/256/128           1202 us         1199 us          583 GFLOP/s=3.49701/s
BM_MatMul_CPU/64/128/10            43.8 us         43.8 us        16094 GFLOP/s=3.74131/s

 Performance counter stats for './bm_matmul --benchmark_filter=BM_MatMul_CPU/':

    23,092,239,160      cpu_core/cycles/                                                      
   111,472,734,176      cpu_core/instructions/           #    4.83  insn per cycle            
        13,992,253      cpu_core/cache-misses/                                                
    22,104,773,261      cpu_core/L1-dcache-loads/                                             
        31,511,720      cpu_core/L1-dcache-load-misses/  #    0.14% of all L1-dcache accesses 
         1,835,219      cpu_core/LLC-loads/                                                   
           752,361      cpu_core/LLC-load-misses/        #   41.00% of all LL-cache accesses  

       6.233058179 seconds time elapsed

       5.805175000 seconds user
       0.370691000 seconds sys
```

```
   30.47 :   6f722:  movss  (%r14,%r13,4),%xmm0 // tensor.h:392
    0.06 :   6f728:  lea    (%r12,%r9,4),%r9
    1.39 :   6f72c:  mulss  %xmm1,%xmm0
    0.34 :   6f730:  addss  (%r9),%xmm0
   32.13 :   6f735:  movss  %xmm0,(%r9)
         : 461   for(tensorSize_t j = 0; j < nColsRight; j++) {
    0.03 :   6f73a:  cmp    %edx,%eax
   31.94 :   6f73c:  jne    6f718 <Tensor::matMulImpl(Tensor const&, Tensor const&, bool, bool)+0x448> // tensor.h:391
```

Problem: Read and write on each loop. To fix this we could swap the loops to write the result into an accumulator, keeping 
the result in a register locally, but that would break the memory access pattern on the right side of the loop. We have 
two choices now: 

1. Use AVX to coalesce memory accesses. Better read/write behavior, less operations. 
2. Use tiling techniques - same idea, allows us to rewrite loop to increase L3- (LLC-) cache hit rate. 

We first use tiling to get memory patterns, and we can use AVX later on top of it for maximum performance squeeze.

Note: We also use the chance to do away some of the safety features when we spot them. E.g. 

```
Tensor Tensor::operator+(const Tensor& other) const {
#ifndef NDEBUG
  if(this->dims != other.dims &&
    !(other.dims.nDims() == 1 && other.dims.get(0) == dims.get(-1))){
    __throw_invalid_argument("Tensors need matching dimensions");
  }
  else if(values->getDevice()!=other.values->getDevice()){
    __throw_runtime_error("Tensors on different devices.");
  }
#endif

// .....
}
```

We can find bugs quickly, but remove small instruction overhead. Possibly negligible, but still good to have.

### Times after fix

Before: 14.5321s
After: 13.9477s

# Step 2-5: Try AVX

### Description

We use AVX to get the matmul correct.

### Times after fix:

Before: 11.2718s
After: 13.8033s

Interestingly, CPU version got faster. Maybe temperature in room? 
Do an analysis to compare two versions: 

### Analysis of result

Auto vectorized loops for scalar version now:

```
         : 448   const ftype leftVal = tiles.left[leftTileRowOffset + m];
    0.00 :   6ea32:  movss  -0x4(%rcx),%xmm1
    0.00 :   6ea37:  shufps $0x0,%xmm1,%xmm1
         : 451   tiles.result[leftTileRowOffset + kk] += leftVal * tiles.right[rightTileRowOffset + kk];
    0.03 :   6ea3b:  mulps  -0xb0(%rdx),%xmm1
    0.00 :   6ea42:  addps  %xmm2,%xmm1
    0.03 :   6ea45:  movaps %xmm1,0x8050(%rax)
    0.00 :   6ea4c:  movaps -0x1a0(%rdx),%xmm2
    0.18 :   6ea53:  mulps  %xmm0,%xmm2
    0.00 :   6ea56:  addps  0x8060(%rax),%xmm2
    0.00 :   6ea5d:  movaps %xmm2,0x8060(%rax)
```

Interestingly, the benchmarks prefer the AVX version. 

**Scalar matmul**: 

```
---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64             33.3 us         33.2 us        21099 GFLOP/s=15.7689/s
BM_MatMul_CPU/256/256/256          1863 us         1863 us          378 GFLOP/s=18.0083/s
BM_MatMul_CPU/512/512/512         14652 us        14650 us           47 GFLOP/s=18.3236/s
BM_MatMul_CPU/1024/1024/1024     118488 us       118473 us            6 GFLOP/s=18.1264/s
BM_MatMul_CPU/64/784/256           1486 us         1485 us          471 GFLOP/s=17.2944/s
BM_MatMul_CPU/64/256/128            236 us          235 us         2977 GFLOP/s=17.8199/s
BM_MatMul_CPU/64/128/10            55.7 us         55.7 us        12536 GFLOP/s=2.94307/s
```

```
     6,119,158,859      task-clock                       #    0.983 CPUs utilized             
               203      context-switches                 #   33.174 /sec                      
                41      cpu-migrations                   #    6.700 /sec                      
           327,558      page-faults                      #   53.530 K/sec                     
    25,713,289,941      cpu_atom/instructions/           #    1.65  insn per cycle              (1.17%)
    61,227,574,418      cpu_core/instructions/           #    2.75  insn per cycle              (34.70%)
    15,549,578,061      cpu_atom/cycles/                 #    2.541 GHz                         (1.20%)
    22,244,783,782      cpu_core/cycles/                 #    3.635 GHz                         (41.66%)
     5,075,506,806      cpu_atom/branches/               #  829.445 M/sec                       (1.25%)
     9,336,318,833      cpu_core/branches/               #    1.526 G/sec                       (48.61%)
        69,486,119      cpu_atom/branch-misses/          #    1.37% of all branches             (1.29%)
        84,171,032      cpu_core/branch-misses/          #    0.90% of all branches             (55.56%)
 #     16.0 %  tma_backend_bound      
                                                  #     20.7 %  tma_bad_speculation    
                                                  #     37.1 %  tma_frontend_bound     
                                                  #     26.2 %  tma_retiring             (62.51%)
 #     16.3 %  tma_bad_speculation    
                                                  #     33.9 %  tma_retiring             (1.33%)
 #     25.7 %  tma_backend_bound      
                                                  #     24.1 %  tma_frontend_bound       (1.36%)
    12,923,830,681      cpu_atom/L1-dcache-loads/        #    2.112 G/sec                       (1.14%)
    16,573,597,586      cpu_core/L1-dcache-loads/        #    2.708 G/sec                       (69.47%)
     1,455,566,977      cpu_core/L1-dcache-load-misses/  #    8.78% of all L1-dcache accesses   (69.50%)
     3,426,553,400      cpu_atom/LLC-loads/              #  559.971 M/sec                       (1.10%)
        98,259,794      cpu_core/LLC-loads/              #   16.058 M/sec                       (69.51%)
     9,613,462,195      cpu_atom/LLC-load-misses/        #  280.56% of all LL-cache accesses    (1.07%)
        87,267,152      cpu_core/LLC-load-misses/        #   88.81% of all LL-cache accesses    (69.50%)
    15,603,566,118      cpu_atom/L1-icache-loads/        #    2.550 G/sec                       (1.03%)
       138,422,928      cpu_atom/L1-icache-load-misses/  #    0.89% of all L1-icache accesses   (1.01%)
       173,264,212      cpu_core/L1-icache-load-misses/                                         (27.77%)
    20,279,132,840      cpu_atom/dTLB-loads/             #    3.314 G/sec                       (0.99%)
    16,516,515,061      cpu_core/dTLB-loads/             #    2.699 G/sec                       (27.75%)
     2,684,543,596      cpu_atom/dTLB-load-misses/       #   13.24% of all dTLB cache accesses  (0.97%)
         8,554,806      cpu_core/dTLB-load-misses/       #    0.05% of all dTLB cache accesses  (27.74%)
     1,409,394,265      cpu_atom/iTLB-load-misses/                                              (0.97%)
           295,925      cpu_core/iTLB-load-misses/                                              (27.74%)

```

**AVX matmul**

```
---------------------------------------------------------------------------------------
Benchmark                             Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64             21.4 us         21.4 us        32326 GFLOP/s=24.4923/s
BM_MatMul_CPU/256/256/256          1127 us         1127 us          622 GFLOP/s=29.7775/s
BM_MatMul_CPU/512/512/512          8615 us         8615 us           81 GFLOP/s=31.1598/s
BM_MatMul_CPU/1024/1024/1024      68145 us        68139 us           10 GFLOP/s=31.5161/s
BM_MatMul_CPU/64/784/256            890 us          890 us          787 GFLOP/s=28.8519/s
BM_MatMul_CPU/64/256/128            145 us          144 us         4903 GFLOP/s=29.0298/s
BM_MatMul_CPU/64/128/10            32.6 us         32.6 us        21446 GFLOP/s=5.02834/s
```

```
     6,391,346,259      task-clock                       #    0.985 CPUs utilized             
               181      context-switches                 #   28.320 /sec                      
                21      cpu-migrations                   #    3.286 /sec                      
           328,200      page-faults                      #   51.351 K/sec                     
     5,774,671,185      cpu_atom/instructions/           #    0.51  insn per cycle              (0.03%)
    57,183,112,275      cpu_core/instructions/           #    2.45  insn per cycle              (35.65%)
    11,294,677,948      cpu_atom/cycles/                 #    1.767 GHz                         (0.03%)
    23,350,587,606      cpu_core/cycles/                 #    3.653 GHz                         (42.79%)
     2,443,486,731      cpu_atom/branches/               #  382.312 M/sec                       (0.05%)
     8,167,530,576      cpu_core/branches/               #    1.278 G/sec                       (49.92%)
       102,100,250      cpu_atom/branch-misses/          #    4.18% of all branches             (0.07%)
        83,043,818      cpu_core/branch-misses/          #    1.02% of all branches             (57.05%)
 #     14.8 %  tma_backend_bound      
                                                  #     21.7 %  tma_bad_speculation    
                                                  #     37.8 %  tma_frontend_bound     
                                                  #     25.8 %  tma_retiring             (64.19%)
 #     28.7 %  tma_bad_speculation    
                                                  #     19.9 %  tma_retiring             (0.09%)
 #     23.3 %  tma_backend_bound      
                                                  #     28.1 %  tma_frontend_bound       (0.11%)
    14,914,861,503      cpu_atom/L1-dcache-loads/        #    2.334 G/sec                       (0.11%)
    16,085,612,407      cpu_core/L1-dcache-loads/        #    2.517 G/sec                       (71.31%)
       839,696,484      cpu_core/L1-dcache-load-misses/  #    5.22% of all L1-dcache accesses   (71.33%)
     1,162,191,375      cpu_atom/LLC-loads/              #  181.838 M/sec                       (0.12%)
        95,471,293      cpu_core/LLC-loads/              #   14.938 M/sec                       (71.35%)
           380,933      cpu_atom/LLC-load-misses/        #    0.03% of all LL-cache accesses    (0.12%)
        82,961,062      cpu_core/LLC-load-misses/        #   86.90% of all LL-cache accesses    (71.36%)
     3,827,853,742      cpu_atom/L1-icache-loads/        #  598.912 M/sec                       (0.10%)
       347,314,327      cpu_atom/L1-icache-load-misses/  #    9.07% of all L1-icache accesses   (0.09%)
       172,161,044      cpu_core/L1-icache-load-misses/                                         (28.51%)
     1,412,703,863      cpu_atom/dTLB-loads/             #  221.034 M/sec                       (0.07%)
    16,106,340,005      cpu_core/dTLB-loads/             #    2.520 G/sec                       (28.52%)
        13,396,728      cpu_atom/dTLB-load-misses/       #    0.95% of all dTLB cache accesses  (0.05%)
         8,002,265      cpu_core/dTLB-load-misses/       #    0.05% of all dTLB cache accesses  (28.53%)
         7,357,983      cpu_atom/iTLB-load-misses/                                              (0.02%)
           654,123      cpu_core/iTLB-load-misses/                                              (28.51%)
```

### Answer

There appears to be much noise. In the scalar version the E-Cores of my CPU get much more heavily utilized. 
Small workload with small tensors in MNIST indicate that noise can dominate. We do not investigate further, 
but instead move on. We consider the benchmark results the more reliable data source.

# Step 3 - Optimize matmul for transposed left case

### Step 3.1

Optimize left transpose for scalar case using tiled matmul approach.

Times after fix: 7.7133s

### Step 3.2

Do the left transposed matmul in AVX as well.

Times after fix: 7.0693s

# Step 4 - Optimize matmul for transposed right case

### Step 4.1

Optimize right transpose for scalar case using tiled matmul approach.

Times after fix: 4.6222s

### Step 4.2

Do the right transposed matmul in AVX as well.

Times after fix: 3.5931s

# Step 5 - Optimize matmul for both transposed case

Will have no impact on our MNIST benchmark, since not used. However, improvement very likely.

# Step 6 - Add AVX (1) support

Minor addition in matmul. We end up with times: 

Times after fix: 3.7613s

### Analysis of why AVX works better than scalar version

[GodBolt](https://godbolt.org/z/Eesbh6Y1r)

Here we can see that compiler chose loop unrolling, not vectorization of inner loop.

# Step 7 - Bypass PLT for backend internal calls

## Analysis

### AVX version

```
    22,603,600,588      task-clock                       #    0.984 CPUs utilized             
             1,652      context-switches                 #   73.086 /sec                      
                84      cpu-migrations                   #    3.716 /sec                      
         1,571,805      page-faults                      #   69.538 K/sec                     
    52,582,262,001      cpu_atom/instructions/           #    1.11  insn per cycle              (0.20%)
   276,718,746,728      cpu_core/instructions/           #    3.30  insn per cycle              (35.58%)
    47,447,603,971      cpu_atom/cycles/                 #    2.099 GHz                         (0.20%)
    83,879,028,747      cpu_core/cycles/                 #    3.711 GHz                         (42.70%)
    10,367,524,657      cpu_atom/branches/               #  458.667 M/sec                       (0.20%)
    47,812,250,940      cpu_core/branches/               #    2.115 G/sec                       (49.82%)
        64,074,152      cpu_atom/branch-misses/          #    0.62% of all branches             (0.17%)
       131,455,461      cpu_core/branch-misses/          #    0.27% of all branches             (56.93%)
 #     14.2 %  tma_backend_bound      
                                                  #     20.4 %  tma_bad_speculation    
                                                  #     39.0 %  tma_frontend_bound     
                                                  #     26.4 %  tma_retiring             (64.04%)
 #      9.5 %  tma_bad_speculation    
                                                  #     30.4 %  tma_retiring             (0.17%)
 #     39.2 %  tma_backend_bound      
                                                  #     20.9 %  tma_frontend_bound       (0.16%)
    42,941,198,696      cpu_atom/L1-dcache-loads/        #    1.900 G/sec                       (0.14%)
   107,563,443,581      cpu_core/L1-dcache-loads/        #    4.759 G/sec                       (71.16%)
     1,051,653,728      cpu_core/L1-dcache-load-misses/  #    0.98% of all L1-dcache accesses   (71.17%)
    18,054,371,905      cpu_atom/LLC-loads/              #  798.739 M/sec                       (0.14%)
       151,626,984      cpu_core/LLC-loads/              #    6.708 M/sec                       (71.19%)
    27,301,494,922      cpu_atom/LLC-load-misses/        #  151.22% of all LL-cache accesses    (0.14%)
       105,557,952      cpu_core/LLC-load-misses/        #   69.62% of all LL-cache accesses    (71.19%)
    41,810,285,455      cpu_atom/L1-icache-loads/        #    1.850 G/sec                       (0.13%)
       333,809,988      cpu_atom/L1-icache-load-misses/  #    0.80% of all L1-icache accesses   (0.16%)
       251,520,387      cpu_core/L1-icache-load-misses/                                         (28.47%)
    41,144,419,666      cpu_atom/dTLB-loads/             #    1.820 G/sec                       (0.17%)
   107,250,186,385      cpu_core/dTLB-loads/             #    4.745 G/sec                       (28.47%)
     6,929,389,786      cpu_atom/dTLB-load-misses/       #   16.84% of all dTLB cache accesses  (0.18%)
        10,092,769      cpu_core/dTLB-load-misses/       #    0.01% of all dTLB cache accesses  (28.46%)
     6,874,382,326      cpu_atom/iTLB-load-misses/                                              (0.18%)
         1,139,100      cpu_core/iTLB-load-misses/                                              (28.46%)

      22.964492129 seconds time elapsed

      20.489733000 seconds user
       2.454709000 seconds sys
```

We find that matmul is still the biggest piece:

```
+   80.62%     0.00%  python3         libBackendCore.so                                            [.] Tensor::matMulImpl(Tensor const&, Tensor const&, bool, bool)      ▒
+   56.50%     0.00%  python3         _core.so                                                     [.] boost::python::objects::caller_py_function_impl<boost::python::det▒
+   56.50%     0.00%  python3         _core.so                                                     [.] boost::python::detail::caller_arity<1u>::impl<void (Tensor::*)(), ▒
+   56.50%     0.00%  python3         _core.so                                                     [.] _object* boost::python::detail::invoke<int, void (Tensor::*)(), bo▒
+   56.50%     0.00%  python3         libBackendCore.so                                            [.] Tensor::backward()                                                ▒
+   53.80%     0.00%  python3         _core.so                                                     [.] cgraph::MatMulNode::backward(Tensor const&)                       ▒
+   50.16%    45.68%  python3         libBackendCore.so                                            [.] std::array<float, 4096ul>::operator[](unsigned long) 
```

Looking into matmul:

```
  11.11 │       mov     -0x88(%rbp),%rax                                                                                                                                 ▒
        │       add     $0x10,%rax                                                                                                                                       ▒
        │       mov     $0xffffffff,%esi                                                                                                                                 ▒
        │       mov     %rax,%rdi                                                                                                                                        ▒
        │     → call    Dimension::get(int) const@plt                                                                                                                    ◆
        │       imul    %ebx,%eax                                                                                                                                        ▒
        │       mov     %eax,-0x54(%rbp)                                                                                                                                 ▒
        │       movl    $0x0,-0x68(%rbp)                                                                                                                                 ▒
        │       movl    $0x0,-0x64(%rbp)                                                                                                                                 ▒
        │       movl    $0x0,-0x60(%rbp)                                                                                                                                 ▒
  22.21 │       movzbl  guard variable for Tensor::matMulImpl(Tensor const&, Tensor const&, bool, bool)::avxVerified,%eax                                                ▒
        │       test    %al,%al                                                                                                                                          ▒
        │       sete    %al                                                                                                                                              ▒
        │       test    %al,%al 
```

This goes through the PLT (Procedure Linkage Table), a rather conservative decision from the compiler to not inline our get() and instead 
introduce indirections in the backend itself. We adjust the compile time options:

```
# optim: avoid PLT on internal calls
target_compile_options(BackendCore PRIVATE
                       -fvisibility=hidden
                       -fvisibility-inlines-hidden)
```

And we have to add this attribute to exposed classes: 

```
#define DLLIB_API __attribute__((visibility("default")))
```

### Times after fix

3.5511s

No significant improvement.

# Step 8 - Fix NDEBUG flag

## Analysis

### AVX version

```
     5,099,919,835      task-clock                       #    0.974 CPUs utilized             
               300      context-switches                 #   58.824 /sec                      
                20      cpu-migrations                   #    3.922 /sec                      
           337,068      page-faults                      #   66.093 K/sec                     
    10,853,495,257      cpu_atom/instructions/           #    1.05  insn per cycle              (0.47%)
    35,447,083,699      cpu_core/instructions/           #    1.93  insn per cycle              (99.46%)
    10,358,963,175      cpu_atom/cycles/                 #    2.031 GHz                         (0.52%)
    18,350,858,653      cpu_core/cycles/                 #    3.598 GHz                         (99.46%)
     2,151,481,525      cpu_atom/branches/               #  421.866 M/sec                       (0.52%)
     6,596,259,569      cpu_core/branches/               #    1.293 G/sec                       (99.46%)
        78,846,050      cpu_atom/branch-misses/          #    3.66% of all branches             (0.52%)
        79,407,361      cpu_core/branch-misses/          #    1.20% of all branches             (99.46%)
 #     41.2 %  tma_backend_bound      
                                                  #     11.0 %  tma_bad_speculation    
                                                  #     17.3 %  tma_frontend_bound     
                                                  #     30.5 %  tma_retiring             (99.46%)
 #     23.9 %  tma_bad_speculation    
                                                  #     22.8 %  tma_retiring             (0.52%)
 #     23.3 %  tma_backend_bound      
                                                  #     29.9 %  tma_frontend_bound       (0.52%)

       5.234938073 seconds time elapsed

       3.847173000 seconds user
       1.351480000 seconds sys
```

Checking out source code we find that this is where we spend most of our horsepower: ```operator[]``` of values_t. Looking insidde: 

```
ftype Tensor::tensorValues_t::operator[](const tensorSize_t idx) const {
#ifdef NDEBUG
  if(idx >= size)
    throw std::out_of_range("Out of range for tensor");
#endif

  switch(device){
    case Device::CPU:
      return values[idx];
    case Device::CUDA:
      #ifdef __CUDA
      {
        ftype res;
        cudaErrchk(cudaMemcpy(&res, values + idx, sizeof(ftype), cudaMemcpyDeviceToHost));
        return res;
      }
      #else
        __throw_invalid_argument("Not compiled with CUDA");
        break;
      #endif
  }

  __throw_runtime_error("Should never reach here.");
  return 0; // suppress warnings
}
```

The beginning should be ifndef for release mode. However, fixing that we realize it bloats the code -> NDEBUG is not set in release mode. 

## Times after fix

CPU: 3.4333s

# Step 9 - Bypass redundant device checks and indirections on memory accesses

## Analysis

```
             1,264      context-switches                 #     60.6 cs/sec  cs_per_second     
                51      cpu-migrations                   #      2.4 migrations/sec  migrations_per_second
         1,576,794      page-faults                      #  75625.4 faults/sec  page_faults_per_second
         20,850.06 msec task-clock                       #      1.0 CPUs  CPUs_utilized       
       124,292,380      cpu_core/branch-misses/          #      0.3 %  branch_miss_rate         (99.87%)
    45,702,464,699      cpu_core/branches/               #   2192.0 M/sec  branch_frequency     (99.87%)
    78,178,161,324      cpu_core/cpu-cycles/             #      3.7 GHz  cycles_frequency       (99.87%)
   253,442,402,566      cpu_core/instructions/           #      3.2 instructions  insn_per_cycle  (99.87%)
        13,597,279      cpu_atom/branch-misses/          #      0.1 %  branch_miss_rate         (0.06%)
    18,837,028,680      cpu_atom/branches/               #    903.5 M/sec  branch_frequency     (0.06%)
    46,637,108,491      cpu_atom/cpu-cycles/             #      2.2 GHz  cycles_frequency       (0.06%)
    70,830,711,657      cpu_atom/instructions/           #      1.7 instructions  insn_per_cycle  (0.08%)
             TopdownL1 (cpu_core)                        #     12.6 %  tma_bad_speculation    
                                                         #     16.4 %  tma_frontend_bound       (99.87%)
                                                         #     36.3 %  tma_backend_bound      
                                                         #     34.7 %  tma_retiring             (99.87%)
             TopdownL1 (cpu_atom)                        #     18.1 %  tma_backend_bound        (0.12%)
                                                         #     28.8 %  tma_frontend_bound       (0.12%)
                                                         #      0.1 %  tma_bad_speculation    
                                                         #     53.0 %  tma_retiring             (0.09%)
```

Perf record gives us this:

```
   6.45%  python3  libc.so.6                                          [.] __memmove_avx_unaligned_erms                                                                 ◆
   4.34%  python3  [kernel.kallsyms]                                  [k] osDevReadReg032                                                                              ▒
   2.85%  python3  [kernel.kallsyms]                                  [k] kernel_init_pages                                                                            ▒
   2.73%  python3  parsers.cpython-314-x86_64-linux-gnu.so            [.] 0x000000000003aef8                                                                           ▒
   2.59%  python3  python3.14                                         [.] _PyEval_EvalFrameDefault                                                                     ▒
   2.44%  python3  libBackendCore.so                                  [.] Tensor::tensorValues_t::operator[](unsigned int) const                                       ▒
   2.30%  python3  libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<true, false>(Tensor&, Tensor const&, Tensor const&, unsigned i▒
   1.97%  python3  libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<false, false>(Tensor&, Tensor const&, Tensor const&, unsigned ▒
   1.92%  python3  libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<false, true>(Tensor&, Tensor const&, Tensor const&, unsigned i▒
   1.33%  python3  libBackendCore.so                                  [.] train::RmsPropOptimizer::step()                                                              ▒
   1.17%  python3  parsers.cpython-314-x86_64-linux-gnu.so            [.] 0x000000000003aefc                                                                           ▒
   1.07%  python3  libBackendCore.so                                  [.] train::OptimizerBase::clipGradients(float)                                                   ▒
   1.03%  python3  libz.so.1.3.1                                      [.] crc32_z                                                                                      ▒
   1.01%  python3  libBackendCore.so                                  [.] Tensor::tensorValues_t::set(float, unsigned int)                                             ▒
   0.82%  python3  _multiarray_umath.cpython-314-x86_64-linux-gnu.so  [.] 0x000000000006f929                                                                           ▒
   0.80%  python3  libBackendCore.so                                  [.] Tensor::tensorValues_t::operator[](unsigned int) const@plt  
```

The memmove is interesting, but at the moment the analysis is cluttered with calls to operator[], get() and set() of tensorvalues_t still. 
We will refactor that, removing those operators to use the data directly in CPU mode, copying what CUDA did as well. We also kick out a few checks like dimension checks in release mode. Worse error messages, but in practive logging the input is enough to debug the problem. -> message/performance trade-off

## Times after fix

CPU: 2.8825s
GPU: 0.2070s (small boost)

# Step 10 - Fix missed bad memory accesses

## Analysis

```
             3,157      context-switches                 #    173.0 cs/sec  cs_per_second     
                73      cpu-migrations                   #      4.0 migrations/sec  migrations_per_second
         1,576,637      page-faults                      #  86384.2 faults/sec  page_faults_per_second
         18,251.45 msec task-clock                       #      1.0 CPUs  CPUs_utilized       
       128,916,849      cpu_core/branch-misses/          #      0.4 %  branch_miss_rate         (99.85%)
    32,241,277,245      cpu_core/branches/               #   1766.5 M/sec  branch_frequency     (99.85%)
    68,006,476,844      cpu_core/cpu-cycles/             #      3.7 GHz  cycles_frequency       (99.85%)
   207,423,094,719      cpu_core/instructions/           #      3.1 instructions  insn_per_cycle  (99.85%)
       140,331,635      cpu_atom/branch-misses/          #      1.7 %  branch_miss_rate         (0.11%)
     8,229,824,661      cpu_atom/branches/               #    450.9 M/sec  branch_frequency     (0.11%)
    33,153,607,692      cpu_atom/cpu-cycles/             #      1.8 GHz  cycles_frequency       (0.09%)
    34,866,596,099      cpu_atom/instructions/           #      1.1 instructions  insn_per_cycle  (0.11%)
             TopdownL1 (cpu_core)                        #      8.1 %  tma_bad_speculation
                                                         #     11.2 %  tma_frontend_bound       (99.85%)
                                                         #     36.6 %  tma_backend_bound
                                                         #     44.1 %  tma_retiring             (99.85%)
             TopdownL1 (cpu_atom)                        #     21.7 %  tma_backend_bound        (0.14%)
                                                         #     32.0 %  tma_frontend_bound       (0.13%)
                                                         #     19.6 %  tma_bad_speculation    
                                                         #     26.8 %  tma_retiring             (0.11%)
```

We see backend bound for core CPUs (performance cores) is bottleneck now.


### Note

Because CPU is much faster now, we can move to larger inputs when profiling via ```perf record```. Before we had 2 epochs, 
50 batches each, and now increased to 2 epochs, 500 batches each. This is what the new first profile looks like (we obviously compiled
non-AVX):

```
Samples: 37K of event 'cpu_core/cycles/P', Event count (approx.): 34604279217
Overhead  Command         Shared Object                                      Symbol
  13.25%  python3         libBackendCore.so                                          [.] void Tensor::matMul2DCpuScalar<true, false>(Tensor&, Tensor const&, Tensor con◆
  11.77%  python3         libBackendCore.so                                          [.] void Tensor::matMul2DCpuScalar<false, true>(Tensor&, Tensor const&, Tensor con▒
  11.38%  python3         libBackendCore.so                                          [.] void Tensor::matMul2DCpuScalar<false, false>(Tensor&, Tensor const&, Tensor co▒
   9.40%  python3         libBackendCore.so                                          [.] Tensor::tensorValues_t::operator[](unsigned int) const                        ▒
   8.99%  python3         libBackendCore.so                                          [.] train::RmsPropOptimizer::step()        
```

Someone is still calling ```tensorValues_t::operator[]``` suspiciously often. Looks like we overlooked some calls in RmsProp (and SGD).

## Times after fix

CPU (scalar matmul): 2.2999s
CPU (AVX matmul): 1.9637s

## Sidenote

Step 7 broke the AVX test, so the steps in between probably all benchmarked on the non-AVX version.

# Step 11 - Parallelize matmul

## Analysis

```
               470      context-switches                 #     31.1 cs/sec  cs_per_second     
                16      cpu-migrations                   #      1.1 migrations/sec  migrations_per_second
         1,575,579      page-faults                      # 104377.9 faults/sec  page_faults_per_second
         15,094.95 msec task-clock                       #      1.0 CPUs  CPUs_utilized       
       130,626,000      cpu_core/branch-misses/          #      0.9 %  branch_miss_rate         (99.95%)
    13,990,559,350      cpu_core/branches/               #    926.8 M/sec  branch_frequency     (99.95%)
    56,489,241,835      cpu_core/cpu-cycles/             #      3.7 GHz  cycles_frequency       (99.95%)
   132,728,185,717      cpu_core/instructions/           #      2.3 instructions  insn_per_cycle  (99.95%)
       130,600,128      cpu_atom/branch-misses/          #      8.7 %  branch_miss_rate         (0.07%)
     2,509,253,475      cpu_atom/branches/               #    166.2 M/sec  branch_frequency     (0.04%)
    28,127,357,777      cpu_atom/cpu-cycles/             #      1.9 GHz  cycles_frequency       (0.02%)
    33,475,389,947      cpu_atom/instructions/           #      1.1 instructions  insn_per_cycle  (0.03%)
             TopdownL1 (cpu_core)                        #      9.2 %  tma_bad_speculation    
                                                         #     12.8 %  tma_frontend_bound       (99.95%)
                                                         #     44.0 %  tma_backend_bound      
                                                         #     33.9 %  tma_retiring             (99.95%)
             TopdownL1 (cpu_atom)                        #     21.2 %  tma_backend_bound        (0.09%)
                                                         #     29.3 %  tma_frontend_bound       (0.08%)
                                                         #     40.2 %  tma_bad_speculation    
                                                         #      9.3 %  tma_retiring             (0.08%)
```

```
  15.74%  python3         libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<true, false>(Tensor&, Tensor const&, Tensor const&, unsi
  13.60%  python3         libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<false, true>(Tensor&, Tensor const&, Tensor const&, unsi
  13.58%  python3         libBackendCore.so                                  [.] void Tensor::matMul2DCpuScalar<false, false>(Tensor&, Tensor const&, Tensor const&, uns
   7.08%  python3         libBackendCore.so                                  [.] train::RmsPropOptimizer::step()
   3.29%  python3         libc.so.6                                          [.] __memmove_avx_unaligned_erms
   3.06%  python3         libBackendCore.so                                  [.] matmul::MatmulTile<float, 64u, 64u, 64u>::loadLeft(float const*, unsigned int, unsigned
   3.00%  python3         libBackendCore.so                                  [.] matmul::MatmulTile<float, 64u, 64u, 64u>::loadRight(float const*, unsigned int, unsigne
   2.61%  python3         libBackendCore.so                                  [.] train::OptimizerBase::clipGradients(float)
   1.82%  python3         libBackendCore.so                                  [.] matmul::MatmulTile<float, 64u, 64u, 64u>::loadRightTransposed(float const*, unsigned in
   1.55%  python3         libBackendCore.so                                  [.] Tensor::reset(float)
   1.45%  python3         libBackendCore.so                                  [.] matmul::MatmulTile<float, 64u, 64u, 64u>::loadLeftTransposed(float const*, unsigned int
   1.14%  python3         libBackendCore.so                                  [.] cgraph::LeakyReLuNode::backward(Tensor const&)
```

Matmul takes around 40% of runtime still. L1 and L2 cache hit rate are good, so we can make better use of those by parallelizing, apart from the computational work being offloaded.

## Times after fix

CPU (scalar): 2.7338s
CPU (AVX): 2.4397s

## Benchmarking

Input likely too small. Check how it scales over sizes. Git commits: 
- Benchmarking single threaded = ca207be1a174a491919d58db3b7862a0d8a29a0e
- Benchmarking multi threaded: Set the PARALLEL_MATMUL_THRESHOLD smaller. We do really small and have the benchmarking results tell us what our machine can do.


### Single-threaded results:

```
------------------------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64                            15.3 us         15.3 us        46222 GFLOP/s=34.3593/s
BM_MatMul_CPU/256/256/256                          826 us          825 us          823 GFLOP/s=40.6685/s
BM_MatMul_CPU/512/512/512                         6327 us         6326 us           85 GFLOP/s=42.4317/s
BM_MatMul_CPU/1024/1024/1024                     51605 us        51600 us           13 GFLOP/s=41.6181/s
BM_MatMul_CPU/64/784/256                           627 us          627 us         1115 GFLOP/s=40.9986/s
BM_MatMul_CPU/64/256/128                           103 us          103 us         6797 GFLOP/s=40.6943/s
BM_MatMul_CPU/64/128/10                           22.9 us         22.9 us        30559 GFLOP/s=7.15418/s
BM_MatMul_TransposeLeft_CPU/64/64/64              15.5 us         15.5 us        45252 GFLOP/s=33.8764/s
BM_MatMul_TransposeLeft_CPU/256/256/256            805 us          805 us          868 GFLOP/s=41.6671/s
BM_MatMul_TransposeLeft_CPU/512/512/512           6599 us         6598 us          105 GFLOP/s=40.684/s
BM_MatMul_TransposeLeft_CPU/1024/1024/1024       58801 us        58794 us           12 GFLOP/s=36.5258/s
BM_MatMul_TransposeLeft_CPU/64/784/256             619 us          619 us         1136 GFLOP/s=41.5211/s
BM_MatMul_TransposeLeft_CPU/64/256/128             101 us          101 us         6897 GFLOP/s=41.3428/s
BM_MatMul_TransposeLeft_CPU/64/128/10             22.8 us         22.8 us        30669 GFLOP/s=7.18353/s
BM_MatMul_TransposeRight_CPU/64/64/64             15.5 us         15.5 us        45160 GFLOP/s=33.8858/s
BM_MatMul_TransposeRight_CPU/256/256/256           788 us          788 us          887 GFLOP/s=42.6065/s
BM_MatMul_TransposeRight_CPU/512/512/512          6450 us         6450 us          107 GFLOP/s=41.6163/s
BM_MatMul_TransposeRight_CPU/1024/1024/1024      52159 us        52157 us           13 GFLOP/s=41.1734/s
BM_MatMul_TransposeRight_CPU/64/784/256            611 us          611 us         1145 GFLOP/s=42.05/s
BM_MatMul_TransposeRight_CPU/64/256/128           99.7 us         99.6 us         7030 GFLOP/s=42.0926/s
BM_MatMul_TransposeRight_CPU/64/128/10            22.8 us         22.8 us        30723 GFLOP/s=7.19213/s
BM_MatMul_TransposeBoth_CPU/64/64/64              15.4 us         15.4 us        45612 GFLOP/s=34.1363/s
BM_MatMul_TransposeBoth_CPU/256/256/256            789 us          789 us          885 GFLOP/s=42.5396/s
BM_MatMul_TransposeBoth_CPU/512/512/512           6843 us         6843 us          102 GFLOP/s=39.2276/s
BM_MatMul_TransposeBoth_CPU/1024/1024/1024       61953 us        61946 us           12 GFLOP/s=34.6669/s
BM_MatMul_TransposeBoth_CPU/64/784/256             607 us          607 us         1143 GFLOP/s=42.3296/s
BM_MatMul_TransposeBoth_CPU/64/256/128            99.2 us         99.2 us         7076 GFLOP/s=42.2696/s
BM_MatMul_TransposeBoth_CPU/64/128/10             22.9 us         22.9 us        30589 GFLOP/s=7.16467/s
BM_MatMul_TransposeBoth_CPU/64/128/10             53.6 us         53.6 us        13041 GFLOP/s=3.05776/s
```

### Multi-threaded results

```
------------------------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------
BM_MatMul_CPU/64/64/64                            37.6 us         8.41 us       115295 GFLOP/s=62.3119/s
BM_MatMul_CPU/256/256/256                          412 us         30.6 us        10000 GFLOP/s=1.09721k/s
BM_MatMul_CPU/512/512/512                         1865 us         91.6 us         9034 GFLOP/s=2.9306k/s
BM_MatMul_CPU/1024/1024/1024                     14280 us          376 us         1000 GFLOP/s=5.71453k/s
BM_MatMul_CPU/64/784/256                           840 us         29.9 us        10000 GFLOP/s=859.153/s
BM_MatMul_CPU/64/256/128                           120 us         6.83 us        96085 GFLOP/s=614.491/s
BM_MatMul_CPU/64/128/10                           38.5 us         5.44 us       114803 GFLOP/s=30.0996/s
BM_MatMul_TransposeLeft_CPU/64/64/64              26.5 us         5.43 us       122380 GFLOP/s=96.5505/s
BM_MatMul_TransposeLeft_CPU/256/256/256            418 us         26.6 us        10000 GFLOP/s=1.26089k/s
BM_MatMul_TransposeLeft_CPU/512/512/512           1964 us         88.6 us         7165 GFLOP/s=3.02856k/s
BM_MatMul_TransposeLeft_CPU/1024/1024/1024       16629 us          438 us         1000 GFLOP/s=4.90614k/s
BM_MatMul_TransposeLeft_CPU/64/784/256             838 us         31.1 us        10000 GFLOP/s=826.476/s
BM_MatMul_TransposeLeft_CPU/64/256/128             126 us         7.01 us        74557 GFLOP/s=597.984/s
BM_MatMul_TransposeLeft_CPU/64/128/10             40.0 us         5.65 us        93758 GFLOP/s=29.0097/s
BM_MatMul_TransposeRight_CPU/64/64/64             25.0 us         5.20 us       102715 GFLOP/s=100.867/s
BM_MatMul_TransposeRight_CPU/256/256/256           429 us         29.7 us        10000 GFLOP/s=1.13013k/s
BM_MatMul_TransposeRight_CPU/512/512/512          2009 us         88.4 us         7275 GFLOP/s=3.03582k/s
BM_MatMul_TransposeRight_CPU/1024/1024/1024      14938 us          355 us         1000 GFLOP/s=6.04354k/s
BM_MatMul_TransposeRight_CPU/64/784/256            779 us         26.6 us        10000 GFLOP/s=965.514/s
BM_MatMul_TransposeRight_CPU/64/256/128            124 us         6.89 us        90767 GFLOP/s=608.727/s
BM_MatMul_TransposeRight_CPU/64/128/10            38.7 us         5.55 us        83965 GFLOP/s=29.516/s
BM_MatMul_TransposeBoth_CPU/64/64/64              28.3 us         5.78 us       122721 GFLOP/s=90.7307/s
BM_MatMul_TransposeBoth_CPU/256/256/256            450 us         30.0 us        10000 GFLOP/s=1.11716k/s
BM_MatMul_TransposeBoth_CPU/512/512/512           2155 us         91.6 us         6979 GFLOP/s=2.93062k/s
BM_MatMul_TransposeBoth_CPU/1024/1024/1024       16238 us          412 us         1000 GFLOP/s=5.20687k/s
BM_MatMul_TransposeBoth_CPU/64/784/256             846 us         36.0 us        10000 GFLOP/s=714.032/s
BM_MatMul_TransposeBoth_CPU/64/256/128             124 us         6.89 us       109234 GFLOP/s=608.322/s
BM_MatMul_TransposeBoth_CPU/64/128/10             45.1 us         6.32 us       122120 GFLOP/s=25.9169/s
```

Result: Multithreaded version wins even on small matrices.

## Analysis

```
           456,478      context-switches                 #   8448.8 cs/sec  cs_per_second     
            14,307      cpu-migrations                   #    264.8 migrations/sec  migrations_per_second
         2,759,220      page-faults                      #  51069.3 faults/sec  page_faults_per_second
         54,028.98 msec task-clock                       #      1.3 CPUs  CPUs_utilized       
       240,175,831      cpu_core/branch-misses/          #      0.9 %  branch_miss_rate         (80.51%)
    26,088,083,098      cpu_core/branches/               #    482.9 M/sec  branch_frequency     (80.51%)
   123,791,816,818      cpu_core/cpu-cycles/             #      2.3 GHz  cycles_frequency       (80.51%)
   257,499,493,434      cpu_core/instructions/           #      2.1 instructions  insn_per_cycle  (80.51%)
       105,103,579      cpu_atom/branch-misses/          #      0.7 %  branch_miss_rate         (9.84%)
    16,135,415,551      cpu_atom/branches/               #    298.6 M/sec  branch_frequency     (9.76%)
   106,572,280,463      cpu_atom/cpu-cycles/             #      2.0 GHz  cycles_frequency       (9.81%)
   185,969,027,235      cpu_atom/instructions/           #      1.7 instructions  insn_per_cycle  (9.84%)
             TopdownL1 (cpu_core)                        #      5.7 %  tma_bad_speculation    
                                                         #      9.5 %  tma_frontend_bound       (80.51%)
                                                         #     45.9 %  tma_backend_bound      
                                                         #     38.9 %  tma_retiring             (80.51%)
             TopdownL1 (cpu_atom)                        #     35.5 %  tma_backend_bound        (11.70%)
                                                         #      8.6 %  tma_frontend_bound       (11.70%)
                                                         #      3.6 %  tma_bad_speculation    
                                                         #     52.3 %  tma_retiring             (11.71%)
```

Use lots of E-cores. Therefore: Core pinning. Finding the P-cores via ```lscpu --extended``` reveals:

```
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ       MHZ
  0    0      0    0 0:0:0:0          yes 4800.0000 400.0000 1159.6340
  1    0      0    0 0:0:0:0          yes 4800.0000 400.0000 1419.8870
  2    0      0    1 4:4:1:0          yes 4800.0000 400.0000 1512.5770
  3    0      0    1 4:4:1:0          yes 4800.0000 400.0000  400.0000
  4    0      0    2 8:8:2:0          yes 4800.0000 400.0000 1588.9550
  5    0      0    2 8:8:2:0          yes 4800.0000 400.0000  962.6430
  6    0      0    3 12:12:3:0        yes 4800.0000 400.0000 1310.6790
  7    0      0    3 12:12:3:0        yes 4800.0000 400.0000  400.0000
  8    0      0    4 20:20:5:0        yes 3600.0000 400.0000 1656.5811
  9    0      0    5 21:21:5:0        yes 3600.0000 400.0000  400.0000
 10    0      0    6 22:22:5:0        yes 3600.0000 400.0000  400.0000
 11    0      0    7 23:23:5:0        yes 3600.0000 400.0000  491.1300
```

