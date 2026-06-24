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

# Step 2: Optimize matmul

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

## Step 2-3: 

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