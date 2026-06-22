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

