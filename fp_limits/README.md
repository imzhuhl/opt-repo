# Peak Floating-point Performance

## Introduction

Test peak GFLOPS of arm processors.

The support architecture is: armv8 with sve and bf16 extention


## How to do it

1. Select instructions
2. Check latency and throughput
3. Write the appropriate assembly code

### Select instructions
For example, I select FMLA instructions and the data type is float32.
Assume that the length of our vector register is 128 bit, thus four 32-bit floating point numbers can be put in.

The basic approach is to write a loop, then count the time, and finally calculate the result, so our kernel code would looks like:

```
.loop:
    fmla z10.s, p0/M, z0.s, z2.s
    fmla z11.s, p0/M, z0.s, z3.s
    fmla z12.s, p0/M, z0.s, z4.s
    fmla z13.s, p0/M, z0.s, z5.s
    fmla z14.s, p0/M, z1.s, z2.s
    fmla z15.s, p0/M, z1.s, z3.s
    fmla z16.s, p0/M, z1.s, z4.s
    fmla z17.s, p0/M, z1.s, z5.s
    subs x0, x0, #1
    bne .loop
```

Two points need to be noted here: 
1. Why write 8 FMLA instructions in one loop?
2. Why use so many registers?

### Check latency and throughput

For my machine, I checked the arm official manual. The FMLA's latency is 4 and throughput is 2.

Latency and throughput are related to instruction pipelining.
Latency means it takes 4 cycles for one FMLA instruction to finish executing, and throughput means how many FMLA instructions are executed per cycle.
Specifically, this means that the FMLA pipeline is divided into 4 stages, and the next FMLA can be fired after one cycle after the previous FMLA is fired. At the same time, there are two execution units that can execute fmla simultaneously, so the throughput can reach 2.

In addition, some register in FMLA has to be both read and written, such as z10.

TODO...



## Results

I test the GFLOPS on Arm Neoverse N2 platform:

* FMLA: 43.883 GFLOP/S
* BFMMLA: 175.578 GFLOP/S

