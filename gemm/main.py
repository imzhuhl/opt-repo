import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np

N = 1024

if __name__ == "__main__":
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32) 

    flop = 2.0*N*N*N
    for i in range(5):
        st = time.monotonic()
        C = A @ B
        et = time.monotonic()
        s = et-st
        print(f"{flop/s * 1e-9:.2f} GFLOP/S, {s*1e3:.2f} ms")
