import gc
import platform
import subprocess
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import mlx.core as mx


def get_chip_model():
    # Use 'sysctl' to get information about the Apple Silicon chip
    try:
        output = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
        )
        return output
    except subprocess.CalledProcessError as e:
        return f"Error retrieving chip model: {e}"


chip = get_chip_model()
bandwidth = {
    "Apple M1 Max": 2**30 * 400,  # https://en.wikipedia.org/wiki/Apple_M1#Memory
    "Apple M2 Ultra": 2**30
    * 800,  # https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra
}[chip]
roof = {
    "Apple M1 Max": 2**40 * 10.4,  # https://en.wikipedia.org/wiki/Apple_M1#GPU
    "Apple M2 Ultra": 2**40 * 27.2,  # https://en.wikipedia.org/wiki/Apple_M2#GPU
}[chip]

DT = mx.float16
R = 10

aint: List[float] = []
perf: List[float] = []

for i in range(16):
    N = 2**i

    a = mx.random.uniform(-1.0, 1.0, [N, N], dtype=DT)
    b = mx.random.uniform(-1.0, 1.0, [N, N], dtype=DT)
    mx.eval(a)
    mx.eval(b)
    duration = 0

    for r in range(R):
        start_time = time.perf_counter()
        c = a @ b
        mx.eval(c)
        duration += time.perf_counter() - start_time

    aint.append(N / 3.0)  # 3.0 due to sizeof(fp16)
    perf.append(N**3 / duration * R * 2)

    del a, b, c
    gc.collect()

diag = [bandwidth * ai for ai in aint]
roof = [roof for _ in aint]

plt.figure(figsize=(8, 6))
plt.loglog(aint, perf, "o", markersize=8, label="matmul performance")
plt.loglog(aint, diag, "-", linewidth=2, label="memory access bound")
plt.loglog(aint, roof, "-", linewidth=2, label="computation bound")
plt.xlabel("arithmetic intensity (f32-ops/byte)")
plt.ylabel("performance (f32-ops/sec)")
plt.title(f"Roofline analysis of matmul on {chip}")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.show()
