The results follow the expected behavior for OpenMP GPU offload. At the
default 200 x 100 grid size, the CPU version is faster because the problem is
too small to hide GPU offload and kernel launch overhead. As the grid size
increases, the GPU has enough parallel work to become much more efficient.

The CPU and GPU kinetic energy values are very close in every case. Small
differences are normal because floating-point reductions can happen in a
different order on CPU and GPU.

The results are shown in the figures:

- runtime_comparison.png compares CPU and GPU runtime directly.
- gpu_speedup.png shows how many times faster the GPU is than the CPU.

The main conclusion: the GPU is not useful for the smallest case,
 but it scales much better and gives strong speedup for the
larger grids.

| Grid size | Scale | CPU runtime (s) | GPU runtime (s) | Speedup (CPU/GPU) |
|---:|---:|---:|---:|---:|
| 200 x 100 | 1x | 0.164190 | 0.537264 | 0.31x |
| 800 x 400 | 4x | 1.750971 | 0.610888 | 2.87x |
| 1600 x 800 | 8x | 8.933740 | 0.974076 | 9.17x |
| 3200 x 1600 | 16x | 53.261161 | 2.290994 | 23.25x |
