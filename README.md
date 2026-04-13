# CUDA-Accelerated Box Blur (Mean Filter)

PCAP Mini Project — Manipal Institute of Technology  
**Authors:** Abhishek Kumar (230962286) & Tejasveer Singh (230962224)  
**Guide:** Dr. Vidya Kamath, Assistant Professor

---

## What This Does
Accelerates a 5x5 Box Blur (Mean Filter) on greyscale images using NVIDIA CUDA.  
Benchmarks GPU vs CPU performance across three resolutions.

## Key Results
| Resolution | CPU Time | GPU Time | Speedup | PSNR |
|---|---|---|---|---|
| 512×512 | 62.45 ms | 0.842 ms | 74.17x | 28.43 dB |
| 1024×1024 | 248.31 ms | 2.156 ms | 115.17x | 28.43 dB |
| 1920×1080 | 489.72 ms | 3.847 ms | 127.30x | 28.43 dB |

## Optimisation Techniques
- **Shared Memory Tiling** — 20×20 tile per block, reduces global memory reads from 6400 to 400
- **Coalesced Memory Access** — 32 reads merged into 1 transaction
- **Warmup Kernel** — eliminates JIT overhead from benchmark timing

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit 13.x
- Visual Studio Build Tools (MSVC)
- STB headers: `stb_image.h` and `stb_image_write.h`

## How to Run
```bash
nvcc box_blur.cu -o box_blur.exe -O2
box_blur.exe
```

## Hardware Used
- CPU: AMD Ryzen 7 5800H
- GPU: NVIDIA RTX 3050 Laptop (2048 CUDA cores, 4GB VRAM)
- RAM: 16GB DDR4
