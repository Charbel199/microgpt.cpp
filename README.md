# MICROGPT.CPP

A C++ implementation of a minimal GPT model inspired by Andrej Karpathy’s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), using only the C++ standard library and a simple memory arena allocator.

The focus is on readability (and optimization) rather than minimizing line count.

# Build & Run

```
g++ -std=c++17 -DDEBUG -Ofast -march=native microgpt.cpp -o microgpt
./microgpt
```

You can remove `-DDEBUG` for slightly faster execution. For maximum performance:

# Performance

Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) was never meant to be fast, its goal is extreme readability and making it easy to learn how a GPT works from scratch. This C++ version focuses on optimization while remaining reasonably readable (barely at this point), because optimization is fun.

All benchmarks run on `Intel Core Ultra 7 165H`, using [PyPy JIT](https://pypy.org/) as the Python performance baseline.

_Note: 16x16 = N_EMBD=16, BLOCK_SIZE=16_

### 16x16 network, 10000 steps

| Implementation | Time | vs PyPy JIT | Main changes |
|---|---|---|---|
| Python (CPython) | 22m 4s | ~6.7x slower | |
| Python (PyPy JIT) | 3m 16s | 1x | JIT compilation |
| [C++ (original)](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4) | 3.3s | **~60x** | Wengert tape, AoS arena, `f64` |
| [C++ (enhanced)](https://github.com/Charbel199/microgpt.cpp/blob/88022beee52dc0f04b7285aea65f27822a5a0e74) | 2.2s | **~88x** | + SoA arena, flat KV cache, stack arrays |
| [Rust](https://github.com/mplekh/rust-microgpt/blob/5089e92df15a0e4955ff23d3a5a62d7b15f97616) | 2.0s | **~97x** | + Op enum backward, `f32`, unsafe ptrs |
| C++ (current) / [Rust (current)](https://github.com/mplekh/rust-microgpt) | 1.3s | **~152x** | + true FMA, stack KV cache, `-Ofast` |

### 64x64 network, 1000 steps

| Implementation | Time | vs PyPy JIT | Main changes |
|---|---|---|---|
| Python (CPython) | 1h 14m | ~10.9x slower | |
| Python (PyPy JIT) | 6m 47s | 1x | JIT compilation |
| [C++ (original)](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4) | 8.2s | **~50x** | Wengert tape, AoS arena, `f64` |
| [C++ (enhanced)](https://github.com/Charbel199/microgpt.cpp/blob/88022beee52dc0f04b7285aea65f27822a5a0e74) | 3.5s | **~115x** | + SoA arena, flat KV cache, stack arrays |
| [Rust](https://github.com/mplekh/rust-microgpt/blob/5089e92df15a0e4955ff23d3a5a62d7b15f97616) | 2.6s | **~157x** | + Op enum backward, `f32`, unsafe ptrs |
| C++ (current) / [Rust (current)](https://github.com/mplekh/rust-microgpt) | 1.6s | **~249x** | + true FMA, stack KV cache, `-Ofast` |

_C++ (original/enhanced) compiled with `g++ -std=c++17 -O3`. C++ (current) compiled with `g++ -std=c++17 -Ofast -march=native`. PyPy benchmarked with [PyPy 7.3.17](https://pypy.org/download.html). Rust compilation profile:_

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

### Rust

The following [rust implementation](https://github.com/mplekh/rust-microgpt/blob/5089e92df15a0e4955ff23d3a5a62d7b15f97616) is faster than [C++ (original)](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4) out of the box. By introducing autograd optimizations, using unsafe pointers, calculating derivatives during the backward pass, and only recording operations during the forward pass, it became faster than [C++ (enhanced)](https://github.com/Charbel199/microgpt.cpp/blob/88022beee52dc0f04b7285aea65f27822a5a0e74). However, C++ (Current) builds on these optimizations and adds true fused multiply-add (FMA), computing gradients only during the backward pass, utilizing `f32`, and compiling with `-Ofast` (matched with the latest [Rust code](https://github.com/mplekh/rust-microgpt)).

---

_For a version that replicates Python's random numbers for an exact match of the original Python output, see [this fork](https://github.com/AntonTimofeev/microgpt.cpp/tree/use_python_random)._




