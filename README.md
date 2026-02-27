# MICROGPT.CPP

A C++ implementation of a minimal GPT model inspired by Andrej Karpathy’s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), using only the C++ standard library and a simple memory arena allocator.

The focus is on readability (and optimization) rather than minimizing line count.

# Build & Run

```
g++ -std=c++17 -O3 -DDEBUG microgpt.cpp -o microgpt
./microgpt
```

You can remove `-DDEBUG` for slightly faster execution.

Adding `-march=native -ffast-math` can squeeze out another couple of % (haven't fully tested that yet).

# Performance

Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) was never meant to be fast, its goal is extreme readability and making it easy to learn how a GPT works from scratch. This C++ version focuses on optimization while remaining reasonably readable (barely at this point), because optimization is fun.

All benchmarks run on `Intel Core Ultra 7 165H`, using [PyPy JIT](https://pypy.org/) as the Python performance baseline.

_Note: 16x16 = N_EMBD=16, BLOCK_SIZE=16_

### 16x16 network, 10000 steps

| Implementation | Time | vs PyPy JIT |
|---|---|---|
| Python (CPython) | 22m 4s | ~6.7x slower |
| Python (PyPy JIT) | 3m 16s | 1x |
| [C++ (original)](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4) | 3.3s | **~60x** |
| [Rust](https://github.com/mplekh/rust-microgpt) | 2.9s | **~68x** |
| C++ (enhanced) | 2.2s | **~89x** |

### 64x64 network, 1000 steps

| Implementation | Time | vs PyPy JIT |
|---|---|---|
| Python (CPython) | 1h 14m | ~10.9x slower |
| Python (PyPy JIT) | 6m 47s | 1x |
| [C++ (original)](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4) | 8.2s | **~50x** |
| [Rust](https://github.com/mplekh/rust-microgpt) | 4.8s | **~85x** |
| C++ (enhanced) | 2.9s | **~140x** |

_C++ compiled with `g++ -std=c++17 -O3`. PyPy benchmarked with [PyPy 7.3.17](https://pypy.org/download.html). Rust compilation profile:_

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
```

C++ (enhanced) is the most performant but requires the most optimization effort.

### Rust

The [rust implementation](https://github.com/mplekh/rust-microgpt) is already faster than C++ (original) out of the box. The remaining gap vs C++ (enhanced) comes from bounds checking on every `Vec::push`, which the C++ version avoids through a pre-allocated arena.