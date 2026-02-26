# MICROGPT.CPP

A C++ implementation of a minimal GPT model inspired by Andrej Karpathy’s [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), using only the C++ standard library and a simple memory arena allocator.

The focus is on readability (and optinmization) rather than minimizing line count.

# Build & Run

```
g++ -std=c++17 -O3 -DDEBUG microgpt.cpp -o microgpt
./microgpt
```

You can remove `-DDEBUG` for slightly faster execution.

# Performance

Andrej Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) was never meant to be fast, its goal is extreme readability and making it easy to learn how a GPT works from scratch. This C++ version focuses on optimization while remaining reasonably readable (barely at this point), because optimization is fun.

On average, the C++ implementation achieves a ~~**~400-500x speedup**~~ **~600x+ speedup** over the Python version depending on network size on my `Intel Core Ultra 7 165H`.

Benchmarks run on the same machine, compiled with `g++ -std=c++17 -O3`.

_Note: 16x16 = N_EMBD=16, BLOCK_SIZE=16_

### 16x16 network, 10000 steps

| Implementation | Time | Speedup |
|---|---|---|
| Python | 22m 4s | 1x |
| C++ (original) | 3.3s | **~400x** |
| C++ (enhanced) | 2.2s | **~600x** |

### 64x64 network, 1000 steps

| Implementation | Time | Speedup |
|---|---|---|
| Python | 1h 14m | 1x |
| C++ (original) | 8.2s | **~540x** |
| C++ (enhanced) | 2.9s | **~1530x** |

_C++ (original) refers to commit [`3e49721`](https://github.com/Charbel199/microgpt.cpp/blob/3e49721ea766058cae617d7fe43092caee1198d4). C++ (enhanced) is the latest version._