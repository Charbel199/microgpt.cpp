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

The original [Python](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) implementation takes around 150 ms per step and about 2 minutes 30 seconds in total to train.

This C++ implementation takes around 0.3 ms per step and about 0.3 seconds in total to train.

This corresponds to a speedup of roughly 400-500×. Andrej Karpathy’s goal was not performance but extreme readability and ease of understanding; this version focuses on optimization while remaining reasonably readable because otpimization is fun.