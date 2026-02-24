# MICROGPT.CPP

A C++ implementation of a minimal GPT model inspired by Andrej Karpathy’s[microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), using only the C++ standard library and a simple memory arena allocator.

The focus is on readability rather than minimizing line count.

# Build & Run

```
g++ -std=c++17 microgpt.cpp -o microgpt
./microgpt
```