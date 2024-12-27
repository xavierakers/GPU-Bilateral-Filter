# GPU Accelerated Bilateral Filter for Video Enhancement

## Optimizations
1. ### CUDA Streams
    - Concurrent Execution: Load in next frame while current frame is being processed
2. ### Shared Memory
    - Reduced latency: Load data into shared memory to reduce global memory accesses 
3. ### Constant Cache
    - Reduced latency: Load spatial kernel into constant cache
4. ### Bit-Packing
    - Pack RGBA 8-bit color channels into 32-bit integers for efficient memory accesses
