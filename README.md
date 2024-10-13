# CUDA K-Means Project

This repository contains an implementation of the k-means clustering algorithm using both CPU and GPU (CUDA). The project focuses on performance comparison between the two implementations by varying the number of points, centroids, and iterations.

## Files Overview

### 1. `KMeansUtil.cu`
This file contains the core utilities for the k-means algorithm:
- **Structs**: Definitions for points and centroids.
- **CPU Implementation**: A standard sequential k-means algorithm.
- **GPU Implementation**: The CUDA parallelized version of the k-means algorithm.
  
### 2. `StastManager.cu`
This file handles the collection and calculation of performance statistics:
- **Time measurement**: For tracking execution time for both CPU and GPU runs.
- **Speedup calculation**: Computes the speedup obtained by running the algorithm on the GPU.

### 3. `main.cu`
This is the main entry point of the program. It:
- **Configures the number of points, centroids, and iterations** for the algorithm.
- **Runs both the CPU and GPU implementations** of k-means.
- **Collects performance statistics** for comparison.
