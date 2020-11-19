#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

// Take two arrays of ints, add corresponding addends and place sum in.. sums array.
__global__ void sumIntArrays_k(const int* addends_1, const int* addends_2, int* sums);

