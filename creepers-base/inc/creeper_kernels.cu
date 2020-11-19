#pragma once
#include <stdio.h>
#include "sumIntArrays_kernel.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void sumIntArrays_k(const int* addends_1, const int* addends_2, int* sums)
{
  int i = threadIdx.x;
  sums[i] = addends_1[i] + addends_2[i];
}
