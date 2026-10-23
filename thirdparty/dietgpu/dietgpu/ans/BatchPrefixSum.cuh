/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dietgpu/utils/GpuRuntime.h"
#include <cub/cub.cuh>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

// 诊断专用（VC_ENCODE_KERNEL_TIMING=1）：prefix-sum 各级 kernel 分段计时。
inline bool encode_ps_timing_enabled() {
  static bool const enabled = std::getenv("VC_ENCODE_KERNEL_TIMING") != nullptr;
  return enabled;
}

// FIXME: at some point, batchExclusivePrefixSum1 can no longer be run with
// 1024 threads. Restrict our max threads to 512
constexpr int kMaxBEPSThreads = 512;

//
// Quick and dirty means of performing a batched exclusive prefix sum in two
// passes :(
//

template <typename T>
struct NoTransform {
  __host__ __device__ __forceinline__ T operator()(const T& v) const {
    return v;
  }
};

template <typename T, int Threads, typename TransformFn>
__global__ void batchExclusivePrefixSum1(
    const T* __restrict__ in,
    T* __restrict__ out,
    void* __restrict__ blockTotal,
    uint32_t batchSize,
    TransformFn fn) {
  uint32_t batch = blockIdx.y;
  uint32_t block = blockIdx.x;
  uint32_t blocksInBatch = gridDim.x;
  uint32_t tid = threadIdx.x;

  int batchIdx = block * Threads + tid;
  bool valid = batchIdx < batchSize;

  int totalIdx = batch * batchSize + batchIdx;
  auto v = valid ? fn(in[totalIdx]) : T(0);

  using Scan = cub::BlockScan<T, Threads>;
  __shared__ typename Scan::TempStorage smem;

  T prefix = 0;
  T total = 0;
  Scan(smem).ExclusiveSum(v, prefix, total);

  if (valid) {
    out[totalIdx] = prefix;
  }

  // Only if this is not provided is 1 level of the tree enough
  if (threadIdx.x == 0 && blockTotal) {
    ((T*)blockTotal)[batch * blocksInBatch + block] = total;
  }
}

// Single block that performs the cross-block prefix sum
template <typename T, int Threads>
__global__ void batchExclusivePrefixSum2(
    void* __restrict__ blockTotal,
    uint32_t batchSize,
    uint32_t blocksInBatch) {
  uint32_t batch = blockIdx.x;
  uint32_t tid = threadIdx.x;

  bool valid = tid < blocksInBatch;
  auto v = valid ? ((T*)blockTotal)[batch * blocksInBatch + tid] : 0;

  using Scan = cub::BlockScan<T, Threads>;
  __shared__ typename Scan::TempStorage smem;

  Scan(smem).ExclusiveSum(v, v);

  if (valid) {
    ((T*)blockTotal)[batch * blocksInBatch + tid] = v;
  }
}

template <typename T, int Threads>
__global__ void batchExclusivePrefixSum3(
    T* __restrict__ out,
    const void* __restrict__ blockTotal,
    uint32_t batchSize) {
  uint32_t batch = blockIdx.y;
  uint32_t block = blockIdx.x;
  uint32_t blocksInBatch = gridDim.x;
  uint32_t tid = threadIdx.x;

  auto vBlock = ((const T*)blockTotal)[batch * blocksInBatch + block];

  int batchIdx = block * Threads + tid;
  bool valid = batchIdx < batchSize;

  int totalIdx = batch * batchSize + batchIdx;

  if (valid) {
    out[totalIdx] += vBlock;
  }
}

inline size_t getBatchExclusivePrefixSumTempSize(
    uint32_t numInBatch,
    uint32_t batchSize) {
  if (batchSize <= kMaxBEPSThreads) {
    return 0;
  } else {
    // number of blocks required
    return divUp(batchSize, kMaxBEPSThreads);
  }
}

// Perform a batched exclusive prefix sum, comprising
// numInBatch x batchSize data
template <typename T, typename TransformFn>
void batchExclusivePrefixSum(
    const T* in_dev,
    T* out_dev,
    void* temp_dev,
    uint32_t numInBatch,
    uint32_t batchSize,
    const TransformFn& fn,
    cudaStream_t stream) {
  // maximum size we can handle with a two-level reduction
  assert(batchSize <= kMaxBEPSThreads * kMaxBEPSThreads);

  // 诊断专用（VC_ENCODE_KERNEL_TIMING=1）：ev[0]=链头，之后每级 kernel 结束
  // 记录一个 event；evLast 为最后一个已记录的下标。
  bool const ekTiming = encode_ps_timing_enabled();
  cudaEvent_t ev[4];
  int evLast = 0;
  if (ekTiming) {
    for (int i = 0; i < 4; ++i) {
      cudaEventCreateWithFlags(&ev[i], cudaEventDefault);
    }
    cudaEventRecord(ev[0], stream);
    evLast = 0;
  }

#define BPS_LEVEL_1(THREADS, TEMP)                        \
  batchExclusivePrefixSum1<T, THREADS, TransformFn>       \
      <<<dim3(blocks, numInBatch), THREADS, 0, stream>>>( \
          in_dev, out_dev, TEMP, batchSize, fn)

#define BPS_LEVEL_2(THREADS)           \
  batchExclusivePrefixSum2<T, THREADS> \
      <<<numInBatch, THREADS, 0, stream>>>(temp_dev, batchSize, blocks)

#define BPS_LEVEL_3(THREADS)                              \
  batchExclusivePrefixSum3<T, THREADS>                    \
      <<<dim3(blocks, numInBatch), THREADS, 0, stream>>>( \
          out_dev, temp_dev, batchSize)

  if (batchSize > kMaxBEPSThreads) {
    // multi-level reduction required
    uint32_t blocks = divUp(batchSize, kMaxBEPSThreads);
    assert(blocks > 1);
    assert(temp_dev); // must have this allocated

    BPS_LEVEL_1(kMaxBEPSThreads, temp_dev);
    if (ekTiming) {
      cudaEventRecord(ev[1], stream);
    }

    if (blocks <= 32) {
      BPS_LEVEL_2(32);
    } else if (blocks <= 64) {
      BPS_LEVEL_2(64);
    } else if (blocks <= 128) {
      BPS_LEVEL_2(128);
    } else if (blocks <= 256) {
      BPS_LEVEL_2(256);
    } else {
      assert(blocks <= kMaxBEPSThreads);
      BPS_LEVEL_2(kMaxBEPSThreads);
    }
    if (ekTiming) {
      cudaEventRecord(ev[2], stream);
    }

    BPS_LEVEL_3(kMaxBEPSThreads);
    if (ekTiming) {
      cudaEventRecord(ev[3], stream);
      evLast = 3;
    }
  } else {
    // single-level reduction
    uint32_t blocks = 1;

    if (batchSize <= 32) {
      BPS_LEVEL_1(32, (T*)nullptr);
    } else if (batchSize <= 64) {
      BPS_LEVEL_1(64, (T*)nullptr);
    } else if (batchSize <= 128) {
      BPS_LEVEL_1(128, (T*)nullptr);
    } else if (batchSize <= 256) {
      BPS_LEVEL_1(256, (T*)nullptr);
    } else {
      assert(batchSize <= kMaxBEPSThreads);
      BPS_LEVEL_1(kMaxBEPSThreads, (T*)nullptr);
    }
    if (ekTiming) {
      cudaEventRecord(ev[1], stream);
      evLast = 1;
    }
  }

  if (ekTiming) {
    cudaEventSynchronize(ev[evLast]);
    float ms1 = 0, ms2 = 0, ms3 = 0;
    cudaEventElapsedTime(&ms1, ev[0], ev[1]);
    if (evLast >= 3) {
      cudaEventElapsedTime(&ms2, ev[1], ev[2]);
      cudaEventElapsedTime(&ms3, ev[2], ev[3]);
    }
    fprintf(stderr,
            "[ek-ps] numInBatch=%u batchSize=%u levels=%d ps1=%.3f "
            "ps2=%.3f ps3=%.3f ms\n",
            numInBatch, batchSize, evLast, ms1, ms2, ms3);
    for (int i = 0; i < 4; ++i) {
      cudaEventDestroy(ev[i]);
    }
  }

#undef BPS_LEVEL_3
#undef BPS_LEVEL_2
#undef BPS_LEVEL_1
}

} // namespace dietgpu
