/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dietgpu/utils/GpuRuntime.h"
#include <cstdint>
#if defined(DIETGPU_USE_MUSA)
#include "dietgpu/utils/MusaWarp.muh"
#endif

namespace dietgpu {

#if defined(__HIP_PLATFORM_AMD__)
using WarpMaskT = uint64_t;
inline constexpr WarpMaskT kFullMask = 0xffffffffffffffffull;
#else
using WarpMaskT = uint32_t;
inline constexpr WarpMaskT kFullMask = 0xffffffffu;
#endif

__device__ __forceinline__ WarpMaskT fullMask() {
#if defined(__HIP_PLATFORM_AMD__)
  return 0xffffffffffffffffull;
#else
  return 0xffffffffu;
#endif
}


__device__ __forceinline__ unsigned int
getBitfield(uint8_t val, int pos, int len) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return ((uint32_t)val >> pos) & ((1u << len) - 1u);
#else
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
#endif
}

__device__ __forceinline__ unsigned int
getBitfield(uint16_t val, int pos, int len) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return ((uint32_t)val >> pos) & ((1u << len) - 1u);
#else
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;"
      : "=r"(ret)
      : "r"((uint32_t)val), "r"(pos), "r"(len));
  return ret;
#endif
}

__device__ __forceinline__ unsigned int
getBitfield(unsigned int val, int pos, int len) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return (val >> pos) & ((1u << len) - 1u);
#else
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
#endif
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return (val >> pos) & ((1ull << len) - 1ull);
#else
  uint64_t ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
#endif
}

__device__ __forceinline__ unsigned int
setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  unsigned int mask = ((1u << len) - 1u) << pos;
  return (val & ~mask) | ((toInsert << pos) & mask);
#else
  unsigned int ret;
  asm("bfi.b32 %0, %1, %2, %3, %4;"
      : "=r"(ret)
      : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
  return ret;
#endif
}

__device__ __forceinline__ uint32_t rotateLeft(uint32_t v, uint32_t shift) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return (v << shift) | (v >> (32u - shift));
#else
  uint32_t out;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
#endif
}

__device__ __forceinline__ uint32_t rotateRight(uint32_t v, uint32_t shift) {
#if defined(DIETGPU_USE_MUSA) || defined(__HIP_PLATFORM_AMD__)
  return (v >> shift) | (v << (32u - shift));
#else
  uint32_t out;
  asm("shf.r.clamp.b32 %0, %1, %2, %3;"
      : "=r"(out)
      : "r"(v), "r"(v), "r"(shift));
  return out;
#endif
}

__device__ __forceinline__ int getLaneId() {
#if defined(DIETGPU_USE_MUSA)
  return __get_laneid();
#elif defined(__HIP_PLATFORM_AMD__)
  return __lane_id();
#else
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
#endif
}

__device__ __forceinline__ WarpMaskT getLaneMaskLt() {
#if defined(__HIP_PLATFORM_AMD__)
  int lane = __lane_id(); // 0..63
  return (lane == 0) ? 0ull : ((1ull << lane) - 1ull);
#elif defined(DIETGPU_USE_MUSA)
  int lane = __get_laneid();
  return (lane == 0) ? 0u : ((1u << lane) - 1u);
#else
  WarpMaskT mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
#endif
}

__device__ __forceinline__ WarpMaskT getLaneMaskLe() {
#if defined(__HIP_PLATFORM_AMD__)
  int lane = __lane_id();
  return (lane == 63) ? 0xffffffffffffffffull : ((1ull << (lane + 1)) - 1ull);
#elif defined(DIETGPU_USE_MUSA)
  int lane = __get_laneid();
  return (lane == 31) ? 0xffffffffu : ((1u << (lane + 1)) - 1u);
#else
  WarpMaskT mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
#endif
}

__device__ __forceinline__ WarpMaskT getLaneMaskGt() {
#if defined(__HIP_PLATFORM_AMD__)
  int lane = __lane_id();
  return ~getLaneMaskLe();
#elif defined(DIETGPU_USE_MUSA)
  int lane = __get_laneid();
  return lane == 31 ? 0u : ~getLaneMaskLe();
#else
  WarpMaskT mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
#endif
}

__device__ __forceinline__ WarpMaskT getLaneMaskGe() {
#if defined(__HIP_PLATFORM_AMD__)
  int lane = __lane_id();
  return ~getLaneMaskLt();
#elif defined(DIETGPU_USE_MUSA)
  return ~getLaneMaskLt();
#else
  WarpMaskT mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
#endif
}


// MUSA 专属归约实现在 .muh；其他后端保持原有实现。
template <typename T>
__device__ inline T warpReduceAllMin(T val) {
#if defined(DIETGPU_USE_MUSA)
  return musa_backend::warpReduceMin(val);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_min_sync(kFullMask, val);
#else
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val = min(val, __shfl_xor_sync(kFullMask, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllMax(T val) {
#if defined(DIETGPU_USE_MUSA)
  return musa_backend::warpReduceMax<T, Width>(val);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_max_sync(kFullMask, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(kFullMask, val, mask, kWarpSize));
  }

  return val;
#endif
}

template <typename T, int Width = kWarpSize>
__device__ inline T warpReduceAllSum(T val) {
#if defined(DIETGPU_USE_MUSA)
  return musa_backend::warpReduceSum<T, Width>(val);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(kFullMask, val);
#else
#pragma unroll
  for (int mask = Width / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(kFullMask, val, mask, kWarpSize);
  }

  return val;
#endif
}

} // namespace dietgpu
