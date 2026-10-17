/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// DietGPU is written against CUDA-shaped runtime names. Keep that internal
// surface stable and select the vendor runtime in one header.
#if defined(DIETGPU_USE_MUSA)

#include <musa.h>
#include <musa_runtime.h>

#define cudaError_t musaError_t
#define cudaStream_t musaStream_t
#define cudaEvent_t musaEvent_t
#define cudaMemcpyKind musaMemcpyKind
#define cudaDeviceProp musaDeviceProp
#define cudaPointerAttributes musaPointerAttributes

#define cudaSuccess musaSuccess
#define cudaErrorNoDevice musaErrorNoDevice
#define cudaErrorInvalidValue musaErrorInvalidValue
#define cudaMemoryTypeHost musaMemoryTypeHost
#define cudaMemoryTypeDevice musaMemoryTypeDevice
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaStreamDefault musaStreamDefault
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaEventDefault musaEventDefault
#define cudaEventDisableTiming musaEventDisableTiming

#define cudaGetErrorString musaGetErrorString
#define cudaGetErrorName musaGetErrorName
#define cudaGetDevice musaGetDevice
#define cudaSetDevice musaSetDevice
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaPointerGetAttributes musaPointerGetAttributes
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaGetLastError musaGetLastError
#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemsetAsync musaMemsetAsync
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStreamWaitEvent musaStreamWaitEvent
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaEventDestroy musaEventDestroy
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEventElapsedTime musaEventElapsedTime
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
  musaOccupancyMaxActiveBlocksPerMultiprocessor

// MUSA has no matching profiler API in the target contract. The wrappers are
// intentionally successful no-ops so profiling hooks do not affect codec
// correctness or link requirements.
inline musaError_t dietgpuMusaProfilerStart() { return musaSuccess; }

inline musaError_t dietgpuMusaProfilerStop() { return musaSuccess; }

#define cudaProfilerStart dietgpuMusaProfilerStart
#define cudaProfilerStop dietgpuMusaProfilerStop

#else

#include <cuda.h>
#include <cuda_runtime.h>

#endif
