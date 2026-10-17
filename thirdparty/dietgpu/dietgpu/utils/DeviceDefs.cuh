/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dietgpu/utils/GpuRuntime.h"

namespace dietgpu {
#if defined(DIETGPU_USE_MUSA)
// MUSA S5000 exposes a CUDA-compatible warp width of 32.
constexpr int kWarpSize = 32;
#elif defined(__HIP_PLATFORM_AMD__)
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif

} // namespace dietgpu
