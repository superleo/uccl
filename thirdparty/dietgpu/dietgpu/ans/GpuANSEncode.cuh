/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "dietgpu/ans/BatchPrefixSum.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSStatistics.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/ans/GpuChecksum.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include "util/debug.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dietgpu {
#define THRUST_DISABLE_EXEC_CHECK

// 诊断专用分段计时（VC_ENCODE_KERNEL_TIMING=1）：用 stream event 测量
// encode 链各 kernel 的 GPU 耗时。仅用于瓶颈归因，不用于正式性能采样。
inline bool encode_kernel_timing_enabled() {
  static bool const enabled = std::getenv("VC_ENCODE_KERNEL_TIMING") != nullptr;
  return enabled;
}

// Experiment knob for the number of consecutive full blocks encoded by each
// warp in ansEncodeBatchFull. Historical note: the "per-warp-slot fixed cost"
// that motivated this merging was later traced to device-side asserts compiled
// into the rANS hot path (the MUSA build was missing -DNDEBUG; fixed in
// float/Makefile.musa, see docs/musa/dietgpu-musa-test-report-20261019.md §15).
// With NDEBUG in place k=1 is optimal. VC_ENCODE_BLOCKS_PER_WARP can still
// raise the merge factor; the unset default stays at 1.
// When VC_ENCODE_BLOCKS_PER_WARP is set it overrides the merge
// factor (clamped to [1, 16]). Unset stays at 1: with -DNDEBUG that is the
// measured optimum, and the old adaptive default was fit on the assert build.
inline uint32_t encode_blocks_per_warp() {
  static uint32_t const v = [] {
    auto e = std::getenv("VC_ENCODE_BLOCKS_PER_WARP");
    if (!e) {
      return 1u;
    }
    unsigned long n = std::strtoul(e, nullptr, 10);
    return n < 1u ? 1u : (n > 16u ? 16u : (uint32_t)n);
  }();
  return v;
}

// maximum raw compressed data block size in bytes
constexpr __host__ __device__ uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
  // (an estimate from zstd)
  return roundUp(
      uncompressedBlockBytes + (uncompressedBlockBytes / 4), kBlockAlignment);
}

inline uint32_t getMaxBlockSizeUnCoalesced(uint32_t uncompressedBlockBytes) {
  // uncoalesced data has a warp state header
  return sizeof(ANSWarpState) + getRawCompBlockMaxSize(uncompressedBlockBytes);
}

inline uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
  return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

// Returns number of values written to the compressed output
// Assumes all lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOneWarp(
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {
  auto lookup = smemLookup[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = (state >= maxStateCheck);

  auto vote = __ballot_sync(kFullMask, write);
  #if defined(__HIP_PLATFORM_AMD__)
    auto prefix = __popcll(vote & getLaneMaskLt());
  #else
    auto prefix = __popc(vote & getLaneMaskLt());
  #endif
  // Some lanes wish to write out their data
  if (write) {
    out[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  constexpr uint32_t kProbBitsMul = 1 << ProbBits;

  uint32_t t = __umulhi(state, div_m1);
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  state = div * kProbBitsMul + mod + cdf;

  // how many values we actually write to the compressed output
  #if defined(__HIP_PLATFORM_AMD__)
    return __popcll(vote);
  #else
    return __popc(vote);
  #endif
}

// Returns number of values written to the compressed output
// Assumes only some lanes in the warp are presented valid input symbols
template <int ProbBits>
__device__ __forceinline__ uint32_t encodeOnePartialWarp(
    // true for the lanes in the warp for which data read is valid
    bool valid,
    ANSStateT& state,
    ANSDecodedT sym,
    uint32_t outOffset,
    ANSEncodedT* __restrict__ out,
    const uint4* __restrict__ smemLookup) {
  auto lookup = smemLookup[sym];

  uint32_t pdf = lookup.x;
  uint32_t cdf = lookup.y;
  uint32_t div_m1 = lookup.z;
  uint32_t div_shift = lookup.w;

  constexpr ANSStateT kStateCheckMul = 1 << (kANSStateBits - ProbBits);

  ANSStateT maxStateCheck = pdf * kStateCheckMul;
  bool write = valid && (state >= maxStateCheck);

  auto vote = __ballot_sync(kFullMask, write);
  #if defined(__HIP_PLATFORM_AMD__)
    auto prefix = __popcll(vote & getLaneMaskLt());
  #else
    auto prefix = __popc(vote & getLaneMaskLt());
  #endif
  // Some lanes wish to write out their data
  if (write) {
    out[outOffset + prefix] = state & kANSEncodedMask;
    state >>= kANSEncodedBits;
  }

  uint32_t t = __umulhi(state, div_m1);
  // We prevent addition overflow here by restricting `state` to < 2^31
  // (kANSStateBits)
  uint32_t div = (t + state) >> div_shift;
  auto mod = state - (div * pdf);

  // calculating ((state / pdf) << ProbBits) + (state % pdf) + cdf
  constexpr uint32_t kProbBitsMul = 1 << ProbBits;
  state = valid ? div * kProbBitsMul + mod + cdf : state;

  // how many values we actually write to the compressed output
  #if defined(__HIP_PLATFORM_AMD__)
    return __popcll(vote);
  #else
    return __popc(vote);
  #endif
}

// Fully encode a single block of data, along with the state for that block as
// the initial header.
// Returns the number of compressed words (ANSEncodedT) written
template <int ProbBits>
__device__ uint32_t ansEncodeWarpBlock(
    // Current lane ID in the warp
    uint32_t laneId,
    // Input for this block
    const ANSDecodedT* __restrict__ in,
    // Number of ANSDecodedT words in this block
    uint32_t inWords,
    // encoded table in smem
    const uint4* __restrict__ table,
    // Output for this block
    ANSWarpState* __restrict__ out) {
  // where we write the compressed words
  ANSEncodedT* outWords = (ANSEncodedT*)(out + 1);

  // Start state value for this warp
  ANSStateT state = kANSStartState;

  uint32_t inOffset = laneId;
  uint32_t outOffset = 0;

  constexpr int kUnroll = 8;

  // Unrolled iterations
  uint32_t limit = roundDown(inWords, kWarpSize * kUnroll);
  {
    ANSDecodedT sym[kUnroll];

    for (; inOffset < limit; inOffset += kWarpSize * kUnroll) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        sym[j] = in[inOffset + j * kWarpSize];
      }

#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        outOffset +=
            encodeOneWarp<ProbBits>(state, sym[j], outOffset, outWords, table);
      }
    }
  }
  
  if (limit != inWords) {
    // Remainder iterations
    limit = roundDown(inWords, kWarpSize);

    // Whole warp iterations
    for (; inOffset < limit; inOffset += kWarpSize) {
      ANSDecodedT sym = in[inOffset];

      outOffset +=
          encodeOneWarp<ProbBits>(state, sym, outOffset, outWords, table);
    }

    // Partial warp iteration
    if (limit != inWords) {
      // Last iteration may not be a full warp
      bool valid = inOffset < inWords;
      ANSDecodedT sym = valid ? in[inOffset] : ANSDecodedT(0);

      outOffset += encodeOnePartialWarp<ProbBits>(
          valid, state, sym, outOffset, outWords, table);
    }
  }

  // Write final state at the beginning (aligned addresses)
  out->warpState[laneId] = state;

  // Number of compressed words written
  return outOffset;
}

// Fully encode a single, full sized block of data, along with the state for
// that block as the initial header.
// Returns the number of compressed words (ANSEncodedT) written
// UseVec4 means that each lane in the warp loads 4 bytes of the input at a
// time, with each byte compressed by the warp in an interleaved fashion.
// Such vectorization must match with decom
template <int ProbBits, int BlockSize, bool UseVec4>
struct ANSEncodeWarpFullBlock;

// template <int ProbBits, int BlockSize>
// struct ANSEncodeWarpFullBlock<ProbBits, BlockSize, true> {
//   static __device__ uint32_t encode(
//     // Current lane ID in the warp
//     uint32_t laneId,
//     // Input for this block
//     const ANSDecodedT* __restrict__ in,
//     // encoded table in smem
//     const uint4* __restrict__ table,
//     // Output for this block
//     ANSWarpState* __restrict__ out) {
//     // where we write the compressed words
//     ANSEncodedT* outWords = (ANSEncodedT*)(out + 1);

//     // Start state value for this warp
//     ANSStateT state = kANSStartState;

//     uint32_t outOffset = 0;

//     using VecT = uint32_t;

//     auto inV = (const VecT*)in;
//     inV += laneId;

//     // kUnroll 4, unroll 2 164.93 us
//     // kUnroll 8, unroll 0 161.86 us
//     constexpr int kUnroll = 16;

//     static_assert(
//       isEvenDivisor((int)BlockSize, (int)(kUnroll * kWarpSize *
//       sizeof(VecT))),
//       "");

//     for (int i = 0; i < BlockSize / (kWarpSize * sizeof(VecT));
//          i += kUnroll, inV += kUnroll * kWarpSize) {
//       VecT symV[kUnroll];

// #pragma unroll
//       for (int j = 0; j < kUnroll; ++j) {
//         symV[j] = inV[j * kWarpSize];
//       }

// #pragma unroll
//       for (int j = 0; j < kUnroll; ++j) {
//         asm volatile("prefetch.global.L2 [%0];" : : "l"(inV + 128 + j * 32));
// #pragma unroll
//         for (int k = 0; k < 4; ++k) {
//           outOffset += encodeOneWarp<ProbBits>(
//             state, symV[j] & 0xff, outOffset, outWords, table);

//           symV[j] >>= 8;
//         }
//       }
//     }

//     // Write final state at the beginning (aligned addresses)
//     out->warpState[laneId] = state;

//     // Number of compressed words written
//     return outOffset;
//   }
// };

template <int ProbBits, int BlockSize>
struct ANSEncodeWarpFullBlock<ProbBits, BlockSize, false> {
  static __device__ uint32_t encode(
      // Current lane ID in the warp
      uint32_t laneId,
      // Input for this block
      const ANSDecodedT* __restrict__ in,
      // encoded table in smem
      const uint4* __restrict__ table,
      // Output for this block
      ANSWarpState* __restrict__ out) {
    // Just use the normal implementation
    return ansEncodeWarpBlock<ProbBits>(laneId, in, BlockSize, table, out);
  }
};

template <int ProbBits, int BlockSize>
__device__ void ansEncodeBlocksFull(
    // input data for all blocks
    const ANSDecodedT* __restrict__ in,
    // length in ANSDecodedT words
    uint32_t uncompressedWords,
    // number of blocks that different warps will process
    uint32_t numBlocks,
    // the stride of each encoded output block
    uint32_t outBlockStride,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    const uint4* __restrict__ table,
    // how many consecutive blocks each warp encodes
    uint32_t blocksPerWarp) {
  // grid-wide warp id
  auto tid = threadIdx.x;
  // so we know the block is warp uniform
  int warpId =
      __shfl_sync(kFullMask, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table[tid];
  }

  __syncthreads();

  // Each warp encodes up to `blocksPerWarp` consecutive blocks; see
  // encode_blocks_per_warp() above for the history of the per-warp-slot cost
  // this merging addressed (device-side asserts, fixed via -DNDEBUG).
  uint32_t block = (uint32_t)warpId * blocksPerWarp;

  // Either the warp is an excess one, or all of its blocks are beyond the end
  if (block >= numBlocks) {
    return;
  }

  uint32_t endBlock = min(block + blocksPerWarp, numBlocks);
  for (; block < endBlock; ++block) {
    // The (at most one) partial tail block is handled by the partial kernel
    if ((block + 1) * BlockSize > uncompressedWords) {
      return;
    }

    uint32_t start = block * BlockSize;
    auto inBlock = in + start;
    auto outBlock = (ANSWarpState*)(out + block * outBlockStride);

    // all input blocks must meet alignment requirements
    assert(isPointerAligned(inBlock, kANSRequiredAlignment));

    auto outWords = ANSEncodeWarpFullBlock<ProbBits, BlockSize, false>::encode(
        laneId, inBlock, smemLookup, outBlock);

    if (laneId == 0) {
      // If the bound on max compressed size is not correct, this assert will
      // go off. This block of data was then somewhat adversarial in terms of
      // incompressibility. In this case, the getRawCompBlockMaxSize max
      // estimate needs to increase.
      assert(outWords <= getRawCompBlockMaxSize(BlockSize) / sizeof(ANSEncodedT));
      compressedWords[block] = outWords;
    }
  }
}

template <int ProbBits, int BlockSize>
__device__ void ansEncodeBlocksPartial(
    // input data for all blocks
    const ANSDecodedT* __restrict__ in,
    // length in ANSDecodedT words
    uint32_t uncompressedWords,
    // number of blocks that different warps will process
    uint32_t numBlocks,
    // the stride of each encoded output block
    uint32_t outBlockStride,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    const uint4* __restrict__ table) {
  int block = numBlocks - 1;
  uint32_t tid = threadIdx.x;
  int laneId = getLaneId();

  __shared__ uint4 smemLookup[kNumSymbols];

  // we always have at least 256 threads
  if (tid < kNumSymbols) {
    smemLookup[tid] = table[tid];
  }

  __syncthreads();

  // We only have a single partial block to handle
  if (tid >= kWarpSize) {
    return;
  }

  // How big is this block?
  uint32_t start = block * BlockSize;
  uint32_t end = min(start + BlockSize, uncompressedWords);

  auto blockSize = end - start;

  // If the end block is a full block, it would have been handled by the full
  // block kernel
  if (blockSize == BlockSize) {
    return;
  }

  auto inBlock = in + start;
  auto outBlock = (ANSWarpState*)(out + block * outBlockStride);

  // all input blocks must meet required alignment
  assert(isPointerAligned(inBlock, kANSRequiredAlignment));

  auto outWords = ansEncodeWarpBlock<ProbBits>(
      laneId, inBlock, blockSize, smemLookup, outBlock);

  if (laneId == 0) {
    // If the bound on max compressed size is not correct, this assert will go
    // off. This block of data was then somewhat adversarial in terms of
    // incompressibility. In this case, the getRawCompBlockMaxSize max estimate
    // needs to increase.
    assert(outWords <= getRawCompBlockMaxSize(BlockSize) / sizeof(ANSEncodedT));
    compressedWords[block] = outWords;
  }
}

template <typename InProvider, int ProbBits, int BlockSize>
__global__ void ansEncodeBatchFull(
    // Input data for all blocks
    InProvider inProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // maximum size of a compressed block
    uint32_t maxCompressedBlockSize,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    // per batch
    // [batch][numBlocks]
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    // [batch][kNumSymbols]
    const uint4* __restrict__ table,
    // how many consecutive blocks each warp encodes
    uint32_t blocksPerWarp) {
  // which batch element we are processing
  auto batch = blockIdx.y;

  // Number of blocks for the current problem
  uint32_t curSize = inProvider.getBatchSize(batch);
  uint32_t numBlocks = divUp(curSize, BlockSize);

  ansEncodeBlocksFull<ProbBits, BlockSize>(
      (const ANSDecodedT*)inProvider.getBatchStart(batch),
      curSize,
      numBlocks,
      maxCompressedBlockSize,
      out + batch * maxNumCompressedBlocks * maxCompressedBlockSize,
      compressedWords + batch * maxNumCompressedBlocks,
      table + batch * kNumSymbols,
      blocksPerWarp);
}

template <typename InProvider, int ProbBits, int BlockSize>
__global__ void ansEncodeBatchPartial(
    // input data for all blocks
    InProvider inProvider,
    // maximum number of blocks across all the batch
    uint32_t maxNumCompressedBlocks,
    // maximum size of a compressed block
    uint32_t maxCompressedBlockSize,
    // address of the output for all blocks
    uint8_t* __restrict__ out,
    // output array of per-block sizes of number of ANSEncodedT words per block
    // per batch
    // [batch][numBlocks]
    uint32_t* __restrict__ compressedWords,
    // the encoding table that we will load into smem
    // [batch][kNumSymbols]
    const uint4* __restrict__ table) {
  // which batch element we are processing
  auto batch = blockIdx.y;

  // Number of blocks for the current problem
  uint32_t curSize = inProvider.getBatchSize(batch);
  uint32_t numBlocks = divUp(curSize, BlockSize);

  ansEncodeBlocksPartial<ProbBits, BlockSize>(
      (const ANSDecodedT*)inProvider.getBatchStart(batch),
      inProvider.getBatchSize(batch),
      numBlocks,
      maxCompressedBlockSize,
      out + batch * maxNumCompressedBlocks * maxCompressedBlockSize,
      compressedWords + batch * maxNumCompressedBlocks,
      table + batch * kNumSymbols);
}

template <typename A, int B>
struct Align {
  typedef uint32_t argument_type;
  typedef uint32_t result_type;

  THRUST_DISABLE_EXEC_CHECK template <typename T>
  __host__ __device__ uint32_t operator()(T x) const {
    constexpr int kDiv = B / sizeof(A);
    constexpr int kSize = kDiv < 1 ? 1 : kDiv;

    return roundUp(x, T(kSize));
  }
};

template <int Threads>
__device__ void ansEncodeCoalesce(
    const uint8_t* __restrict__ inUncoalescedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords,
    const uint32_t* __restrict__ compressedWordsPrefix,
    const uint32_t* __restrict__ checksum,
    const uint4* __restrict__ table,
    uint32_t probBits,
    bool useChecksum,
    uint32_t numBlocks,
    uint32_t uncompressedWords,
    uint32_t blocksPerCTA,
    uint8_t* __restrict__ out,
    uint32_t* __restrict__ compressedBytes) {
  auto tid = threadIdx.x;
  // Each CTA coalesces up to `blocksPerCTA` consecutive compressed blocks;
  // see the coalesce launch site in ansEncodeBatchDevice for the history of
  // the per-CTA cost model behind this merging.
  auto block0 = blockIdx.x * blocksPerCTA;

  ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)out;

  // The first block will be responsible for the coalesced header
  if (blockIdx.x == 0) {
    if (tid == 0) {
      uint32_t totalCompressedWords = 0;

      // Could be a header for a zero sized array
      if (numBlocks > 0) {
        totalCompressedWords =
            // total number of compressed words in all blocks
            // this is already a multiple of kBlockAlignment /
            // sizeof(ANSEncodedT)
            compressedWordsPrefix[numBlocks - 1] +
            // this is not yet a multiple of kBlockAlignment /
            // sizeof(ANSEncodedT), but needs to be
            roundUp(
                compressedWords[numBlocks - 1],
                kBlockAlignment / sizeof(ANSEncodedT));
      }

      ANSCoalescedHeader header;
      header.setMagicAndVersion();
      header.setNumBlocks(numBlocks);
      header.setTotalUncompressedWords(uncompressedWords);
      header.setTotalCompressedWords(totalCompressedWords);
      header.setProbBits(probBits);
      header.setUseChecksum(useChecksum);

      if (useChecksum) {
        header.setChecksum(*checksum);
      }

      if (compressedBytes) {
        *compressedBytes = header.getTotalCompressedSize();
      }

      *headerOut = header;
    }

    auto probsOut = headerOut->getSymbolProbs();

    // Write out pdf
    for (int i = tid; i < kNumSymbols; i += Threads) {
      probsOut[i] = table[i].x;
    }
  }

  if (block0 >= numBlocks) {
    return;
  }

  uint32_t blockEnd = block0 + blocksPerCTA;
  if (blockEnd > numBlocks) {
    blockEnd = numBlocks;
  }

  // Write per-block warp state (one warp state per compressed block)
  for (auto i = tid; i < (blockEnd - block0) * kWarpSize; i += Threads) {
    auto b = block0 + i / kWarpSize;
    auto warpStateIn =
        (ANSWarpState*)(inUncoalescedBlocks + b * uncoalescedBlockStride);

    headerOut->getWarpStates()[b].warpState[i % kWarpSize] =
        warpStateIn->warpState[i % kWarpSize];
  }

  auto blockWordsOut = headerOut->getBlockWords(numBlocks);

  // Write out per-block word length
  for (auto i = tid; i < blockEnd - block0; i += Threads) {
    auto b = block0 + i;
    uint32_t lastBlockWords = uncompressedWords % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    uint32_t blockWords =
        (b == numBlocks - 1) ? lastBlockWords : kDefaultBlockSize;

    blockWordsOut[b] = uint2{
        (blockWords << 16) | compressedWords[b], compressedWordsPrefix[b]};
  }

  // We always have a valid multiple of kBlockAlignment bytes on both
  // uncoalesced src and coalesced dest, even though numWords (actual encoded
  // words) may be less than that
  using LoadT = uint4;
  static_assert(sizeof(LoadT) == kBlockAlignment, "");

  for (auto b = block0; b < blockEnd; ++b) {
    // Number of compressed words in this block
    uint32_t numWords = compressedWords[b];

    uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

    auto inT = (const LoadT*)(inUncoalescedBlocks +
                              b * uncoalescedBlockStride +
                              sizeof(ANSWarpState));
    auto outT = (LoadT*)(headerOut->getBlockDataStart(numBlocks) +
                         compressedWordsPrefix[b]);

    for (uint32_t i = tid; i < limitEnd; i += Threads) {
      outT[i] = inT[i];
    }
  }
}

template <typename SizeProvider, typename OutProvider, int Threads>
__global__ void ansEncodeCoalesceBatch(
    const uint8_t* __restrict__ inUncoalescedBlocks,
    SizeProvider sizeProvider,
    uint32_t maxNumCompressedBlocks,
    uint32_t uncoalescedBlockStride,
    const uint32_t* __restrict__ compressedWords,
    const uint32_t* __restrict__ compressedWordsPrefix,
    const uint32_t* __restrict__ checksum,
    const uint4* __restrict__ table,
    uint32_t probBits,
    bool useChecksum,
    uint32_t blocksPerCTA,
    OutProvider outProvider,
    uint32_t* __restrict__ compressedBytes) {
  auto batch = blockIdx.y;
  auto uncompressedWords = sizeProvider.getBatchSize(batch);

  // Number of compressed blocks in this batch element
  auto numBlocks = divUp(uncompressedWords, kDefaultBlockSize);

  // Advance all pointers to handle our specific batch member
  inUncoalescedBlocks +=
      batch * uncoalescedBlockStride * maxNumCompressedBlocks;
  compressedWords += batch * maxNumCompressedBlocks;
  compressedWordsPrefix += batch * maxNumCompressedBlocks;
  compressedBytes += batch;
  checksum += batch;
  table += batch * kNumSymbols;

  ansEncodeCoalesce<Threads>(
      inUncoalescedBlocks,
      uncoalescedBlockStride,
      compressedWords,
      compressedWordsPrefix,
      checksum,
      table,
      probBits,
      useChecksum,
      numBlocks,
      uncompressedWords,
      blocksPerCTA,
      (uint8_t*)outProvider.getBatchStart(batch),
      compressedBytes);
}

template <typename InProvider, typename OutProvider>
void ansEncodeBatchDevice(
    StackDeviceMemory& res,
    const ANSCodecConfig& config,
    uint32_t numInBatch,
    InProvider inProvider,
    const uint32_t* histogram_dev,
    uint32_t maxSize,
    OutProvider outProvider,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      divUp(maxUncompressedWords, kDefaultBlockSize);

  // 诊断专用（VC_ENCODE_KERNEL_TIMING=1）：event 序列
  // [0]=链头 [1]=quantize 后 [2]=Full 后 [3]=Partial 后
  // [4]=prefix-sum 后 [5]=Coalesce 后
  bool const ekTiming = encode_kernel_timing_enabled();
  cudaEvent_t ev[6];
  if (ekTiming) {
    for (int i = 0; i < 6; ++i) {
      cudaEventCreateWithFlags(&ev[i], cudaEventDefault);
    }
    cudaEventRecord(ev[0], stream);
  }

  // 1. Compute symbol statistics
  auto table_dev = res.alloc<uint4>(stream, numInBatch * kNumSymbols);

  if (histogram_dev) {
    // use pre-calculated histogram
    ansCalcWeights(
        numInBatch,
        config.probBits,
        inProvider,
        histogram_dev,
        table_dev.data(),
        stream);
  } else {
    auto tempHistogram_dev =
        res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

    // need to calculate a histogram
    ansHistogramBatch(numInBatch, inProvider, tempHistogram_dev.data(), stream);

    ansCalcWeights(
        numInBatch,
        config.probBits,
        inProvider,
        tempHistogram_dev.data(),
        table_dev.data(),
        stream);
  }

  if (ekTiming) {
    cudaEventRecord(ev[1], stream);
  }

  // 2. Compute checksum on input data (optional)
  auto checksum_dev = res.alloc<uint32_t>(stream, numInBatch);
  if (config.useChecksum) {
    checksumBatch(numInBatch, inProvider, checksum_dev.data(), stream);
  }

  // 3. Allocate memory for the per-warp results
  // How much space in bytes we need to reserve for each warp's output
  uint32_t uncoalescedBlockStride =
      getMaxBlockSizeUnCoalesced(kDefaultBlockSize);

  auto compressedBlocks_dev = res.alloc<uint8_t>(
      stream, numInBatch * maxNumCompressedBlocks * uncoalescedBlockStride);

  // +1 in order to get the final sum as well
  auto compressedWords_dev =
      res.alloc<uint32_t>(stream, numInBatch * maxNumCompressedBlocks);

  // Exclusive prefix sum of the compressed sizes (so we know where to write in
  // the contiguous output). The offsets are aligned to a multple of 4
  auto compressedWordsPrefix_dev =
      res.alloc<uint32_t>(stream, numInBatch * maxNumCompressedBlocks);

  // Run per-warp encoding
  // (only if we have blocks to compress)
  if (maxNumCompressedBlocks > 0) {
    constexpr int kThreads = 256;

    // One block per warp unless VC_ENCODE_BLOCKS_PER_WARP overrides it.
    // Merging warps was a win only while device asserts were still in the
    // rANS path; -DNDEBUG made k=1 the faster default.
    uint32_t blocksPerWarp = encode_blocks_per_warp();

    // The grid for the full block kernel
    auto gridFull = dim3(
        divUp(
            (int)maxNumCompressedBlocks,
            (kThreads / kWarpSize) * (int)blocksPerWarp),
        numInBatch);

    // The grid for the partial block kernel; at most 1 partial block per input
    // in the batch
    auto gridPartial = dim3(1, numInBatch);

#define RUN_ENCODE(BITS)                                       \
  do {                                                         \
    ansEncodeBatchFull<InProvider, BITS, kDefaultBlockSize>    \
        <<<gridFull, kThreads, 0, stream>>>(                   \
            inProvider,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_dev.data(),                       \
            compressedWords_dev.data(),                       \
            table_dev.data(),                                  \
            blocksPerWarp);                                    \
    if (ekTiming) {                                            \
      cudaEventRecord(ev[2], stream);                          \
    }                                                          \
    ansEncodeBatchPartial<InProvider, BITS, kDefaultBlockSize> \
        <<<gridPartial, kThreads, 0, stream>>>(                \
            inProvider,                                        \
            maxNumCompressedBlocks,                            \
            uncoalescedBlockStride,                            \
            compressedBlocks_dev.data(),                       \
            compressedWords_dev.data(),                        \
            table_dev.data());                                 \
    if (ekTiming) {                                            \
      cudaEventRecord(ev[3], stream);                          \
    }                                                          \
  } while (false)

    switch (config.probBits) {
      case 9:
        RUN_ENCODE(9);
        break;
      case 10:
        RUN_ENCODE(10);
        break;
      case 11:
        RUN_ENCODE(11);
        break;
      default:
        UCCL_CHECK(false) << "unhandled pdf precision " << config.probBits;
    }

#undef RUN_ENCODE
  }

  // Perform exclusive prefix sum of the number of compressed words per block,
  // so we know where to write the output. We align the blocks so that we can
  // write state values at 4 byte alignment at the beginning.
  // FIXME: probably some way to do this via thrust::exclusive_scan_by_key with
  // transform iterators and what not
  if (maxNumCompressedBlocks > 0) {
    auto sizeRequired =
        getBatchExclusivePrefixSumTempSize(numInBatch, maxNumCompressedBlocks);

    // FIXME: we can run a more minimal segmented prefix sum instead of using
    // maxNumCompressedBlocks
    if (sizeRequired == 0) {
      batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
          compressedWords_dev.data(),
          compressedWordsPrefix_dev.data(),
          nullptr,
          numInBatch,
          maxNumCompressedBlocks,
          Align<ANSEncodedT, kBlockAlignment>(),
          stream);
    } else {
      auto tempPrefixSum_dev = res.alloc<uint8_t>(stream, sizeRequired);

      batchExclusivePrefixSum<uint32_t, Align<ANSEncodedT, kBlockAlignment>>(
          compressedWords_dev.data(),
          compressedWordsPrefix_dev.data(),
          tempPrefixSum_dev.data(),
          numInBatch,
          maxNumCompressedBlocks,
          Align<ANSEncodedT, kBlockAlignment>(),
          stream);
    }
  }

  if (ekTiming) {
    cudaEventRecord(ev[4], stream);
  }

  // Coalesce the data into one contiguous buffer
  // Even if there is nothing to compress, we still need to create a compression
  // header
  {
    // 64-thread CTAs, one compressed block each. Merging several blocks
    // into one CTA was measured while device asserts were still compiled
    // in; -DNDEBUG removed that per-slot cost, so the default stays at 1.
    constexpr int kThreads = 64;
    uint32_t blocksPerCTA = 1;

    auto grid = dim3(
        std::max(divUp(maxNumCompressedBlocks, blocksPerCTA), 1U), numInBatch);

    ansEncodeCoalesceBatch<InProvider, OutProvider, kThreads>
        <<<grid, kThreads, 0, stream>>>(
            compressedBlocks_dev.data(),
            inProvider,
            maxNumCompressedBlocks,
            uncoalescedBlockStride,
            compressedWords_dev.data(),
            compressedWordsPrefix_dev.data(),
            checksum_dev.data(),
            table_dev.data(),
            config.probBits,
            config.useChecksum,
            blocksPerCTA,
            outProvider,
            outSize_dev);
  }

  if (ekTiming) {
    cudaEventRecord(ev[5], stream);
    cudaEventSynchronize(ev[5]);
    float msQuantize = 0, msFull = 0, msPartial = 0, msPs = 0, msCoalesce = 0;
    cudaEventElapsedTime(&msQuantize, ev[0], ev[1]);
    if (maxNumCompressedBlocks > 0) {
      cudaEventElapsedTime(&msFull, ev[1], ev[2]);
      cudaEventElapsedTime(&msPartial, ev[2], ev[3]);
      cudaEventElapsedTime(&msPs, ev[3], ev[4]);
    }
    cudaEventElapsedTime(&msCoalesce, ev[4], ev[5]);
    fprintf(stderr,
            "[ek] numInBatch=%u blocks=%u quantize=%.3f full=%.3f "
            "partial=%.3f ps=%.3f coalesce=%.3f ms\n",
            numInBatch, maxNumCompressedBlocks, msQuantize, msFull, msPartial,
            msPs, msCoalesce);
    for (int i = 0; i < 6; ++i) {
      cudaEventDestroy(ev[i]);
    }
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu

#undef RUN_ENCODE_ALL
