#pragma once
#include "common.h"
#include "memory_allocator.h"
#include "util/gpu_rt.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

size_t get_min_compress_bytes_from_env();
extern size_t const& kMinCompressBytes;

size_t get_compress_buffer_bytes_from_env();
extern size_t const& kCompressBufferSize;

enum class CompressStrategy {
  kNone,         // no compression
  kSplitOnly,    // only split, no encode
  kSplitEncode,  // split + encode (default)
};

CompressStrategy get_compress_strategy_from_env();

#if defined USE_DIETGPU
#if defined(DIETGPU_USE_MUSA)
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/GreenContext.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include <musa_fp16.h>
#elif defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || \
    defined(__HIPCC__)
#include "dietgpu/float/GpuFloatCodec_hip.h"
#include "dietgpu/utils/DeviceUtils_hip.h"
#include "dietgpu/utils/GreenContext_hip.h"
#include "dietgpu/utils/StackDeviceMemory_hip.h"
#include <hip/hip_fp16.h>
#else
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/GreenContext.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include <cuda_fp16.h>
#endif

dietgpu::FloatType to_dietgpu(FloatType t);
FloatType from_dietgpu(dietgpu::FloatType t);

/**
 * @brief Wrapper around dietgpu::FloatCompressSplitContext that exposes a
 * FloatType-based constructor and get_float_type() accessor, hiding the
 * internal dietgpu::FloatType from external callers.
 */
struct FloatCompressCtx : public dietgpu::FloatCompressSplitContext,
                          public CompressionContext {
  FloatCompressCtx();
  explicit FloatCompressCtx(FloatType ft);

  // Raw device buffer for the [in_ptr, inSize, out_ptr] tuple in params_dev.
  // Plain cudaMalloc so each ctx frees independently (no LIFO constraint).
  void* raw_params_dev = nullptr;

  FloatCompressCtx(FloatCompressCtx const&) = delete;
  FloatCompressCtx& operator=(FloatCompressCtx const&) = delete;

  ~FloatCompressCtx() override;

  FloatType get_float_type() const override;
  size_t get_max_size() const override;
};

#else

/**
 * @brief Dummy device allocation that mimics dietgpu's
 * StackDeviceMemory::Reservation. Provides a no-op release() method for
 * compatibility.
 */
struct DummyDevAlloc {
  void release() {}
  void* data() { return nullptr; }
};

/**
 * @brief Dummy compression context that mirrors the interface of
 * dietgpu::FloatCompressSplitContext. This allows code to compile
 * without #ifdef guards scattered throughout.
 */
struct DummyCompressCtx : public CompressionContext {
  FloatType float_type = FloatType::kUndefined;
  size_t max_size = 0;
  DummyDevAlloc params_dev;
  DummyDevAlloc histogram_dev;
  DummyDevAlloc toComp_dev;

  DummyCompressCtx();
  explicit DummyCompressCtx(FloatType ft);

  FloatType get_float_type() const override;
  size_t get_max_size() const override;
};

#endif

CompressCtx make_compress_ctx(FloatType ft);

enum class SplitBatchResult {
  kOk,      // split submitted; compress_buffer_slot is assigned
  kBusy,    // every fitting slot is in flight; caller should drain and retry
  kFailed,  // message cannot fit a slot, or the request is invalid
};

/**
 * @brief Abstract interface for compressor backends.
 *
 * This pure virtual interface defines the compression/decompression operations
 * without any DietGPU-specific types. Implementations can use DietGPU or
 * provide a null/stub implementation.
 */
class ICompressorBackend {
 public:
  virtual ~ICompressorBackend() = default;

  /**
   * @brief Get the compression buffer.
   * @return Shared pointer to the compression buffer RegMemBlock.
   */
  virtual std::shared_ptr<RegMemBlock> get_compress_buffer() const = 0;

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  virtual std::shared_ptr<RegMemBlock> get_decompress_buffer() const = 0;

  /**
   * @brief Compress a send request's data.
   * @param req The send request to compress.
   * @return true on success, false on failure.
   */
  virtual bool compress(std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   * @param input The RemoteMemInfo containing compressed data.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type.
   * @return true on success, false on failure.
   */
  virtual bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                          FloatType float_type) = 0;

  // Queue decompress asynchronously. on_done fires via gpuLaunchHostFunc after
  // the kernel completes; it must not call any GPU APIs.
  virtual void decompress_async(RemoteMemInfo const& input, RegMemBlock& output,
                                FloatType float_type, gpuHostFn_t on_done,
                                void* user_data) = 0;

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  virtual bool should_compress(size_t size) = 0;

  /**
   * @brief Check if data should be compressed and split first.
   * @param size The size in bytes to check.
   * @return true if should compress and split first.
   */
  virtual bool should_compress_and_split_first(size_t size) = 0;

  /**
   * @brief Prepare a context for split+encode two-phase compression.
   * @param addr Device pointer to the input float data.
   * @param size Size of the input data in bytes.
   * @param ctx Compression context.
   */
  virtual void prepare_split_context(void* addr, size_t size,
                                     CompressCtx ctx) = 0;

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   * @param req The send request with compress_ctx already prepared.
   * @return kOk when the split is queued, kBusy when the slot pool is full,
   * kFailed when the message cannot fit a slot or the request is invalid.
   */
  virtual SplitBatchResult compress_split_one_batch(
      std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   * @param req The send request with compress_ctx already split.
   * @return compressed size on success, 0 on failure.
   */
  virtual uint32_t compress_encode_one_batch(
      std::shared_ptr<RDMASendRequest> req) = 0;

  /**
   * @brief Release a writer-side compress-buffer slot previously assigned to a
   * request (called once all chunk ACKs for the request are in, i.e. the NIC
   * no longer DMAs from the slot). No-op for backends without slots.
   */
  virtual void free_compress_slot(int slot) = 0;

  /**
   * @brief Get the compression strategy.
   * @return The current compression strategy.
   */
  virtual CompressStrategy get_compress_strategy() = 0;
};

/**
 * @brief Null compressor backend that provides no-op implementations.
 *
 * This backend is used when DietGPU is not available. All compression
 * operations return false or no-op results.
 */
class NullCompressorBackend : public ICompressorBackend {
 public:
  NullCompressorBackend();
  ~NullCompressorBackend() override;

  std::shared_ptr<RegMemBlock> get_compress_buffer() const override;

  std::shared_ptr<RegMemBlock> get_decompress_buffer() const override;

  bool compress(std::shared_ptr<RDMASendRequest> req) override;

  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  FloatType float_type) override;

  void decompress_async(RemoteMemInfo const& input, RegMemBlock& output,
                        FloatType float_type, gpuHostFn_t on_done,
                        void* user_data) override;

  bool should_compress(size_t size) override;

  bool should_compress_and_split_first(size_t size) override;

  void prepare_split_context(void* addr, size_t size, CompressCtx ctx) override;

  SplitBatchResult compress_split_one_batch(
      std::shared_ptr<RDMASendRequest> req) override;

  uint32_t compress_encode_one_batch(
      std::shared_ptr<RDMASendRequest> req) override;

  void free_compress_slot(int slot) override {}

  CompressStrategy get_compress_strategy() override;

 private:
  CompressStrategy compress_strategy_;
};

#ifdef USE_DIETGPU
/**
 * @brief DietGPU-based compressor backend.
 *
 * This backend uses DietGPU library for GPU float compression and
 * decompression.
 */
class DietGPUCompressorBackend : public ICompressorBackend {
 public:
  DietGPUCompressorBackend();
  ~DietGPUCompressorBackend() override;

  std::shared_ptr<RegMemBlock> get_compress_buffer() const override;

  std::shared_ptr<RegMemBlock> get_decompress_buffer() const override;

  bool compress(std::shared_ptr<RDMASendRequest> req) override;

  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  FloatType float_type) override;

  void decompress_async(RemoteMemInfo const& input, RegMemBlock& output,
                        FloatType float_type, gpuHostFn_t on_done,
                        void* user_data) override;

  bool should_compress(size_t size) override;

  bool should_compress_and_split_first(size_t size) override;

  void prepare_split_context(void* addr, size_t size, CompressCtx ctx) override;

  SplitBatchResult compress_split_one_batch(
      std::shared_ptr<RDMASendRequest> req) override;

  uint32_t compress_encode_one_batch(
      std::shared_ptr<RDMASendRequest> req) override;

  void free_compress_slot(int slot) override;

  CompressStrategy get_compress_strategy() override;

 private:
  uint32_t* dev_compressed_size_;
  std::shared_ptr<RegMemBlock> buffer_;            // Compression buffer
  std::shared_ptr<RegMemBlock> decompressBuffer_;  // Decompression buffer
  gpuStream_t stream_;                             // GPU stream
  dietgpu::StackDeviceMemory* res_;  // Device memory for compress/decompress
  CompressStrategy compress_strategy_;
  // Device owning stream_/buffer_/decompressBuffer_/res_. Codec operations
  // may run on engine background threads whose current device differs;
  // kernel launches must be pinned to this device.
  int device_idx_;
  // Serializes stream_, res_, and dev_compressed_size_. Registration
  // (prepare_split_context) and the send thread both use this stream.
  std::mutex codec_mu_;

  // Writer-side compress-buffer slot pool. Pipelined writers keep several
  // messages in flight. Each in-flight compressed message needs its own
  // staging interval inside buffer_ so the next message's split kernels
  // cannot overwrite bytes the NIC may still be DMA-ing. Slots are 256 MiB;
  // assigned in compress_split and released on full-ACK. Slot capacity must
  // cover getMaxFloatCompressedSize() of the largest message.
  static size_t constexpr kSlotBytes = 256ull * 1024 * 1024;
  std::mutex slot_mu_;
  std::vector<char> slot_busy_;  // one byte per slot; nonzero = in use

  int alloc_compress_slot(FloatCompressCtx& ctx);
  size_t slot_capacity(int slot) const {
    uint64_t begin = static_cast<uint64_t>(slot) * kSlotBytes;
    if (!buffer_ || begin >= buffer_->size) return 0;
    uint64_t avail = buffer_->size - begin;
    if (avail > kSlotBytes) avail = kSlotBytes;
    return static_cast<size_t>(avail);
  }
  void* compress_slot_base(int slot) {
    return static_cast<char*>(buffer_->addr) +
           static_cast<uint64_t>(slot) * kSlotBytes;
  }
};

#endif  // USE_DIETGPU

/**
 * @brief Compressor class for handling GPU float compression and decompression.
 *
 * This class is a facade that delegates to the appropriate backend based on
 * compile-time configuration (USE_DIETGPU). It manages a singleton instance
 * and forwards all operations to the selected backend.
 */
class Compressor {
 public:
  static Compressor& get_instance();

  // Non-copyable, non-movable
  Compressor(Compressor const&) = delete;
  Compressor& operator=(Compressor const&) = delete;
  Compressor(Compressor&&) = delete;
  Compressor& operator=(Compressor&&) = delete;

  /**
   * @brief Get the compression buffer.
   * @return Shared pointer to the compression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> get_compress_buffer() const;

  /**
   * @brief Get the decompression buffer.
   * @return Shared pointer to the decompression buffer RegMemBlock.
   */
  std::shared_ptr<RegMemBlock> get_decompress_buffer() const;

  /**
   * @brief Compress a send request's data.
   * @param req The send request to compress.
   * @return true on success, false on failure.
   */
  bool compress(std::shared_ptr<RDMASendRequest> req);

  /**
   * @brief Decompress data from RemoteMemInfo to RegMemBlock.
   * @param input The RemoteMemInfo containing compressed data.
   * @param output The RegMemBlock to write decompressed data to.
   * @param float_type The float type as uint32_t.
   * @return true on success, false on failure.
   */
  bool decompress(RemoteMemInfo const& input, RegMemBlock& output,
                  FloatType float_type);

  void decompress_async(RemoteMemInfo const& input, RegMemBlock& output,
                        FloatType float_type, gpuHostFn_t on_done,
                        void* user_data);

  /**
   * @brief Check if a request should be compressed based on size threshold.
   * @param size The size in bytes to check.
   * @return true if the size meets the minimum compression threshold.
   */
  bool should_compress(size_t size);

  /**
   * @brief Check if data should be compressed and split first.
   * @param size The size in bytes to check.
   * @return true if should compress and split first.
   */
  bool should_compress_and_split_first(size_t size);

  /**
   * @brief Prepare a context for split+encode two-phase compression.
   * @param addr Device pointer to the input float data.
   * @param size Size of the input data in bytes.
   * @param ctx Compression context.
   */
  void prepare_split_context(void* addr, size_t size, CompressCtx ctx);

  /**
   * @brief Phase 1 of two-phase compression: split float data.
   * @param req The send request with compress_ctx already prepared.
   * @return kOk when the split is queued, kBusy when the slot pool is full,
   * kFailed when the message cannot fit a slot or the request is invalid.
   */
  SplitBatchResult compress_split_one_batch(
      std::shared_ptr<RDMASendRequest> req);

  /**
   * @brief Phase 2 of two-phase compression: ANS encode.
   * @param req The send request with compress_ctx already split.
   * @return compressed size on success, 0 on failure.
   */
  uint32_t compress_encode_one_batch(std::shared_ptr<RDMASendRequest> req);

  /** Release a writer-side compress-buffer slot (full-ACK of the request). */
  void free_compress_slot(int slot);

  /**
   * @brief Get the compression strategy.
   * @return The current compression strategy.
   */
  CompressStrategy get_compress_strategy();

 private:
  Compressor();

  ~Compressor();

  std::unique_ptr<ICompressorBackend> backend_;
};
