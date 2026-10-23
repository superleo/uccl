#include "compression.h"
#include "util/debug.h"
#include "util/gpu_rt.h"
#include "util/util.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

// 诊断专用分段计时（VC_COMP_PHASE_TIMING=1）：打印 writer 侧压缩路径中
// 各次同步等待的实际耗时。会额外增加 host 侧计时开销，仅用于瓶颈归因，
// 不得用于正式性能采样。
namespace {
inline bool comp_phase_timing_enabled() {
  static bool const enabled = std::getenv("VC_COMP_PHASE_TIMING") != nullptr;
  return enabled;
}
}  // namespace

size_t get_min_compress_bytes_from_env() {
  char const* env = std::getenv("UCCL_P2P_MIN_COMPRESS_BYTES");
  if (!env) return 16ull * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 16ull * 1024 * 1024;
  return static_cast<size_t>(v);
}

size_t const& kMinCompressBytes = []() -> size_t const& {
  static size_t v = get_min_compress_bytes_from_env();
  return v;
}();

size_t get_compress_buffer_bytes_from_env() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_BUFFER_BYTES");
  if (!env) return 2ull * 1024 * 1024 * 1024;
  char* end = nullptr;
  long long v = std::strtoll(env, &end, 10);
  if (end == env || v <= 0) return 2ull * 1024 * 1024 * 1024;
  return static_cast<size_t>(v);
}

size_t const& kCompressBufferSize = []() -> size_t const& {
  static size_t v = get_compress_buffer_bytes_from_env();
  return v;
}();

CompressStrategy get_compress_strategy_from_env() {
  char const* env = std::getenv("UCCL_P2P_COMPRESS_STRATEGY");

  if (!env || env[0] == '\0') {
    return CompressStrategy::kNone;
  }

  std::string s(env);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (s == "none" || s == "off" || s == "0") {
    return CompressStrategy::kNone;
  }

  if (s == "split" || s == "split_only") {
    return CompressStrategy::kSplitOnly;
  }

  if (s == "encode" || s == "split_encode" || s == "full" || s == "1") {
    return CompressStrategy::kSplitEncode;
  }

  return CompressStrategy::kSplitEncode;
}

#if defined USE_DIETGPU
dietgpu::FloatType to_dietgpu(FloatType t) {
  switch (t) {
    case FloatType::kFloat16:
      return dietgpu::FloatType::kFloat16;
    case FloatType::kBFloat16:
      return dietgpu::FloatType::kBFloat16;
    case FloatType::kFloat32:
      return dietgpu::FloatType::kFloat32;
    case FloatType::kFloat8E4M3FN:
      return dietgpu::FloatType::kFloat8E4M3FN;
    case FloatType::kFloat8E5M2:
      return dietgpu::FloatType::kFloat8E5M2;
    case FloatType::kUndefined:
    default:
      return dietgpu::FloatType::kUndefined;
  }
}

FloatType from_dietgpu(dietgpu::FloatType t) {
  switch (t) {
    case dietgpu::FloatType::kFloat16:
      return FloatType::kFloat16;
    case dietgpu::FloatType::kBFloat16:
      return FloatType::kBFloat16;
    case dietgpu::FloatType::kFloat32:
      return FloatType::kFloat32;
    case dietgpu::FloatType::kFloat8E4M3FN:
      return FloatType::kFloat8E4M3FN;
    case dietgpu::FloatType::kFloat8E5M2:
      return FloatType::kFloat8E5M2;
    default:
      return FloatType::kUndefined;
  }
}

FloatCompressCtx::FloatCompressCtx() = default;

FloatCompressCtx::FloatCompressCtx(FloatType ft)
    : dietgpu::FloatCompressSplitContext(to_dietgpu(ft)) {}

FloatCompressCtx::~FloatCompressCtx() {
  if (raw_params_dev) {
    // dietgpu's ~GpuMemoryReservation does `if (ptr) CHECK(res)`. Our
    // hand-constructed params_dev has res=nullptr, so clear ptr first to
    // disarm the assertion; we own the buffer and free it directly.
    params_dev.ptr = nullptr;
    params_dev.res = nullptr;
    gpuFree(raw_params_dev);
    raw_params_dev = nullptr;
  }
}

FloatType FloatCompressCtx::get_float_type() const {
  return from_dietgpu(float_type);
}

size_t FloatCompressCtx::get_max_size() const { return maxSize; }

CompressCtx make_compress_ctx(FloatType ft) {
  return std::make_shared<FloatCompressCtx>(ft);
}

#else

DummyCompressCtx::DummyCompressCtx() = default;

DummyCompressCtx::DummyCompressCtx(FloatType ft)
    : float_type(ft), max_size(0) {}

FloatType DummyCompressCtx::get_float_type() const { return float_type; }

size_t DummyCompressCtx::get_max_size() const { return max_size; }

CompressCtx make_compress_ctx(FloatType ft) {
  return std::make_shared<DummyCompressCtx>(ft);
}

#endif

NullCompressorBackend::NullCompressorBackend()
    : compress_strategy_(CompressStrategy::kNone) {}

NullCompressorBackend::~NullCompressorBackend() = default;

std::shared_ptr<RegMemBlock> NullCompressorBackend::get_compress_buffer()
    const {
  return nullptr;
}

std::shared_ptr<RegMemBlock> NullCompressorBackend::get_decompress_buffer()
    const {
  return nullptr;
}

bool NullCompressorBackend::compress(std::shared_ptr<RDMASendRequest> /*req*/) {
  return false;
}

bool NullCompressorBackend::decompress(RemoteMemInfo const& /*input*/,
                                       RegMemBlock& /*output*/,
                                       FloatType /*float_type*/) {
  return false;
}

void NullCompressorBackend::decompress_async(RemoteMemInfo const& /*input*/,
                                             RegMemBlock& /*output*/,
                                             FloatType /*float_type*/,
                                             gpuHostFn_t on_done,
                                             void* user_data) {
  if (on_done) on_done(user_data);  // synchronous fallback
}

bool NullCompressorBackend::should_compress(size_t /*size*/) { return false; }

bool NullCompressorBackend::should_compress_and_split_first(size_t /*size*/) {
  return false;
}

void NullCompressorBackend::prepare_split_context(void* /*addr*/,
                                                  size_t /*size*/,
                                                  CompressCtx /*ctx*/) {}

SplitBatchResult NullCompressorBackend::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> /*req*/) {
  return SplitBatchResult::kFailed;
}

uint32_t NullCompressorBackend::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> /*req*/) {
  return 0;
}

CompressStrategy NullCompressorBackend::get_compress_strategy() {
  return compress_strategy_;
}

#ifdef USE_DIETGPU
namespace {
// The compressor's stream, buffers, and codec kernels belong to a single
// device, but compress/decompress can be invoked from engine background
// threads whose current device differs. A kernel launched from a thread
// whose current device does not own the stream fails with
// musaErrorInvalidResourceHandle. Pin the device for the duration of each
// codec operation and restore the caller's device afterwards.
class ScopedCodecDeviceGuard {
 public:
  explicit ScopedCodecDeviceGuard(int device) : switched_(false) {
    GPU_RT_CHECK(gpuGetDevice(&prev_));
    if (prev_ != device) {
      GPU_RT_CHECK(gpuSetDevice(device));
      switched_ = true;
    }
  }
  ~ScopedCodecDeviceGuard() {
    if (switched_) {
      (void)gpuSetDevice(prev_);
    }
  }

 private:
  int prev_;
  bool switched_;
};
}  // namespace

DietGPUCompressorBackend::DietGPUCompressorBackend()
    : stream_(nullptr), res_(nullptr), dev_compressed_size_(nullptr) {
  compress_strategy_ = get_compress_strategy_from_env();
  if (compress_strategy_ == CompressStrategy::kNone) {
    return;
  }
  // Capture the device that will own every codec resource below; later
  // operations on background threads must be pinned to it.
  device_idx_ = dietgpu::getCurrentDevice();
  // Initialize green context BEFORE creating any CUDA resources (streams,
  // memory) so that they all belong to the green context.
  dietgpu::initGreenContextIfNeeded(device_idx_);

  GPU_RT_CHECK(gpuMalloc(&dev_compressed_size_, sizeof(uint32_t)));
  // Initialize compression buffer
  auto allocator = std::make_shared<MemoryAllocator>();
  buffer_ = allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

  // Initialize decompression buffer
  decompressBuffer_ =
      allocator->allocate(kCompressBufferSize, MemoryType::GPU, nullptr);

  // Initialize GPU stream
  GPU_RT_CHECK(gpuStreamCreate(&stream_));

  // Initialize StackDeviceMemory for compress/decompress operations
  res_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                        3 * kCompressBufferSize);

  // Writer-side slot pool: pipelined writers keep multiple compressed
  // messages in flight; each needs its own staging interval in buffer_.
  size_t num_slots = kCompressBufferSize / kSlotBytes;
  if (num_slots < 1) num_slots = 1;
  slot_busy_.assign(num_slots, 0);
  // Per-ctx split metadata is now allocated/freed directly by each
  // FloatCompressCtx via gpuMalloc/gpuFree — no stack allocator needed.
}

DietGPUCompressorBackend::~DietGPUCompressorBackend() {
  if (stream_) {
    gpuStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (res_) {
    delete res_;
    res_ = nullptr;
  }
  if (dev_compressed_size_) {
    GPU_RT_CHECK(gpuFree(dev_compressed_size_));
  }
}

std::shared_ptr<RegMemBlock> DietGPUCompressorBackend::get_compress_buffer()
    const {
  return buffer_;
}

std::shared_ptr<RegMemBlock> DietGPUCompressorBackend::get_decompress_buffer()
    const {
  return decompressBuffer_;
}

bool DietGPUCompressorBackend::compress(std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(!req || !req->local_mem || !req->compress_ctx || !stream_ ||
               !res_ || !buffer_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress - Invalid parameters";
    return false;
  }
  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);
  // Setup compression config
  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  // Calculate element count from bytes
  uint32_t numFloats = dietgpu::getElementCountFromBytes(
      to_dietgpu(ctx->get_float_type()), req->local_mem->size);

  // Setup batch (single element batch)
  void const* inPtrs[1] = {req->local_mem->addr};
  uint32_t inSizes[1] = {numFloats};
  void* outPtrs[1] = {buffer_->addr};

  // Compress
  dietgpu::floatCompress(*res_, compressConfig,
                         1,  // numInBatch
                         inPtrs, inSizes, outPtrs, dev_compressed_size_,
                         stream_);
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));
  // Get compressed size
  uint32_t compressedSize = 0;
  GPU_RT_CHECK(gpuMemcpy(&compressedSize, dev_compressed_size_,
                         sizeof(compressedSize), gpuMemcpyDeviceToHost));

  UCCL_LOG(INFO, UCCL_RDMA)
      << "DietGPUCompressorBackend: Compressed " << req->local_mem->size
      << " bytes to " << compressedSize << " bytes, ratio: "
      << static_cast<float>(compressedSize) /
             static_cast<float>(req->local_mem->size)
      << "x";

  // Update request to use compressed buffer
  req->local_mem = std::make_shared<RegMemBlock>(
      buffer_->addr, compressedSize, buffer_->mr_array, buffer_->type);
  return true;
}

bool DietGPUCompressorBackend::decompress(RemoteMemInfo const& input,
                                          RegMemBlock& output,
                                          FloatType float_type) {
  if (unlikely(!stream_ || !res_)) {
    UCCL_LOG(WARN)
        << "DietGPUCompressorBackend::decompress - Invalid internal state";
    return false;
  }

  if (unlikely(input.addr == 0 || input.length == 0)) {
    UCCL_LOG(WARN)
        << "DietGPUCompressorBackend::decompress - Invalid input parameters";
    return false;
  }

  if (unlikely(output.addr == nullptr || output.size == 0)) {
    UCCL_LOG(WARN)
        << "DietGPUCompressorBackend::decompress - Invalid output parameters";
    return false;
  }

  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  dietgpu::FloatType ft = to_dietgpu(float_type);

  // Setup decompression config
  dietgpu::FloatDecompressConfig decompressConfig;
  decompressConfig.floatType = ft;
  decompressConfig.useChecksum = false;
  decompressConfig.is16ByteAligned = true;

  // Calculate element count from bytes for output capacity
  uint32_t numFloats = dietgpu::getElementCountFromBytes(ft, output.size);

  // Setup batch for decompression
  void const* compInPtrs[1] = {reinterpret_cast<void const*>(input.addr)};
  void* decompOutPtrs[1] = {output.addr};
  uint32_t outCapacities[1] = {numFloats};

  // Decompress
  dietgpu::FloatDecompressStatus status =
      dietgpu::floatDecompress(*res_, decompressConfig,
                               1,  // numInBatch
                               compInPtrs, decompOutPtrs, outCapacities,
                               nullptr,  // out_success_dev (optional)
                               nullptr,  // out_size_dev (optional)
                               stream_);

  GPU_RT_CHECK(gpuStreamSynchronize(stream_));

  if (unlikely(status.error != dietgpu::FloatDecompressError::None)) {
    UCCL_LOG(ERROR) << "DietGPUCompressorBackend: Decompression failed!";
    return false;
  }

  UCCL_LOG(INFO, UCCL_RDMA)
      << "DietGPUCompressorBackend: Decompressed data from " << input.length
      << " bytes to " << output.size << " bytes";

  return true;
}

void DietGPUCompressorBackend::decompress_async(RemoteMemInfo const& input,
                                                RegMemBlock& output,
                                                FloatType float_type,
                                                gpuHostFn_t on_done,
                                                void* user_data) {
  if (unlikely(!stream_ || !res_ || input.addr == 0 || input.length == 0 ||
               output.addr == nullptr || output.size == 0)) {
    UCCL_LOG(WARN) << "decompress_async - invalid params; firing on_done "
                      "synchronously";
    if (on_done) on_done(user_data);
    return;
  }
  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  dietgpu::FloatType ft = to_dietgpu(float_type);
  dietgpu::FloatDecompressConfig cfg;
  cfg.floatType = ft;
  cfg.useChecksum = false;
  cfg.is16ByteAligned = true;
  uint32_t numFloats = dietgpu::getElementCountFromBytes(ft, output.size);
  void const* compIn[1] = {reinterpret_cast<void const*>(input.addr)};
  void* decompOut[1] = {output.addr};
  uint32_t outCap[1] = {numFloats};
  // Queue kernel — NO gpuStreamSynchronize here. on_done fires on a CUDA
  // host thread after the kernel completes.
  dietgpu::floatDecompress(*res_, cfg, 1, compIn, decompOut, outCap, nullptr,
                           nullptr, stream_);
  if (on_done) {
    GPU_RT_CHECK(gpuLaunchHostFunc(stream_, on_done, user_data));
  }
}

bool DietGPUCompressorBackend::should_compress(size_t size) {
  if (compress_strategy_ == CompressStrategy::kNone) {
    return false;
  }
  if (unlikely(size > buffer_->size)) {
    UCCL_LOG(ERROR) << "DietGPUCompressorBackend::should_compress: data size ("
                    << size << " bytes) exceeds compress buffer capacity ("
                    << buffer_->size << " bytes), skipping compression";
    return false;
  }
  return size >= kMinCompressBytes;
}

bool DietGPUCompressorBackend::should_compress_and_split_first(size_t size) {
  return compress_strategy_ == CompressStrategy::kSplitOnly &&
         should_compress(size);
}

void DietGPUCompressorBackend::prepare_split_context(void* addr, size_t size,
                                                     CompressCtx ctx) {
  if (unlikely(ctx == nullptr)) {
    std::cout << "unlikely(ctx == nullptr)" << std::endl;
    return;
  }
  if (unlikely(!should_compress(size))) {
    return;
  }
  // kUndefined means no float_type was supplied; skip compression.
  if (unlikely(ctx->get_float_type() == FloatType::kUndefined)) {
    return;
  }
  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  auto float_ctx = std::static_pointer_cast<FloatCompressCtx>(ctx);
  dietgpu::FloatType float_type = to_dietgpu(float_ctx->get_float_type());
  uint32_t numFloats = static_cast<uint32_t>(
      dietgpu::getElementCountFromBytes(float_type, size));
  // FP8 types pack two values into one uint16 pair, so the split kernel
  // expects the pair count (half the element count).
  if (dietgpu::isFloat8Type(float_type)) {
    assert(numFloats % 2 == 0);
    numFloats /= 2;
  }

  // params_dev layout: [in_ptr, inSize, out_ptr]. Plain cudaMalloc so
  // each ctx owns its buffer independently; raw_params_dev holds ownership.
  uintptr_t hostParams[3];
  hostParams[0] = reinterpret_cast<uintptr_t>(addr);
  hostParams[1] = static_cast<uintptr_t>(numFloats);
  hostParams[2] = reinterpret_cast<uintptr_t>(buffer_->addr);

  void* raw = nullptr;
  GPU_RT_CHECK(gpuMalloc(&raw, sizeof(hostParams)));
  GPU_RT_CHECK(gpuMemcpyAsync(raw, hostParams, sizeof(hostParams),
                              gpuMemcpyHostToDevice, stream_));
  GPU_RT_CHECK(gpuStreamSynchronize(stream_));

  if (float_ctx->raw_params_dev) {
    // Re-prepare: clear stale params_dev before reassigning, then free old
    // buffer.
    float_ctx->params_dev.ptr = nullptr;
    float_ctx->params_dev.res = nullptr;
    gpuFree(float_ctx->raw_params_dev);
  }
  float_ctx->raw_params_dev = raw;
  float_ctx->params_dev = dietgpu::GpuMemoryReservation<uintptr_t>(
      /*res=*/nullptr, dietgpu::getCurrentDevice(), stream_, raw,
      /*num=*/3, /*sizeAllocated=*/sizeof(hostParams));
  float_ctx->maxSize = size;
}

// Assign an exclusive writer-side staging slot for this request. Returns -1
// when the message's worst-case compressed size cannot fit a slot or when all
// slots are busy (caller may retry after draining completions). On success the
// ctx's device-side out pointer is re-aimed (stream-ordered before the split
// kernels) at the slot interval so concurrent in-flight messages never alias.
int DietGPUCompressorBackend::alloc_compress_slot(FloatCompressCtx& ctx) {
  dietgpu::FloatType ft = to_dietgpu(ctx.get_float_type());
  // getMaxFloatCompressedSize takes an element count and halves fp8 itself.
  uint32_t elems = static_cast<uint32_t>(
      dietgpu::getElementCountFromBytes(ft, ctx.maxSize));
  uint32_t need = dietgpu::getMaxFloatCompressedSize(ft, elems);
  int slot = -1;
  bool any_fits = false;
  {
    std::lock_guard<std::mutex> lk(slot_mu_);
    for (size_t i = 0; i < slot_busy_.size(); ++i) {
      if (slot_capacity(static_cast<int>(i)) < need) continue;
      any_fits = true;
      if (!slot_busy_[i]) {
        slot_busy_[i] = 1;
        slot = static_cast<int>(i);
        break;
      }
    }
  }
  if (slot < 0) {
    if (!any_fits) {
      UCCL_LOG(WARN) << "alloc_compress_slot: max compressed size " << need
                     << " exceeds every compress-buffer slot";
      return -2;
    }
    return -1;
  }
  uintptr_t out_word = reinterpret_cast<uintptr_t>(compress_slot_base(slot));
  GPU_RT_CHECK(gpuMemcpyAsync(
      static_cast<char*>(ctx.raw_params_dev) + 2 * sizeof(uintptr_t),
      &out_word, sizeof(uintptr_t), gpuMemcpyHostToDevice, stream_));
  return slot;
}

void DietGPUCompressorBackend::free_compress_slot(int slot) {
  if (slot < 0) return;
  std::lock_guard<std::mutex> lk(slot_mu_);
  if (slot >= 0 && static_cast<size_t>(slot) < slot_busy_.size()) {
    slot_busy_[slot] = 0;
  }
}

SplitBatchResult DietGPUCompressorBackend::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(!req || !req->compress_ctx || !stream_ || !res_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress_split_one_batch - "
                      "Invalid parameters";
    return SplitBatchResult::kFailed;
  }
  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);

  int slot = alloc_compress_slot(*ctx);
  if (slot == -1) return SplitBatchResult::kBusy;
  if (slot < 0) return SplitBatchResult::kFailed;
  req->compress_buffer_slot = slot;

  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  dietgpu::floatCompressSplitOneBatch(*res_, compressConfig, stream_, *ctx);

  int32_t data_size =
      dietgpu::roundDown(dietgpu::getUncompDataSizeFromByteSize(
                             compressConfig.floatType, ctx->maxSize),
                         ChunkSplitStrategy::kMessageChunkSizeKB);
  req->local_mem = std::make_shared<RegMemBlock>(
      compress_slot_base(slot), data_size, buffer_->mr_array, buffer_->type);
  return SplitBatchResult::kOk;
}

uint32_t DietGPUCompressorBackend::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  if (unlikely(!req || !req->compress_ctx || !stream_ || !res_ || !buffer_)) {
    UCCL_LOG(WARN) << "DietGPUCompressorBackend::compress_encode_one_batch - "
                      "Invalid parameters";
    return 0;
  }
  std::lock_guard<std::mutex> codec_lock(codec_mu_);
  ScopedCodecDeviceGuard guard(device_idx_);
  if (req->compress_buffer_slot < 0 ||
      static_cast<size_t>(req->compress_buffer_slot) >= slot_busy_.size()) {
    UCCL_LOG(WARN) << "compress_encode_one_batch: missing compress-buffer slot";
    return 0;
  }
  auto ctx = std::static_pointer_cast<FloatCompressCtx>(req->compress_ctx);

  dietgpu::FloatCompressConfig compressConfig;
  compressConfig.floatType = to_dietgpu(ctx->get_float_type());
  compressConfig.useChecksum = false;
  compressConfig.is16ByteAligned = true;

  bool const timing = comp_phase_timing_enabled();
  auto const t_esub0 = timing ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
  dietgpu::floatCompressEncodeOneBatch(*res_, compressConfig, *ctx,
                                       dev_compressed_size_, stream_);
  // Launch-to-return only. The synchronous D2H below waits for the encode
  // kernel, so a separate gpuStreamSynchronize would serialize the hot path
  // twice. size_d2h therefore includes kernel completion plus the copy.
  auto const t_launch = timing ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
  // Get compressed size
  uint32_t compressedSize = 0;
  GPU_RT_CHECK(gpuMemcpy(&compressedSize, dev_compressed_size_,
                         sizeof(compressedSize), gpuMemcpyDeviceToHost));
  uint32_t uncompressed_size = req->local_mem->size;
  if (timing) {
    auto t_d2h = std::chrono::steady_clock::now();
    // wire_bytes = 两段写入的总和（split 原样段 + ANS 压缩尾段）；
    // msg_bytes = 原始消息大小，ratio 即线路扩展因子。
    uint32_t const msg_bytes = static_cast<uint32_t>(ctx->maxSize);
    fprintf(stderr,
            "[compress] encode_submit=%.3fms size_d2h=%.3fms "
            "wire_bytes=%u msg_bytes=%u split_raw=%u ans_tail=%u ratio=%.4f\n",
            std::chrono::duration<double, std::milli>(t_launch - t_esub0)
                .count(),
            std::chrono::duration<double, std::milli>(t_d2h - t_launch).count(),
            compressedSize, msg_bytes, uncompressed_size,
            compressedSize - uncompressed_size,
            msg_bytes ? static_cast<float>(compressedSize) /
                            static_cast<float>(msg_bytes)
                      : 0.0f);
  }

  UCCL_LOG(INFO, UCCL_RDMA)
      << "DietGPUCompressorBackend: Encode done, " << req->local_mem->size
      << " bytes -> " << compressedSize << " bytes, ratio: "
      << static_cast<float>(compressedSize) /
             static_cast<float>(req->local_mem->size)
      << "x";

  // Update request to use compressed buffer
  void* slot_base = compress_slot_base(req->compress_buffer_slot);
  req->local_mem = std::make_shared<RegMemBlock>(
      static_cast<char*>(slot_base) + uncompressed_size,
      compressedSize - uncompressed_size, buffer_->mr_array, buffer_->type);
  req->remote_mem->addr = req->remote_mem->addr + uncompressed_size;
  ctx->histogram_dev.release();
  ctx->toComp_dev.release();
  return compressedSize;
}

CompressStrategy DietGPUCompressorBackend::get_compress_strategy() {
  return compress_strategy_;
}
#endif  // USE_DIETGPU

Compressor& Compressor::get_instance() {
  static Compressor instance;
  return instance;
}

std::shared_ptr<RegMemBlock> Compressor::get_compress_buffer() const {
  return backend_->get_compress_buffer();
}

std::shared_ptr<RegMemBlock> Compressor::get_decompress_buffer() const {
  return backend_->get_decompress_buffer();
}

bool Compressor::compress(std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress(req);
}

bool Compressor::decompress(RemoteMemInfo const& input, RegMemBlock& output,
                            FloatType float_type) {
  return backend_->decompress(input, output, float_type);
}

void Compressor::decompress_async(RemoteMemInfo const& input,
                                  RegMemBlock& output, FloatType float_type,
                                  gpuHostFn_t on_done, void* user_data) {
  backend_->decompress_async(input, output, float_type, on_done, user_data);
}

bool Compressor::should_compress(size_t size) {
  return backend_->should_compress(size);
}

bool Compressor::should_compress_and_split_first(size_t size) {
  return backend_->should_compress_and_split_first(size);
}

void Compressor::prepare_split_context(void* addr, size_t size,
                                       CompressCtx ctx) {
  backend_->prepare_split_context(addr, size, ctx);
}

SplitBatchResult Compressor::compress_split_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress_split_one_batch(req);
}

uint32_t Compressor::compress_encode_one_batch(
    std::shared_ptr<RDMASendRequest> req) {
  return backend_->compress_encode_one_batch(req);
}

void Compressor::free_compress_slot(int slot) {
  backend_->free_compress_slot(slot);
}

CompressStrategy Compressor::get_compress_strategy() {
  return backend_->get_compress_strategy();
}

Compressor::Compressor() {
#ifdef USE_DIETGPU
  backend_ = std::make_unique<DietGPUCompressorBackend>();
#else
  backend_ = std::make_unique<NullCompressorBackend>();
#endif
}

Compressor::~Compressor() = default;
