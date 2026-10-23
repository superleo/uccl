"""CPU/static contracts for the DietGPU MUSA adapter.

These tests intentionally do not require a MUSA SDK or GPU. Real SDK compile
and S5000 roundtrip coverage belongs to the local verification sheet.
"""

import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "thirdparty/dietgpu/dietgpu/utils/GpuRuntime.h"
MUSA_MAKEFILE = ROOT / "thirdparty/dietgpu/dietgpu/float/Makefile.musa"
P2P_MAKEFILE = ROOT / "p2p/Makefile.musa"
DEVICE_DEFS = ROOT / "thirdparty/dietgpu/dietgpu/utils/DeviceDefs.cuh"
PTX_UTILS = ROOT / "thirdparty/dietgpu/dietgpu/utils/PtxUtils.cuh"
MUSA_WARP = ROOT / "thirdparty/dietgpu/dietgpu/utils/MusaWarp.muh"
ENGINE = ROOT / "p2p/engine.cc"
COMPRESSION = ROOT / "p2p/rdma/compression.cc"


class DietGpuMusaContractTest(unittest.TestCase):
    def test_runtime_adapter_compiles_against_musa_shaped_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            include_dir = pathlib.Path(directory)
            (include_dir / "musa_runtime.h").write_text(
                textwrap.dedent(
                    """
                    #pragma once
                    typedef int musaError_t;
                    typedef void* musaStream_t;
                    typedef void* musaEvent_t;
                    typedef int musaMemcpyKind;
                    typedef struct { int maxThreadsPerBlock; int sharedMemPerBlock;
                                     int major; } musaDeviceProp;
                    typedef struct { int type; int device; } musaPointerAttributes;
                    enum {
                      musaSuccess = 0, musaErrorNoDevice = 1,
                      musaErrorInvalidValue = 2, musaMemoryTypeHost = 3,
                      musaMemoryTypeDevice = 4, musaStreamDefault = 0,
                      musaStreamNonBlocking = 1, musaEventDefault = 0,
                      musaEventDisableTiming = 2, musaMemcpyHostToDevice = 3,
                      musaMemcpyDeviceToHost = 4, musaMemcpyDeviceToDevice = 5
                    };
                    inline const char* musaGetErrorString(musaError_t) { return ""; }
                    inline const char* musaGetErrorName(musaError_t) { return ""; }
                    inline musaError_t musaGetDevice(int*) { return musaSuccess; }
                    inline musaError_t musaSetDevice(int) { return musaSuccess; }
                    inline musaError_t musaGetDeviceCount(int*) { return musaSuccess; }
                    inline musaError_t musaDeviceSynchronize() { return musaSuccess; }
                    inline musaError_t musaGetLastError() { return musaSuccess; }
                    inline musaError_t musaProfilerStart() { return musaSuccess; }
                    inline musaError_t musaProfilerStop() { return musaSuccess; }
                    """
                )
            )
            (include_dir / "musa.h").write_text("#pragma once\n")
            source = include_dir / "contract.cc"
            source.write_text(
                textwrap.dedent(
                    """
                    #define DIETGPU_USE_MUSA 1
                    #include "dietgpu/utils/GpuRuntime.h"
                    int main() {
                      cudaStream_t stream = nullptr;
                      cudaDeviceProp prop{};
                      cudaPointerAttributes attrs{};
                      cudaError_t status = cudaSuccess;
                      return stream == nullptr && prop.major == 0 &&
                                     attrs.device == 0 && status == cudaSuccess
                          ? 0
                          : 1;
                    }
                    """
                )
            )
            result = subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-DDIETGPU_USE_MUSA=1",
                    "-I",
                    str(include_dir),
                    "-I",
                    str(ROOT / "thirdparty/dietgpu"),
                    "-c",
                    str(source),
                    "-o",
                    str(include_dir / "contract.o"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_green_context_is_a_musa_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            include_dir = pathlib.Path(directory)
            (include_dir / "musa_runtime.h").write_text(
                "#pragma once\n"
                "typedef int musaError_t;\n"
                "enum { musaSuccess = 0 };\n"
            )
            (include_dir / "musa.h").write_text("#pragma once\n")
            source = include_dir / "green_context.cc"
            source.write_text(
                textwrap.dedent(
                    """
                    #define DIETGPU_USE_MUSA 1
                    #include "dietgpu/utils/GreenContext.h"
                    int main() {
                      dietgpu::initGreenContextIfNeeded(0);
                      return 0;
                    }
                    """
                )
            )
            result = subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-I",
                    str(include_dir),
                    "-I",
                    str(ROOT / "thirdparty/dietgpu"),
                    str(source),
                    "-o",
                    str(include_dir / "green_context"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_musa_build_files_define_mvp_contract(self):
        self.assertIn("--offload-arch=$(MUSA_ARCH)", MUSA_MAKEFILE.read_text())
        self.assertIn("DIETGPU_USE_MUSA", MUSA_MAKEFILE.read_text())
        self.assertIn("MUSA_CXX_INCLUDE", MUSA_MAKEFILE.read_text())
        p2p_makefile = P2P_MAKEFILE.read_text()
        self.assertIn("USE_DIETGPU", p2p_makefile)
        self.assertIn("libdietgpu_float_musa", p2p_makefile)

    def test_device_sources_have_explicit_musa_branches(self):
        self.assertIn("DIETGPU_USE_MUSA", DEVICE_DEFS.read_text())
        self.assertIn("DIETGPU_USE_MUSA", PTX_UTILS.read_text())

    def test_musa_warp_reductions_are_compile_time_unrolled(self):
        source = MUSA_WARP.read_text()
        self.assertIn("#pragma unroll", source)
        self.assertNotIn("unroll(disable)", source)

    def test_gpu_proxy_threads_pin_the_local_device_once(self):
        source = ENGINE.read_text()
        for method in (
            "void Endpoint::send_proxy_thread_func()",
            "void Endpoint::recv_proxy_thread_func()",
        ):
            start = source.index(method)
            body = source[start : source.index("\n}", start) + 2]
            self.assertIn("GPU_RT_CHECK(gpuSetDevice(local_gpu_idx_));", body)

    def test_split_and_encode_do_not_repeat_codec_synchronization(self):
        source = COMPRESSION.read_text()
        split_start = source.index(
            "SplitBatchResult DietGPUCompressorBackend::compress_split_one_batch("
        )
        split_end = source.index(
            "uint32_t DietGPUCompressorBackend::compress_encode_one_batch(",
            split_start,
        )
        encode_start = split_end
        encode_end = source.index(
            "CompressStrategy DietGPUCompressorBackend::get_compress_strategy()",
            encode_start,
        )
        self.assertNotIn("gpuStreamSynchronize(stream_)", source[split_start:split_end])
        self.assertNotIn(
            "gpuStreamSynchronize(stream_)", source[encode_start:encode_end]
        )


if __name__ == "__main__":
    unittest.main()
