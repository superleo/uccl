// MUSA 原生编译入口；复用共享 codec 主体，不同时编译对应 .cu。
#ifndef DIETGPU_USE_MUSA
#define DIETGPU_USE_MUSA
#endif
#include "GpuFloatCompress.cu"
