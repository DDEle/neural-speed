/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "xetla.hpp"
#include <utils/utils.hpp>
#define UT_DEBUG 1
using namespace gpu::xetla;
//The number of times the kernel is executed
constexpr int ITER = 1;

class test1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 16;
    static constexpr size_t mat_n = 32;
    static constexpr size_t mat_k = 32;
    static constexpr size_t wg_m = 16;
    static constexpr size_t wg_n = 16;
    static constexpr size_t sg_m = 16;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 16;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 1;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class test2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 22016;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 128;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 128;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 4;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 12288;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv3 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 11008;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv4 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 11008;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 32;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 4;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv5 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 8;
    static constexpr size_t mat_n = 151936;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv6 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 12288;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv7 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv8 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 11008;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv9 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 11008;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 4;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};
class qkv10 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 151936;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 16;
    static constexpr size_t dequant_s = 64;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t local_kslicing = 8;
    static constexpr size_t global_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = int4x2;
    using data_type_c = fp16;
};

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float, typename data_type_bias = data_type_a>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        data_type_bias *bias, uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());

    // BiasAdd
    for (uint32_t i = 0; i < gold_C.size(); ++i) {
        uint32_t col = i % n;
        gold_C[i] += bias[col];
    }

    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result = buff_cmp::xetla_buff_cmp(data, other, "gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <class Test>
void dequantize_gemm_run(int iter) {
    using namespace gpu;
    // Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;
    constexpr uint32_t global_kslicing = Test::global_kslicing;
    constexpr uint32_t local_kslicing = Test::local_kslicing;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    constexpr size_t dequant_s = Test::dequant_s;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_zero_pt = int4x2;
    using data_type_scale = fp16;
    using data_type_acc_in = fp16;
    using data_type_acc = fp16;
    using data_type_bias = fp16;

    constexpr mem_layout layout_a = Test::layout_a;
    constexpr mem_layout layout_b = Test::layout_b;

    constexpr size_t size_a = matrix_m * matrix_k;
    constexpr size_t size_b = matrix_k * matrix_n / 2;

    constexpr size_t size_scale_m = matrix_k / dequant_s;
    constexpr size_t size_scale_n = matrix_n;
    constexpr size_t size_scale = size_scale_m * size_scale_n;

    constexpr size_t size_zero_pt_m = matrix_k / dequant_s;
    constexpr size_t size_zero_pt_n = matrix_n / 2;
    constexpr size_t size_zero_pt = size_zero_pt_m * size_zero_pt_n;

    constexpr size_t size_c = matrix_m * matrix_n;
    constexpr size_t size_bias = matrix_n;

    uint32_t lda = layout_a == mem_layout::row_major ? matrix_k : matrix_m;
    uint32_t ldb = layout_b == mem_layout::row_major ? matrix_n : matrix_k;
    uint32_t ldc = matrix_n;
    //     uint32_t ld_scale = size_scale_n;
    //     uint32_t ld_zero_pt = size_zero_pt_n;

    // Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    using tile_shape = xetla::group::tile_shape_t<wg_tile_n, wg_tile_m,
            sg_tile_n, sg_tile_m>;
    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 0;

    using mem_desc_a_t = xetla::mem_desc_t<data_type_a, layout_a,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_a)>;
    using mem_desc_b_t = xetla::mem_desc_t<data_type_b, layout_b,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_b)>;
    using mem_desc_c_t = xetla::mem_desc_t<data_type_c, mem_layout::row_major,
            mem_space::global, DEVICE_MEM_ALIGNMENT / sizeof(data_type_c)>;

    using mem_desc_bias_t = xetla::mem_desc_t<data_type_bias,
            mem_layout::row_major, mem_space::global,
            DEVICE_MEM_ALIGNMENT / sizeof(data_type_bias)>;

    using compute_attr = xetla::group::compute_attr_t<data_type_acc_in,
            data_type_acc_in, data_type_acc>;
    using perf_tuning_knob = xetla::group::perf_tuning_knob_t<sg_tile_k,
            prefetch_distance, periodic_sync_interval>;

    using compute_policy
            = xetla::group::compute_policy_int4_dequantize_xmx<compute_attr,
                    perf_tuning_knob, data_type_scale, data_type_zero_pt,
                    gpu::xetla::group::quant_mode::S4_FULLRANGE_NO_ZP,
                    dequant_s, gpu_arch::Dg2>;

    using gemm_t = xetla::group::gemm_t<compute_policy, tile_shape,
            mem_desc_a_t, mem_desc_b_t>;

    using bias_op_t = gpu::xetla::subgroup::bias_add_op_t<mem_desc_bias_t,
            gpu_arch::Dg2>;
    using tile_op_t = gpu::xetla::subgroup::chained_tile_op_t<bias_op_t>;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_tile_op<tile_op_t, gpu_arch::Dg2>,
            tile_shape, mem_desc_c_t>;

    using group_swizzle = xetla::kernel::group_swizzle_default<gpu_arch::Dg2>;
    using gemm_op_t = xetla::kernel::gemm_universal_t<
            gpu::xetla::kernel::dispatch_policy_int4_dequantize_kslicing<
                    group_swizzle, global_kslicing, local_kslicing>,
            gemm_t, epilogue_t>;

    size_t size_acc = gemm_op_t::get_acc_buf_size(matrix_m, matrix_n);
    size_t size_cnt = gemm_op_t::get_cnt_buf_size(matrix_m, matrix_n);

    //Define and initialize the data required for the calculation
    auto *A_h = static_cast<data_type_a *>(
            malloc_host(size_a * sizeof(data_type_a), context));
    auto *A_hh = static_cast<data_type_a *>(
            malloc_host(size_a * sizeof(data_type_a), context));
    auto *B_h = static_cast<data_type_b *>(
            malloc_host(size_b * sizeof(data_type_b), context));
    auto *C_h = static_cast<data_type_c *>(
            malloc_host(size_c * sizeof(data_type_c), context));
    auto *Acc_h = static_cast<data_type_acc *>(
            malloc_host(size_acc * sizeof(data_type_acc), context));
    auto *Cnt_h = static_cast<uint32_t *>(
            malloc_host(size_cnt * sizeof(uint32_t), context));
    auto *scale_h = static_cast<data_type_scale *>(
            malloc_host(size_scale * sizeof(data_type_scale), context));
    auto *zero_pt_h = static_cast<data_type_zero_pt *>(
            malloc_host(size_zero_pt * sizeof(data_type_zero_pt), context));
    auto *bias_h = static_cast<data_type_bias *>(
            malloc_host(size_bias * sizeof(data_type_bias), context));

    auto *A_d = static_cast<data_type_a *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_a * sizeof(data_type_a), device, context));
    auto *B_d = static_cast<data_type_b *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_b * sizeof(data_type_b), device, context));
    auto *C_d = static_cast<data_type_c *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_c * sizeof(data_type_c), device, context));
    auto *Acc_d = static_cast<data_type_acc *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_acc * sizeof(data_type_acc), device, context));
    auto *Cnt_d
            = static_cast<uint32_t *>(aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_cnt * sizeof(uint32_t), device, context));
    auto *scale_d = static_cast<data_type_scale *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_scale * sizeof(data_type_scale), device, context));
    auto *zero_pt_d = static_cast<data_type_zero_pt *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_zero_pt * sizeof(data_type_zero_pt), device, context));
    auto *bias_d = static_cast<data_type_bias *>(
            aligned_alloc_device(DEVICE_MEM_ALIGNMENT,
                    size_bias * sizeof(data_type_bias), device, context));

    for (unsigned i = 0; i < size_a; ++i) {
        A_h[i] = random_float();
#ifdef UT_DEBUG
        A_h[i] = 1.f;
        A_h[i] = i % 16 + i / 16 * 100;
#endif
    }

        for (size_t i = 0; i < matrix_k; i++) {
            for (size_t j = 0; j < matrix_m; j++) {
                // A_hh[i * matrix_m + j] = A_h[i * matrix_m + j];
//
                A_hh[i * matrix_m + j] = A_h[j * matrix_k + i];
            }
        }

    for (unsigned i = 0; i < size_b; ++i) {
        B_h[i] = uint8_t(random_uint8());
#ifdef UT_DEBUG
        B_h[i] = 153;
#endif
    }

    for (unsigned i = 0; i < size_scale; ++i) {
        scale_h[i] = random_float();
#ifdef UT_DEBUG
        scale_h[i] = 1.f;
#endif
    }

    for (unsigned i = 0; i < size_zero_pt; ++i) {
        zero_pt_h[i] = 0.f;
    }

    for (unsigned i = 0; i < size_c; ++i) {
        C_h[i] = 0;
    }

    for (unsigned i = 0; i < size_acc; ++i) {
        Acc_h[i] = 0;
    }

    for (unsigned i = 0; i < size_cnt; ++i) {
        Cnt_h[i] = 0;
    }

    for (unsigned i = 0; i < size_bias; ++i) {
        bias_h[i] = random_float();
#ifdef UT_DEBUG
        bias_h[i] = 0.f;
#endif
    }

    queue.memcpy((void *)A_d, (void *)A_hh, size_a * sizeof(data_type_a))
            .wait();
    queue.memcpy((void *)B_d, (void *)B_h, size_b * sizeof(data_type_b)).wait();
    queue.memcpy((void *)C_d, (void *)C_h, size_c * sizeof(data_type_c)).wait();
    queue.memcpy((void *)Acc_d, (void *)Acc_h, size_acc * sizeof(data_type_acc))
            .wait();
    queue.memcpy((void *)Cnt_d, (void *)Cnt_h, size_cnt * sizeof(uint32_t))
            .wait();
    queue.memcpy((void *)scale_d, (void *)scale_h,
                 size_scale * sizeof(data_type_scale))
            .wait();
    queue.memcpy((void *)zero_pt_d, (void *)zero_pt_h,
                 size_zero_pt * sizeof(data_type_zero_pt))
            .wait();
    queue.memcpy((void *)bias_d, (void *)bias_h,
                 size_bias * sizeof(data_type_bias))
            .wait();

    // set up gemm arguments
    typename bias_op_t::shape_t bias_add_shape(matrix_n, 1, matrix_n);
    using epilogue_args_t = epilogue_t::arguments_t;

    epilogue_args_t epilogue_args({//epilogue_args init list
            // It accepts the base pointer to matrix D, and its dimensions
            {bias_d, bias_add_shape}});

    typename gemm_op_t::template arguments_t<compute_policy::quant_type>
            gemm_arg(matrix_m, matrix_k, matrix_n, A_d, lda, B_d, ldb, C_d, ldc,
                    scale_d, matrix_n, Acc_d, Cnt_d, epilogue_args);

    cl::sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);
    //     if (!gemm_op_t::can_implement(gemm_arg)) {
    //         std::cout << "The arguments cannot be supported, aborting ... "
    //                   << std::endl;
    //         FAIL();
    //     }

    size_t ops = 2 * matrix_m * matrix_n * matrix_k + matrix_m * matrix_n;
    profiling_helper prof("dequantize_gemm", ops, "gflops");
    int constexpr warm = 0;
    try {
        for (int i = 0; i < iter + warm; i++) {
            if (i >= warm) prof.cpu_start();
            auto e_esimd = queue.submit([&](handler &cgh) {
                cgh.parallel_for(
                        nd_range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                            // allocate slm and nbarrier resource
                            slm_barrier_init<gemm_op_t>();
                            gemm_op_t gemm_op;
                            gemm_op(item, gemm_arg);
                        });
            });
            if (i >= warm) {
                e_esimd.wait();
                prof.cpu_end();
                prof.add_gpu_event(e_esimd);
            }
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    std::vector<fp16> dequantize_b(matrix_k * matrix_n, 0);
    for (uint32_t i = 0; i < matrix_k / dequant_s; i++) {
        for (uint32_t j = 0; j < matrix_n / 2; j++) {
            int start_in = i * dequant_s * matrix_n / 2 + j;
            int start_out = i * dequant_s * matrix_n + j * 2;
            int start_scale = i * size_scale_n + j * 2;
            for (uint32_t ii = 0; ii < dequant_s; ii++) {
                uint8_t data_in = B_h[start_in + ii * matrix_n / 2];
                int8_t data_0 = int8_t(data_in & 0x0f) - 8;
                int8_t data_1 = int8_t(data_in >> 4) - 8;
                dequantize_b[start_out + ii * matrix_n]
                        = fp16(data_0) * scale_h[start_scale];
                dequantize_b[start_out + ii * matrix_n + 1]
                        = fp16(data_1) * scale_h[start_scale + 1];
            }
        }
    }

    queue.memcpy((void *)C_h, (void *)C_d, size_c * sizeof(data_type_c)).wait();
    ASSERT_EQ(0,
            gemm_result_validate(A_h, dequantize_b.data(), C_h, bias_h,
                    matrix_m, matrix_k, matrix_n, layout_a, layout_b));

    free(A_h, context);
    free(A_hh, context);
    free(B_h, context);
    free(C_h, context);
    free(scale_h, context);
    free(zero_pt_h, context);
    free(A_d, context);
    free(B_d, context);
    free(C_d, context);
    free(scale_d, context);
    free(zero_pt_d, context);
    free(Acc_h, context);
    free(Cnt_h, context);
    free(Acc_d, context);
    free(Cnt_d, context);
}

template <typename T>
class dequantize_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(dequantize_gemm_test);

TYPED_TEST_P(dequantize_gemm_test, esimd) {
    dequantize_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(dequantize_gemm_test, esimd);
using tests = ::testing::Types<test1>;
// using tests = ::testing::Types<qkv1, qkv2, qkv3, qkv4, qkv5, qkv6, qkv7, qkv8,
//         qkv9, qkv10>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        dequantize_gemm_test_suite, dequantize_gemm_test, tests);
