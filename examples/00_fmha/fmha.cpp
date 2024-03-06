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
#include "fmha_forward_causal_strided.hpp"
#include "fmha_policy_causal.h"
#include <tests/utils/utils.hpp>

#include "xetla.hpp"

enum class kslicing_impl_t : uint8_t { none = 0, global = 1, local = 2 };

template <kslicing_impl_t kslicing_type = kslicing_impl_t::none>
void fmha_run(uint32_t iter, uint32_t warmup = 10) {
    // // GEMM_UNIVERSAL input size
    // size_t matrix_m = 4096;
    // size_t matrix_n = 4096;
    // size_t matrix_k = 4096;

    uint32_t num_batches = 1;
    uint32_t num_heads = 16;
    uint32_t head_size = 256;
    uint32_t num_queries = 1;
    uint32_t num_keys = 33;

    // size_t size_a = matrix_m * matrix_k;
    // size_t size_b = matrix_k * matrix_n;
    // size_t size_c = matrix_m * matrix_n;

    using FMHA_T = half;

    // Define SYCL queue, context and device
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto queue = sycl::queue(properties);
    auto context = queue.get_info<info::queue::context>();
    auto device = queue.get_info<info::queue::device>();

    std::cout << "Running on " << device.get_info<info::device::name>() << "\n";

    auto Q = alloc_device_and_init<FMHA_T>(
            num_batches * num_heads * head_size * num_queries,
            [](FMHA_T *data, size_t idx) {
                data[idx] = static_cast<FMHA_T>(random_float());
            },
            queue, device, context);
    auto K = alloc_device_and_init<FMHA_T>(
            num_batches * num_heads * head_size * num_keys,
            [](FMHA_T *data, size_t idx) {
                data[idx] = static_cast<FMHA_T>(random_float());
            },
            queue, device, context);
    auto V = alloc_device_and_init<FMHA_T>(
            num_batches * num_heads * head_size * num_keys,
            [](FMHA_T *data, size_t idx) {
                data[idx] = static_cast<FMHA_T>(random_float());
            },
            queue, device, context);
    auto DST = alloc_device_and_init<FMHA_T>(
            num_batches * num_heads * head_size * num_queries,
            [](FMHA_T *data, size_t idx) {
                data[idx] = static_cast<FMHA_T>(random_float());
            },
            queue, device, context);

    using fmha_forward_op_t = gpu::xetla::fmha::fmha_forward_causal_strided_t<
            gpu::xetla::fmha_policy_8x256x256, FMHA_T, gpu_arch::Igpu, false,
            false, false, false, false>;

    sycl::nd_range<3> nd_range = fmha_forward_op_t::get_nd_range(
            num_batches * num_heads, num_queries);
    fmha_forward_op_t::check_slm_size(queue.get_info<info::queue::device>());
    int64_t ops = static_cast<int64_t>(2) * num_batches * num_heads * head_size
            * num_queries;
    profiling_helper prof("gemm_universal", ops, "gflops");
    for (uint32_t i = 0; i < iter + warmup; i++) {
        if (i >= warmup) { prof.cpu_start(); }
        auto gpu_event = queue.submit([&](handler &cgh) {
            cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
                typename fmha_forward_op_t::arguments_t kern_args(Q, K, V,
                        nullptr, nullptr, nullptr, DST, num_batches, num_heads,
                        head_size, num_queries, num_keys, 1.f, 0, 0, 0);
                fmha_forward_op_t {}(item, kern_args);
            });
        });
        gpu_event.wait();

        if (i >= warmup) {
            prof.cpu_end();
            prof.add_gpu_event(gpu_event);
        }
    }

    // performance
    prof.print_profiling_result(profiling_selector::GPU);

    free(Q, context);
    free(K, context);
    free(V, context);
    free(DST, context);
}

int main() {
    fmha_run<kslicing_impl_t::none>(1);
}
