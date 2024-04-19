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
#include "fmha_kernels.hpp"
#include "fmha_forward.hpp"
#include "xetla.hpp"

namespace gpu::xetla::fmha {
template <
    typename fmha_policy,
    typename scalar_t,
    int arch_tag,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsCausal,
    bool kSeqLast,
    bool kIsTraining,
    bool kIsDropout>
KERNELS_API void fmha_i<
    fmha_policy,
    scalar_t,
    arch_tag,
    kUseAlibi,
    kUseBias,
    kIsCausal,
    kSeqLast,
    kIsTraining,
    kIsDropout>::kernel_cgf(sycl::handler& cgh, arguments_t& args) {
  using fmha_forward_op_t = gpu::xetla::fmha::fmha_forward_t<
      fmha_policy,
      scalar_t,
      static_cast<gpu_arch>(arch_tag),
      kUseAlibi,
      kUseBias,
      kIsCausal,
      kSeqLast,
      kIsTraining,
      kIsDropout>;

  sycl::nd_range<3> nd_range =
      fmha_forward_op_t::get_nd_range(args.uB * args.uN, args.uF);
  // fmha_forward_op_t::check_slm_size(queue.get_info<sycl::info::queue::device>());
  // std::cout << "slm_size:\t" << fmha_forward_op_t::get_slm_size() << std::endl;
  // std::cout << "global_size:\t" << nd_range.get_global_range()[0] << ",\t"
  //           << nd_range.get_global_range()[1] << ",\t"
  //           << nd_range.get_global_range()[2] << std::endl;
  // std::cout << "local_size:\t" << nd_range.get_local_range()[0] << ",\t"
  //           << nd_range.get_local_range()[1] << ",\t"
  //           << nd_range.get_local_range()[2] << std::endl;

  cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
    fmha_forward_op_t{}(item, args);
  });
}

template struct KERNELS_API fmha_i<
    stage0<fmha_policy_32x128x128>,
    sycl::half,
    0,
    false,
    false,
    false,
    true,
    false,
    false>;
} // namespace gpu::xetla::fmha
