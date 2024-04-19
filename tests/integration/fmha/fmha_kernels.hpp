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
#pragma once
#include <CL/sycl.hpp>
#include "fmha_forward_policy.h"

#define KERNELS_API __attribute__((__visibility__("default")))

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
struct KERNELS_API fmha_i {
  using accum_t = float;
  struct KERNELS_API arguments_t {
    // Input tensors
    scalar_t* Q_ptr; // [B, F, N, H] - query
    scalar_t* K_ptr; // [B, T, N, H] - key
    scalar_t* V_ptr; // [B, T, N, H] - value
    scalar_t* A_ptr = nullptr; // [B, N, 1, T] - Alibi
    scalar_t* B_ptr = nullptr; // [1/B, 1/N, 1/F, M] - bias
    uint8_t* Dp_ptr = nullptr; // [B, N, F, T] - dropout mask
    // Output tensor
    scalar_t* O_ptr; // raw: [B, F, N, H]; permute: [B, N, F, H] - output
    accum_t* L_ptr; // logsum softmax: [B, N, F]
    // Dimension size
    uint32_t uB;
    uint32_t uN;
    uint32_t uNkv;
    uint32_t uH;
    uint32_t uF;
    uint32_t uT;
    uint32_t bias_strideB;
    uint32_t bias_strideN;
    uint32_t bias_strideF;
    // Softmax scale is the reciprocal square root of head size by default
    accum_t sm_scale;
    // Dropout scale is computed from dropout prob
    accum_t dp_prob;
    accum_t dp_scale;
    uint32_t uAT;
    uint32_t uMT;
    uint64_t seed;
    uint64_t offset;
    bool is_bias_add;

    inline arguments_t() = default;
    inline arguments_t(
        scalar_t* query,
        scalar_t* key,
        scalar_t* value,
        scalar_t* alibi,
        scalar_t* bias,
        uint8_t* dropout,
        scalar_t* out,
        accum_t* logsumsoftmax,
        uint32_t num_batches,
        uint32_t num_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t num_queries,
        uint32_t num_keys,
        uint32_t bias_strideB,
        uint32_t bias_strideN,
        uint32_t bias_strideF,
        accum_t sm_scale,
        accum_t dropout_prob,
        uint32_t alibi_padded_block_size,
        uint32_t attn_mask_padded_block_size,
        uint64_t seed_t,
        uint64_t offset_t)
        : Q_ptr(query),
          K_ptr(key),
          V_ptr(value),
          A_ptr(alibi),
          B_ptr(bias),
          Dp_ptr(dropout),
          O_ptr(out),
          L_ptr(logsumsoftmax),
          uB(num_batches),
          uN(num_heads),
          uNkv(num_kv_heads),
          uH(head_size),
          uF(num_queries),
          uT(num_keys),
          bias_strideB(bias_strideB),
          bias_strideN(bias_strideN),
          bias_strideF(bias_strideF),
          sm_scale(sm_scale),
          dp_prob(dropout_prob),
          dp_scale(1.f / (1.f - dropout_prob)),
          uAT(alibi_padded_block_size),
          uMT(attn_mask_padded_block_size),
          seed(seed_t),
          offset(offset_t),
          is_bias_add(bias_strideF == 0) {}
  };

  // void check_slm_size(const sycl::device& d);
  KERNELS_API static void kernel_cgf(sycl::handler& cgh, arguments_t& args);
};

} // namespace gpu::xetla::fmha
