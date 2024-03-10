#include <tests/utils/utils.hpp>

template <gpu_arch arch_tag>
void sdp_fwd_run(uint32_t iter, uint32_t warmup = 10);

template <gpu_arch arch_tag>
struct main_wrapper {
    static void exec() {
        // This example implements scaled-dot-production with batch_size: 16,
        // num_heads: 16, sequence_length: 512, head_size: 64. It will be shown how to
        // remap the index space of each work-item used for gemm1, softmax and gemm2.

        // Description:
        // Scaled-dot-production mechanism can be seen as two chained batch MatMul
        // with a softmax in the middle layer. It can be described as following
        // mathematical expression:
        //   softmax(Q · (K.transpose(-1, -2)) * (1 / sqr_root(num_heads)) +
        //   attn_mask) · V
        // where:
        //   Q, K, V: input data
        //   shape(Q) = [16 x 16, 512, 64]
        //   shape(K) = [16 x 16, 512, 64]
        //   shape(V) = [16 x 16, 512, 64]
        //   shape(attn_mask) = [16, 512, 512]
        //   shape(DST) = [16, 512, 16, 64]

        // This kernel is designed to execute the following task:
        // 1: S = (Q · (K.transpose(-1, -2))) * (1 / sqr_root(num_heads)) + attn_mask
        // 2: S' = softmax(S)
        // 3: O = S' · V
        sdp_fwd_run<arch_tag>(10);
    };
};

int main() {
    dispatch_arch<main_wrapper>::exec();
    return 0;
}
