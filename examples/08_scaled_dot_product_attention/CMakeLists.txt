
add_library(sdp_impl_dg2 scaled_dot_product_attention_kern.cpp)
target_compile_options(sdp_impl_dg2 PUBLIC -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=intel_gpu_dg2_g10)
target_compile_definitions(sdp_impl_dg2 PRIVATE XETLA_AOT_DEVICE_DG2=1)
target_link_options(sdp_impl_dg2 PUBLIC
    -fsycl-max-parallel-link-jobs=4
    -fsycl-device-code-split=per_kernel
    -fsycl-targets=intel_gpu_dg2_g10
)

add_library(sdp_impl_xe scaled_dot_product_attention_kern.cpp)
target_compile_options(sdp_impl_xe PUBLIC -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=intel_gpu_pvc)
target_compile_definitions(sdp_impl_xe PRIVATE XETLA_AOT_DEVICE_XE=1)
target_link_options(sdp_impl_dg2 PUBLIC
    -fsycl-max-parallel-link-jobs=4
    -fsycl-device-code-split=per_kernel
    -fsycl-targets=intel_gpu_pvc
)

set(TARGET sdp)

#build test
add_executable(${TARGET} scaled_dot_product_attention_main.cpp)
target_compile_options(${TARGET} PRIVATE
-fsycl-device-code-split=per_kernel
)
target_link_options(${TARGET} PRIVATE
-fsycl-max-parallel-link-jobs=4
-fsycl-device-code-split=per_kernel
)
target_link_libraries(${TARGET} sdp_impl_dg2 sdp_impl_xe)

