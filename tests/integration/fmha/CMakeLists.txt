set(TARGET fmha)

#build test
# add_integration_test(${ProjectId} ${src})  # TODO: make it as a integration_test target
add_executable(${TARGET} fmha.cpp)

add_library(fmha_kernels SHARED fmha_kernels.cpp)
target_compile_options(fmha_kernels PRIVATE -fsycl)
target_link_options(fmha_kernels PRIVATE -fsycl)
set_target_properties(fmha_kernels PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(${TARGET} PRIVATE fmha_kernels)
