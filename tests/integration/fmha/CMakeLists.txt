set(TARGET fmha)

#build test
# add_integration_test(${ProjectId} ${src})  # TODO: make it as a integration_test target
add_executable(${TARGET} fmha.cpp)

set(XETLA_SUPPORTED_ARCHITECTURES xe_lpg;xe_hpg;xe_hpc)
set(device_map_xe_lpg mtl-p)
set(device_map_xe_hpg xe-hpg)
set(device_map_xe_hpc pvc)

set(XETLA_USE_AOT ON)

function(add_shared_for_arch target arch)
    string(TOUPPER ${arch} arch_upper)
    set (target_arch "${target}_${arch}")
    add_library(${target_arch} SHARED ${ARGN})
    target_compile_options(${target_arch} PRIVATE -fsycl)
    target_compile_definitions(${target_arch} PRIVATE "XETLA_KERNEL_${arch_upper}")
    target_link_options(${target_arch} PRIVATE -fsycl)
    set_target_properties(${target_arch} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    if (XETLA_USE_AOT)
        target_compile_options(${target_arch} PRIVATE -fsycl-targets=spir64_gen)
        target_link_options(${target_arch} PRIVATE -fsycl-targets=spir64_gen)
        target_link_options(${target_arch} PRIVATE -Xs "-device ${device_map_${arch}}")
    endif()
endfunction()

function(add_shared_for_architectures target)
    foreach (arch IN LISTS XETLA_SUPPORTED_ARCHITECTURES)
        add_shared_for_arch(${target} ${arch} ${ARGN})
    endforeach()
endfunction()

function(target_link_shared_for_architectures target mod lib)
    foreach (arch IN LISTS XETLA_SUPPORTED_ARCHITECTURES)
        target_link_libraries(${target} ${mod} "${lib}_${arch}")
    endforeach()
endfunction()

add_shared_for_architectures(fmha_kernels fmha_kernels.cpp)
target_link_shared_for_architectures(${TARGET} PRIVATE fmha_kernels)
