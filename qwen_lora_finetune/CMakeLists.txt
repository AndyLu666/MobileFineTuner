cmake_minimum_required(VERSION 3.16)
project(QwenLoRAUnified CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Operators root
get_filename_component(OPERATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../operators" ABSOLUTE)
set(OPERATORS_LIB "${OPERATORS_ROOT}/build/lib/liboperators.a")

if(EXISTS "${OPERATORS_LIB}")
    message(STATUS "Using prebuilt operators static library: ${OPERATORS_LIB}")
    add_library(operators STATIC IMPORTED GLOBAL)
    set_target_properties(operators PROPERTIES
        IMPORTED_LOCATION "${OPERATORS_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPERATORS_ROOT};${OPERATORS_ROOT}/finetune_ops;${OPERATORS_ROOT}/opt_ops;${OPERATORS_ROOT}/opt_ops/core_ext;${OPERATORS_ROOT}/opt_ops/parameter"
    )
else()
    message(STATUS "Prebuilt operators static library not found, building from subdirectory...")
    add_subdirectory("${OPERATORS_ROOT}" operators_build)
endif()

# Executables: wikitext & mmlu
add_executable(train_wikitext
    src/wikitext_main.cpp
)
target_link_libraries(train_wikitext PRIVATE operators)

add_executable(train_mmlu
    src/mmlu_main.cpp
)
target_link_libraries(train_mmlu PRIVATE operators)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(train_wikitext PRIVATE ${ACCELERATE_LIBRARY})
        target_link_libraries(train_mmlu PRIVATE ${ACCELERATE_LIBRARY})
    endif()
endif()


