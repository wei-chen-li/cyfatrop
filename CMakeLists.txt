cmake_minimum_required(VERSION 3.19)
project(cyfatrop LANGUAGES C CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
include(ExternalProject)

# Empty overrides select the pinned archives; explicit overrides skip checksums.
set(CYFATROP_FATROP_URL "" CACHE STRING "Fatrop archive override (no checksum verification)")
set(CYFATROP_BLASFEO_URL "" CACHE STRING "BLASFEO archive override (no checksum verification)")
set(fatrop_url "${CYFATROP_FATROP_URL}")
set(fatrop_hash_args)
if(fatrop_url STREQUAL "")
  set(fatrop_url "https://codeload.github.com/meco-group/fatrop/tar.gz/86929bad166f49109133a1bf7a5ae6ecb343b2a3")
  set(fatrop_hash_args URL_HASH SHA256=e955ea29272fbf97d286fa1c5558725182582891289c0e0f07d9f075a6207ff1)
endif()
set(blasfeo_url "${CYFATROP_BLASFEO_URL}")
set(blasfeo_hash_args)
if(blasfeo_url STREQUAL "")
  set(blasfeo_url "https://codeload.github.com/giaf/blasfeo/tar.gz/5cb0a065895479e7dc54c1771fec48548aec0746")
  set(blasfeo_hash_args URL_HASH SHA256=942aa57ceefcd96ba4de874ba72305b34d173f3cfd98552ed211ca722446560d)
endif()
set(CYFATROP_BLASFEO_TARGET GENERIC CACHE STRING "BLASFEO CPU target")
set(CYFATROP_BUILD_JOBS 2 CACHE STRING "Parallel jobs per native dependency build")
set(deps "${CMAKE_CURRENT_BINARY_DIR}/native")
file(MAKE_DIRECTORY "${deps}/include")
set(common_args
  -DCMAKE_INSTALL_PREFIX=${deps} -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})

ExternalProject_Add(blasfeo_build
  URL "${blasfeo_url}"
  ${blasfeo_hash_args}
  CMAKE_ARGS ${common_args} -DTARGET=${CYFATROP_BLASFEO_TARGET}
    -DLA=HIGH_PERFORMANCE -DMF=PANELMAJ -DBLASFEO_TESTING=OFF
    -DBLASFEO_EXAMPLES=OFF -DBLASFEO_BENCHMARKS=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${CYFATROP_BUILD_JOBS}
  BUILD_BYPRODUCTS "${deps}/lib/libblasfeo.a"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON)
ExternalProject_Add(fatrop_build
  URL "${fatrop_url}"
  ${fatrop_hash_args}
  DEPENDS blasfeo_build
  CMAKE_ARGS ${common_args} -DCMAKE_PREFIX_PATH=${deps}
    -DWITH_BUILD_BLASFEO=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
    -DBUILD_WITH_C_INTERFACE=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${CYFATROP_BUILD_JOBS}
  BUILD_BYPRODUCTS "${deps}/lib/libfatrop.a"
  LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON)

add_library(cyfatrop_fatrop STATIC IMPORTED)
set_target_properties(cyfatrop_fatrop PROPERTIES
  IMPORTED_LOCATION "${deps}/lib/libfatrop.a"
  INTERFACE_INCLUDE_DIRECTORIES "${deps}/include")
add_library(cyfatrop_blasfeo STATIC IMPORTED)
set_target_properties(cyfatrop_blasfeo PROPERTIES IMPORTED_LOCATION "${deps}/lib/libblasfeo.a")

nanobind_add_module(_native cyfatrop/cpp/bindings.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
target_link_libraries(_native PRIVATE cyfatrop_fatrop cyfatrop_blasfeo ${CMAKE_DL_LIBS})
if(UNIX)
  target_link_libraries(_native PRIVATE m)
endif()
add_dependencies(_native fatrop_build)
install(TARGETS _native LIBRARY DESTINATION cyfatrop)
