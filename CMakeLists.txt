cmake_minimum_required(VERSION 3.20)
project(dynasor VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
endif()

# Header-only library target
add_library(dynasor INTERFACE)
add_library(dynasor::dynasor ALIAS dynasor)

target_include_directories(dynasor
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dynasor>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(dynasor INTERFACE cxx_std_20)

# Example executable
add_executable(dynasor_example dynasor/src.cpp)
target_link_libraries(dynasor_example PRIVATE dynasor::dynasor)

# Install rules
include(GNUInstallDirs)

install(TARGETS dynasor
        EXPORT dynasorTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES dynasor/dynasor.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT dynasorTargets
        FILE dynasorTargets.cmake
        NAMESPACE dynasor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynasor)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    dynasorConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/dynasorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dynasorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynasor
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dynasorConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dynasorConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dynasor
)
