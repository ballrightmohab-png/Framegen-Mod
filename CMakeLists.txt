cmake_minimum_required(VERSION 3.22)

project(levilaunchroid_framegen_mod LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ANDROID)
    message(FATAL_ERROR "This mod targets Android only. Configure with the Android NDK toolchain.")
endif()

include(FetchContent)

# Point this at a checkout of https://github.com/LiteLDev/preloader-android
# (the same submodule LeviLaunchroid itself uses at
# app/src/main/cpp/preloader). Defaults to that path if you build from
# inside a full LeviLaunchroid checkout with submodules initialized.
get_filename_component(LL_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(PRELOADER_ANDROID_ROOT
    "${LL_REPO_ROOT}/app/src/main/cpp/preloader"
    CACHE PATH "Path to LiteLDev/preloader-android")

if(NOT EXISTS "${PRELOADER_ANDROID_ROOT}/include/pl/Mod.hpp")
    message(FATAL_ERROR
        "PRELOADER_ANDROID_ROOT must point to a preloader-android checkout. "
        "Missing: ${PRELOADER_ANDROID_ROOT}/include/pl/Mod.hpp\n"
        "Clone https://github.com/LiteLDev/preloader-android and pass "
        "-DPRELOADER_ANDROID_ROOT=<path>, or build from inside a "
        "LeviLaunchroid checkout with 'git submodule update --init'.")
endif()

option(LL_FRAMEGEN_LINK_PRELOADER
       "Build and link the local preloader CMake target"
       ON)

if(LL_FRAMEGEN_LINK_PRELOADER AND NOT TARGET preloader)
    add_subdirectory("${PRELOADER_ANDROID_ROOT}" "${CMAKE_BINARY_DIR}/_preloader_android")
endif()

# pl::log::Logger's formatting helpers are fmt-based; link it directly
# since we call getLogger().info/error(...) with format arguments.
if(NOT TARGET fmt::fmt)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.2.0)
    set(FMT_DOC OFF CACHE BOOL "Disable fmt docs" FORCE)
    set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
    set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install rules" FORCE)
    FetchContent_MakeAvailable(fmt)
endif()

if(ANDROID_ABI)
    set(LL_FRAMEGEN_OUTPUT_ABI "${ANDROID_ABI}")
else()
    set(LL_FRAMEGEN_OUTPUT_ABI "android")
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out/${LL_FRAMEGEN_OUTPUT_ABI}")

find_library(LL_LOG_LIB log)
find_library(LL_EGL_LIB EGL)
find_library(LL_GLES2_LIB GLESv2)

add_library(framegen_mod SHARED
    src/FrameGenMod.cpp)
set_target_properties(framegen_mod PROPERTIES OUTPUT_NAME "framegen_mod")

target_include_directories(framegen_mod PRIVATE
    "${PRELOADER_ANDROID_ROOT}/include")

target_compile_options(framegen_mod PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall -Wextra -Wpedantic -fvisibility=hidden -fvisibility-inlines-hidden>)

target_link_libraries(framegen_mod PRIVATE fmt::fmt)

if(LL_LOG_LIB)
    target_link_libraries(framegen_mod PRIVATE "${LL_LOG_LIB}")
endif()
if(LL_EGL_LIB)
    target_link_libraries(framegen_mod PRIVATE "${LL_EGL_LIB}")
endif()
if(LL_GLES2_LIB)
    target_link_libraries(framegen_mod PRIVATE "${LL_GLES2_LIB}")
endif()

if(TARGET preloader)
    target_link_libraries(framegen_mod PRIVATE preloader)
else()
    target_link_options(framegen_mod PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif()
