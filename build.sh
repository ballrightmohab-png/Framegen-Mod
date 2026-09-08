#!/usr/bin/env bash
# Build and package the Frame Generator mod for LeviLaunchroid.
#
# Usage:
#   ./build.sh --preloader-root /path/to/preloader-android [--ndk /path/to/ndk] [--clean]
set -euo pipefail

ABI="arm64-v8a"
BUILD_TYPE="Release"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
PRELOADER_ROOT=""
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preloader-root) PRELOADER_ROOT="$2"; shift 2 ;;
    --ndk) NDK="$2"; shift 2 ;;
    --abi) ABI="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PRELOADER_ROOT" ]]; then
  echo "error: --preloader-root is required (checkout of https://github.com/LiteLDev/preloader-android)" >&2
  exit 1
fi
PRELOADER_ROOT="$(cd "$PRELOADER_ROOT" && pwd)"
if [[ ! -f "$PRELOADER_ROOT/include/pl/Mod.hpp" ]]; then
  echo "error: $PRELOADER_ROOT does not look like a preloader-android checkout" >&2
  exit 1
fi

if [[ -z "$NDK" ]]; then
  echo "error: pass --ndk or set ANDROID_NDK_HOME/ANDROID_NDK_ROOT" >&2
  exit 1
fi
TOOLCHAIN="$NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
  echo "error: Android NDK toolchain not found at $TOOLCHAIN" >&2
  exit 1
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SOURCE_DIR/build/android-$ABI-$BUILD_TYPE"

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
  rm -rf "$BUILD_DIR"
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DPRELOADER_ANDROID_ROOT="$PRELOADER_ROOT"

cmake --build "$BUILD_DIR" --target framegen_mod

OUT_DIR="$BUILD_DIR/out/$ABI"
DIST_DIR="$SOURCE_DIR/dist/$ABI"
rm -rf "$DIST_DIR"
PACKAGE_DIR="$DIST_DIR/framegen-mod"
mkdir -p "$PACKAGE_DIR"

cp "$OUT_DIR/libframegen_mod.so" "$PACKAGE_DIR/"
cp "$SOURCE_DIR/manifest.json" "$PACKAGE_DIR/"

ARCHIVE_PATH="$DIST_DIR/framegen-mod.levipack"
(cd "$PACKAGE_DIR" && zip -r -q "$ARCHIVE_PATH" .)

echo "Built:"
echo "  $PACKAGE_DIR"
echo "  $ARCHIVE_PATH"
