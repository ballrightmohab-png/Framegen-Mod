# Frame Generator — LeviLaunchroid native mod

Inserts a GPU-blended frame between every two real rendered frames, so the
display presents roughly twice as many frames per second as the game
actually renders.

## How it works

- Hooks `eglSwapBuffers` (a stable Android system entry point, resolved at
  runtime via `dlsym`) using `pl::memory::hook` from the preloader SDK —
  no Minecraft-internal offsets or version-specific signatures needed.
- Keeps the last two presented frames as GL textures (`glCopyTexImage2D`
  from the default framebuffer right before each real swap).
- On every swap after the first, draws a full-screen blend of the previous
  and current frame (`mix(prev, cur, t)`), presents that as an extra
  frame, then redraws and presents the true current frame.
- A Mod Menu module ("Frame Generator") lets you toggle it on/off and
  adjust "Interpolation Strength" (the blend weight) live in-game.

## What this is *not*

This is **temporal blending**, not motion-vector-based frame generation
(DLSS 3 / FSR 3 style). It's the realistic thing to build without engine
access to motion vectors or an optical-flow pass. Expect:

- **Ghosting/double-imaging** on fast camera movement or fast-moving
  entities — the inserted frame is a cross-fade, not a motion-compensated
  guess.
- **~1 extra frame of input-to-display latency**, same tradeoff every
  frame-generation technique makes.
- Best results at higher base framerates and slower camera movement
  (building, exploring) — worst on quick PvP flicks or spinning fast.

If you want true motion-compensated interpolation later, the next step
would be adding a cheap per-pixel motion-vector estimate (e.g. a small
block-matching pass between prev/cur downsampled frames) and warping
`prev` toward `cur` before blending — a meaningfully bigger project than
this file, but this hook + texture pipeline is the foundation for it.

## Building

I couldn't compile this in my own sandbox (no network access to fetch the
`preloader-android` submodule/`fmt`, no Android NDK installed there), so
this is source you build the same way you'd build any of
LeviLaunchroid's own `examples/`:

1. Get the SDK the mod links against:
   ```
   git clone https://github.com/LiteLDev/preloader-android
   ```
   (This is the same submodule referenced at
   `app/src/main/cpp/preloader` in the LeviLaunchroid repo — if you
   already have a full checkout with submodules initialized
   (`git submodule update --init`), you can point at that instead.)

2. Install an Android NDK (r26+) — via Android Studio's SDK Manager, or
   standalone.

3. Build:
   - Windows (PowerShell):
     ```
     .\build.ps1 -PreloaderRoot C:\path\to\preloader-android
     ```
   - Linux/macOS/Termux:
     ```
     ./build.sh --preloader-root /path/to/preloader-android
     ```
   Both scripts will look for the NDK via `ANDROID_NDK_HOME` /
   `ANDROID_NDK_ROOT`, or accept `-Ndk` / `--ndk` explicitly.

4. Output:
   ```
   dist/arm64-v8a/framegen-mod/manifest.json
   dist/arm64-v8a/framegen-mod/libframegen_mod.so
   dist/arm64-v8a/framegen-mod.levipack
   ```

## Installing

Import `framegen-mod.levipack` into LeviLaunchroid the same way you'd
import any other native mod package, then enable "Frame Generator" from
the in-game Mod Menu overlay.

## Files

| File | Purpose |
| --- | --- |
| `src/FrameGenMod.cpp` | Lifecycle, `eglSwapBuffers` hook, GL blending pipeline, Mod Menu registration. |
| `manifest.json` | `type: preload-native` package manifest. |
| `CMakeLists.txt` | Links against `preloader-android` + `fmt`, `EGL`, `GLESv2`. |
| `build.ps1` / `build.sh` | Build + package into a `.levipack`. |
