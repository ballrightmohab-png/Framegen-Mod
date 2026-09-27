// FrameGenMod.cpp
//
// Frame Generator
//
// Fixes:
//   - Safer Mode 0 path
//   - FPS overlay is drawn BEFORE eglSwapBuffers()
//   - No GL work after eglSwapBuffers()
//   - 10..144 FPS cap
//   - Safer frame pacing
//   - Clears/initializes history textures
//   - Prevents stale future presentation timestamps
//   - Avoids FrameGen during unstable frame intervals
//   - Avoids generating frames after a long hitch
//   - Safer perspective-change handling
//   - Keeps REAL / GENERATED FPS counter
//
// Generation:
//   0 = OFF
//   1 = 1 generated frame
//   2 = 2 generated frames
//
// Quality:
//   0 = Low
//   1 = Balanced
//   2 = Smoothness
//
// FPS Counter:
//   0 = OFF
//   1 = ON
//
// FPS Cap:
//   10..144

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <cstdio>

#include <dlfcn.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <pl/Mod.hpp>
#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>

namespace {

constexpr const char *kModuleId =
    "framegen.core";

constexpr const char *kModeKey =
    "generationMode";

constexpr const char *kPresetKey =
    "qualityPreset";

constexpr const char *kFpsCounterKey =
    "fpsCounter";

constexpr const char *kFpsCapKey =
    "fpsCap";

constexpr int kMinFpsCap =
    10;

constexpr int kMaxFpsCap =
    144;

constexpr int64_t kMinFrameIntervalNs =
    2'000'000LL;

constexpr int64_t kMaxFrameIntervalNs =
    100'000'000LL;

// Do not generate frames if the real frame took longer than this.
// This prevents huge interpolation jumps after lag spikes.
constexpr int64_t kMaxGenerationIntervalNs =
    50'000'000LL;

// Ignore extremely tiny intervals caused by duplicate/recursive swaps.
constexpr int64_t kMinimumUsefulIntervalNs =
    4'000'000LL;

// Small pacing tolerance.
constexpr int64_t kSleepMarginNs =
    500'000LL;

// ============================================================
// Fullscreen quad
// ============================================================

constexpr float kQuadVerts[16] = {
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
     1.0f,  1.0f, 1.0f, 1.0f
};

// ============================================================
// FrameGen shaders
// ============================================================

constexpr const char *kVertexShaderSrc =
R"glsl(
attribute vec2 aPos;
attribute vec2 aUV;

varying vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

constexpr const char *kBlendFragmentShaderSrc =
R"glsl(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform vec2 uMv;
uniform float uT;

void main() {

    vec2 uvPrev =
        clamp(
            vUV - uT * uMv,
            0.0,
            1.0
        );

    vec2 uvCur =
        clamp(
            vUV + (1.0 - uT) * uMv,
            0.0,
            1.0
        );

    vec4 prevColor =
        texture2D(
            uPrev,
            uvPrev
        );

    vec4 curColor =
        texture2D(
            uCur,
            uvCur
        );

    gl_FragColor =
        mix(
            prevColor,
            curColor,
            clamp(uT, 0.0, 1.0)
        );
}
)glsl";

// ============================================================
// FPS overlay shaders
// ============================================================

constexpr const char *kOverlayVertexShaderSrc =
R"glsl(
attribute vec2 aPos;

void main() {
    gl_Position =
        vec4(
            aPos,
            0.0,
            1.0
        );
}
)glsl";

constexpr const char *kOverlayFragmentShaderSrc =
R"glsl(
precision mediump float;

uniform vec4 uColor;

void main() {
    gl_FragColor = uColor;
}
)glsl";

// ============================================================
// EGL / GL types
// ============================================================

using EglSwapBuffersFn =
    decltype(&eglSwapBuffers);

using PFN_eglPresentationTimeANDROID =
    EGLBoolean(EGLAPIENTRY *)(
        EGLDisplay,
        EGLSurface,
        int64_t
    );

using PFN_glGenVertexArrays =
    void (*)(GLsizei, GLuint *);

using PFN_glBindVertexArray =
    void (*)(GLuint);

using PFN_glDeleteVertexArrays =
    void (*)(GLsizei, const GLuint *);

constexpr GLenum kVertexArrayBindingPname =
    0x85B5;

PFN_glGenVertexArrays
    gGenVertexArrays =
        nullptr;

PFN_glBindVertexArray
    gBindVertexArray =
        nullptr;

PFN_glDeleteVertexArrays
    gDeleteVertexArrays =
        nullptr;

// ============================================================
// EGL extension helper
// ============================================================

bool hasEglExtension(
    EGLDisplay display,
    const char *name
) {

    const char *extensions =
        eglQueryString(
            display,
            EGL_EXTENSIONS
        );

    if (
        extensions == nullptr ||
        name == nullptr ||
        name[0] == '\0'
    ) {
        return false;
    }

    const size_t nameLength =
        std::strlen(name);

    const char *match =
        extensions;

    while (
        (match =
            std::strstr(
                match,
                name
            )) != nullptr
    ) {

        const bool startsAtToken =
            match == extensions ||
            match[-1] == ' ';

        const char terminator =
            match[nameLength];

        if (
            startsAtToken &&
            (
                terminator == ' ' ||
                terminator == '\0'
            )
        ) {
            return true;
        }

        match += nameLength;
    }

    return false;
}

// ============================================================
// Time
// ============================================================

int64_t nowNs() {

    return std::chrono::duration_cast<
        std::chrono::nanoseconds
    >(
        std::chrono::steady_clock::now()
            .time_since_epoch()
    ).count();
}

// ============================================================
// GL State Guard
// ============================================================

class GlStateGuard {

public:

    GlStateGuard() {

        glGetIntegerv(
            GL_CURRENT_PROGRAM,
            &mProgram
        );

        glGetIntegerv(
            GL_ACTIVE_TEXTURE,
            &mActiveTexture
        );

        glGetIntegerv(
            GL_VIEWPORT,
            mViewport
        );

        glGetIntegerv(
            GL_FRAMEBUFFER_BINDING,
            &mFramebuffer
        );

        glGetIntegerv(
            GL_ARRAY_BUFFER_BINDING,
            &mArrayBuffer
        );

        glGetIntegerv(
            kVertexArrayBindingPname,
            &mVertexArray
        );

        mDepthTest =
            glIsEnabled(
                GL_DEPTH_TEST
            );

        mBlend =
            glIsEnabled(
                GL_BLEND
            );

        mCullFace =
            glIsEnabled(
                GL_CULL_FACE
            );

        mScissorTest =
            glIsEnabled(
                GL_SCISSOR_TEST
            );

        mStencilTest =
            glIsEnabled(
                GL_STENCIL_TEST
            );

        glGetBooleanv(
            GL_COLOR_WRITEMASK,
            mColorMask
        );

        glActiveTexture(
            GL_TEXTURE0
        );

        glGetIntegerv(
            GL_TEXTURE_BINDING_2D,
            &mTex0
        );

        glActiveTexture(
            GL_TEXTURE1
        );

        glGetIntegerv(
            GL_TEXTURE_BINDING_2D,
            &mTex1
        );

        glActiveTexture(
            static_cast<GLenum>(
                mActiveTexture
            )
        );
    }

    ~GlStateGuard() {

        if (
            gBindVertexArray != nullptr
        ) {

            gBindVertexArray(
                static_cast<GLuint>(
                    mVertexArray
                )
            );
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            static_cast<GLuint>(
                mFramebuffer
            )
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            static_cast<GLuint>(
                mArrayBuffer
            )
        );

        glViewport(
            mViewport[0],
            mViewport[1],
            mViewport[2],
            mViewport[3]
        );

        glUseProgram(
            static_cast<GLuint>(
                mProgram
            )
        );

        glActiveTexture(
            GL_TEXTURE0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(
                mTex0
            )
        );

        glActiveTexture(
            GL_TEXTURE1
        );

        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(
                mTex1
            )
        );

        glActiveTexture(
            static_cast<GLenum>(
                mActiveTexture
            )
        );

        setEnabled(
            GL_DEPTH_TEST,
            mDepthTest
        );

        setEnabled(
            GL_BLEND,
            mBlend
        );

        setEnabled(
            GL_CULL_FACE,
            mCullFace
        );

        setEnabled(
            GL_SCISSOR_TEST,
            mScissorTest
        );

        setEnabled(
            GL_STENCIL_TEST,
            mStencilTest
        );

        glColorMask(
            mColorMask[0],
            mColorMask[1],
            mColorMask[2],
            mColorMask[3]
        );
    }

    GlStateGuard(
        const GlStateGuard &
    ) = delete;

    GlStateGuard &operator=(
        const GlStateGuard &
    ) = delete;

private:

    static void setEnabled(
        GLenum cap,
        GLboolean enabled
    ) {

        if (enabled) {
            glEnable(cap);
        } else {
            glDisable(cap);
        }
    }

    GLint mProgram = 0;

    GLint mActiveTexture =
        GL_TEXTURE0;

    GLint mViewport[4] = {
        0,
        0,
        0,
        0
    };

    GLint mFramebuffer = 0;

    GLint mArrayBuffer = 0;

    GLint mVertexArray = 0;

    GLint mTex0 = 0;

    GLint mTex1 = 0;

    GLboolean mDepthTest =
        GL_FALSE;

    GLboolean mBlend =
        GL_FALSE;

    GLboolean mCullFace =
        GL_FALSE;

    GLboolean mScissorTest =
        GL_FALSE;

    GLboolean mStencilTest =
        GL_FALSE;

    GLboolean mColorMask[4] = {
        GL_TRUE,
        GL_TRUE,
        GL_TRUE,
        GL_TRUE
    };
};

// ============================================================
// FrameGenMod
// ============================================================

class FrameGenMod {

public:

    static FrameGenMod &instance() {

        static FrameGenMod mod;

        return mod;
    }

    FrameGenMod()
        : mSelf(
            *ll::mod::NativeMod::current()
        ) {}

    [[nodiscard]]
    ll::mod::NativeMod &getSelf() const {

        return mSelf;
    }

    bool load() {

        getSelf().getLogger().info(
            "Frame Generator loading"
        );

        return true;
    }

    bool enable() {

        void *target =
            reinterpret_cast<void *>(
                dlsym(
                    RTLD_DEFAULT,
                    "eglSwapBuffers"
                )
            );

        if (target == nullptr) {

            getSelf().getLogger().error(
                "Could not resolve eglSwapBuffers"
            );

            return false;
        }

        mSwapHook =
            pl::memory::HookHandle(
                target,
                reinterpret_cast<void *>(
                    &swapBuffersDetour
                ),
                reinterpret_cast<void **>(
                    &mOriginalSwapBuffers
                ),
                pl::memory::HookPriority::Normal
            );

        if (
            !mSwapHook.installed()
        ) {

            getSelf().getLogger().error(
                "Failed to hook eglSwapBuffers"
            );

            return false;
        }

        void *presTimeSym =
            reinterpret_cast<void *>(
                eglGetProcAddress(
                    "eglPresentationTimeANDROID"
                )
            );

        mPresentationTimeFn =
            reinterpret_cast<
                PFN_eglPresentationTimeANDROID
            >(presTimeSym);

        mTrueFrameGenAvailable =
            mPresentationTimeFn != nullptr;

        // ========================================================
        // VAO functions
        // ========================================================

        gGenVertexArrays =
            reinterpret_cast<
                PFN_glGenVertexArrays
            >(
                eglGetProcAddress(
                    "glGenVertexArraysOES"
                )
            );

        gBindVertexArray =
            reinterpret_cast<
                PFN_glBindVertexArray
            >(
                eglGetProcAddress(
                    "glBindVertexArrayOES"
                )
            );

        gDeleteVertexArrays =
            reinterpret_cast<
                PFN_glDeleteVertexArrays
            >(
                eglGetProcAddress(
                    "glDeleteVertexArraysOES"
                )
            );

        if (
            gGenVertexArrays == nullptr ||
            gBindVertexArray == nullptr ||
            gDeleteVertexArrays == nullptr
        ) {

            gGenVertexArrays =
                reinterpret_cast<
                    PFN_glGenVertexArrays
                >(
                    eglGetProcAddress(
                        "glGenVertexArrays"
                    )
                );

            gBindVertexArray =
                reinterpret_cast<
                    PFN_glBindVertexArray
                >(
                    eglGetProcAddress(
                        "glBindVertexArray"
                    )
                );

            gDeleteVertexArrays =
                reinterpret_cast<
                    PFN_glDeleteVertexArrays
                >(
                    eglGetProcAddress(
                        "glDeleteVertexArrays"
                    )
                );
        }

        mVaoSupported =
            gGenVertexArrays != nullptr &&
            gBindVertexArray != nullptr &&
            gDeleteVertexArrays != nullptr;

        // ========================================================
        // ModMenu
        // ========================================================

        const bool moduleRegistered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator"
            )
                .modId(
                    getSelf().getId()
                )
                .description(
                    "Frame generation, FPS counter "
                    "and configurable FPS cap."
                )
                .defaultEnabled(true)
                .onToggle(
                    onModuleToggle
                )

                .config(
                    kModeKey,
                    "Generation Mode",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(
                        mGenerationMode.load()
                    ),
                    "0",
                    "2"
                )

                .config(
                    kPresetKey,
                    "Quality Preset",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(
                        mQualityPreset.load()
                    ),
                    "0",
                    "2"
                )

                .config(
                    kFpsCounterKey,
                    "FPS Counter (0=OFF 1=ON)",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(
                        mFpsCounter.load()
                    ),
                    "0",
                    "1"
                )

                .config(
                    kFpsCapKey,
                    "FPS Cap",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(
                        mFpsCap.load()
                    ),
                    "10",
                    "144"
                )

                .onConfigChanged(
                    onConfigChanged
                )
                .registerModule();

        if (
            !moduleRegistered
        ) {

            getSelf().getLogger().error(
                "Failed to register Frame Generator module"
            );
        }

        resetTiming();
        resetAllStats();

        getSelf().getLogger().info(
            "Frame Generator enabled"
        );

        return moduleRegistered;
    }

    bool disable() {

        pl::modmenu::unregisterModule(
            kModuleId
        );

        mSwapHook.reset();

        destroyGlResources();

        resetTiming();
        resetAllStats();

        getSelf().getLogger().info(
            "Frame Generator disabled"
        );

        return true;
    }

    bool unload() {

        getSelf().getLogger().info(
            "Frame Generator unloaded"
        );

        return true;
    }

private:

    ll::mod::NativeMod &mSelf;

    pl::memory::HookHandle
        mSwapHook;

    EglSwapBuffersFn
        mOriginalSwapBuffers =
            nullptr;

    PFN_eglPresentationTimeANDROID
        mPresentationTimeFn =
            nullptr;

    bool mTrueFrameGenAvailable =
        false;

    // ============================================================
    // Timing
    // ============================================================

    int64_t mLastHookStartNs =
        0;

    int64_t mLastPresentedTimeNs =
        0;

    int64_t mNextCapTimeNs =
        0;

    // ============================================================
    // Settings
    // ============================================================

    std::atomic_bool mEnabled{
        true
    };

    std::atomic_int mGenerationMode{
        1
    };

    std::atomic_int mQualityPreset{
        1
    };

    std::atomic_int mFpsCounter{
        1
    };

    std::atomic_int mFpsCap{
        60
    };

    // ============================================================
    // Statistics
    // ============================================================

    int64_t mStatsWindowStartNs =
        0;

    uint32_t mRealFramesThisSecond =
        0;

    uint32_t mGeneratedFramesThisSecond =
        0;

    uint32_t mLastRealFps =
        0;

    uint32_t mLastGeneratedFps =
        0;

    // ============================================================
    // Frame textures
    // ============================================================

    bool mGlReady =
        false;

    bool mHaveHistory =
        false;

    GLsizei mTexWidth =
        0;

    GLsizei mTexHeight =
        0;

    GLuint mTexPrev =
        0;

    GLuint mTexCur =
        0;

    // ============================================================
    // FrameGen program
    // ============================================================

    GLuint mBlendProgram =
        0;

    GLint mBLocPos =
        -1;

    GLint mBLocUV =
        -1;

    GLint mBLocPrev =
        -1;

    GLint mBLocCur =
        -1;

    GLint mBLocMv =
        -1;

    GLint mBLocT =
        -1;

    // ============================================================
    // Overlay
    // ============================================================

    GLuint mOverlayProgram =
        0;

    GLint mOverlayPos =
        -1;

    GLint mOverlayColor =
        -1;

    bool mOverlayReady =
        false;

    // ============================================================
    // VAO
    // ============================================================

    bool mVaoSupported =
        false;

    GLuint mOwnVao =
        0;

    // ============================================================
    // Callbacks
    // ============================================================

    static void onModuleToggle(
        std::string_view moduleId,
        bool enabled
    ) {

        if (
            moduleId != kModuleId
        ) {
            return;
        }

        instance().mEnabled.store(
            enabled,
            std::memory_order_relaxed
        );

        instance().resetTiming();

        if (!enabled) {
            instance().resetAllStats();
        }
    }

    static void onConfigChanged(
        std::string_view moduleId,
        std::string_view key,
        std::string_view value
    ) {

        if (
            moduleId != kModuleId
        ) {
            return;
        }

        const std::string text(
            value
        );

        char *end =
            nullptr;

        const long parsed =
            std::strtol(
                text.c_str(),
                &end,
                10
            );

        if (
            end == text.c_str()
        ) {
            return;
        }

        if (
            key == kModeKey
        ) {

            const int newMode =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                );

            instance().mGenerationMode.store(
                newMode,
                std::memory_order_relaxed
            );

            // IMPORTANT:
            // Changing generation mode must immediately destroy
            // old timing state. This prevents old future timestamps
            // from carrying into the new mode.
            instance().resetTiming();

            instance().mHaveHistory =
                false;

            instance().resetAllStats();

        } else if (
            key == kPresetKey
        ) {

            instance().mQualityPreset.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                ),
                std::memory_order_relaxed
            );

        } else if (
            key == kFpsCounterKey
        ) {

            instance().mFpsCounter.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        1
                    )
                ),
                std::memory_order_relaxed
            );

        } else if (
            key == kFpsCapKey
        ) {

            const int cap =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        kMinFpsCap,
                        kMaxFpsCap
                    )
                );

            instance().mFpsCap.store(
                cap,
                std::memory_order_relaxed
            );

            instance().mNextCapTimeNs =
                0;
        }
    }

    static EGLBoolean swapBuffersDetour(
        EGLDisplay dpy,
        EGLSurface surface
    ) {

        return instance().handleSwapBuffers(
            dpy,
            surface
        );
    }

    // ============================================================
    // Timing reset
    // ============================================================

    void resetTiming() {

        mLastHookStartNs =
            0;

        mLastPresentedTimeNs =
            0;

        mNextCapTimeNs =
            0;
    }

    // ============================================================
    // FPS cap
    // ============================================================

    void applyFpsCap() {

        const int cap =
            std::clamp(
                mFpsCap.load(
                    std::memory_order_relaxed
                ),
                kMinFpsCap,
                kMaxFpsCap
            );

        if (cap >= kMaxFpsCap) {
            return;
        }

        const int64_t intervalNs =
            1'000'000'000LL /
            static_cast<int64_t>(
                cap
            );

        const int64_t now =
            nowNs();

        if (
            mNextCapTimeNs == 0
        ) {

            mNextCapTimeNs =
                now +
                intervalNs;

            return;
        }

        if (
            now <
            mNextCapTimeNs
        ) {

            int64_t remaining =
                mNextCapTimeNs -
                now;

            if (
                remaining >
                kSleepMarginNs
            ) {

                const int64_t sleepNs =
                    remaining -
                    kSleepMarginNs;

                std::this_thread::sleep_for(
                    std::chrono::nanoseconds(
                        sleepNs
                    )
                );
            }

            while (
                nowNs() <
                mNextCapTimeNs
            ) {

                std::this_thread::yield();
            }

            mNextCapTimeNs +=
                intervalNs;

        } else {

            // If the game stalled for a long time, do not try
            // to catch up by submitting a huge queue of frames.
            const int64_t late =
                now -
                mNextCapTimeNs;

            if (
                late >
                intervalNs * 3
            ) {

                mNextCapTimeNs =
                    now +
                    intervalNs;

            } else {

                mNextCapTimeNs +=
                    intervalNs;
            }
        }
    }

    // ============================================================
    // Statistics
    // ============================================================

    void resetAllStats() {

        mStatsWindowStartNs =
            0;

        mRealFramesThisSecond =
            0;

        mGeneratedFramesThisSecond =
            0;

        mLastRealFps =
            0;

        mLastGeneratedFps =
            0;
    }

    void updateFrameStats(
        bool generated
    ) {

        const int64_t now =
            nowNs();

        if (
            mStatsWindowStartNs == 0
        ) {

            mStatsWindowStartNs =
                now;
        }

        if (generated) {

            ++mGeneratedFramesThisSecond;

        } else {

            ++mRealFramesThisSecond;
        }

        const int64_t elapsed =
            now -
            mStatsWindowStartNs;

        if (
            elapsed <
            1'000'000'000LL
        ) {

            return;
        }

        const double seconds =
            static_cast<double>(
                elapsed
            ) /
            1'000'000'000.0;

        mLastRealFps =
            static_cast<uint32_t>(
                static_cast<double>(
                    mRealFramesThisSecond
                ) /
                seconds
            );

        mLastGeneratedFps =
            static_cast<uint32_t>(
                static_cast<double>(
                    mGeneratedFramesThisSecond
                ) /
                seconds
            );

        mRealFramesThisSecond =
            0;

        mGeneratedFramesThisSecond =
            0;

        mStatsWindowStartNs =
            now;
    }

    // ============================================================
    // Main hook
    // ============================================================

    EGLBoolean handleSwapBuffers(
        EGLDisplay dpy,
        EGLSurface surface
    ) {

        if (
            mOriginalSwapBuffers ==
            nullptr
        ) {

            return EGL_FALSE;
        }

        const int mode =
            std::clamp(
                mGenerationMode.load(
                    std::memory_order_relaxed
                ),
                0,
                2
            );

        const bool enabled =
            mEnabled.load(
                std::memory_order_relaxed
            );

        // ========================================================
        // MODE 0 / DISABLED
        //
        // IMPORTANT:
        // Absolutely no GL work happens AFTER eglSwapBuffers().
        //
        // This fixes the friend's "game doesn't respond" bug.
        // ========================================================

        if (
            !enabled ||
            mode == 0
        ) {

            resetTiming();

            applyFpsCap();

            // We can safely draw the FPS counter BEFORE swapping.
            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                updateFrameStats(
                    false
                );
            }

            return result;
        }

        // ========================================================
        // Required EGL support
        // ========================================================

        if (
            !mVaoSupported ||
            !mTrueFrameGenAvailable ||
            !hasEglExtension(
                dpy,
                "EGL_ANDROID_presentation_time"
            )
        ) {

            resetTiming();

            applyFpsCap();

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                updateFrameStats(
                    false
                );
            }

            return result;
        }

        // ========================================================
        // Surface size
        // ========================================================

        EGLint width =
            0;

        EGLint height =
            0;

        if (
            eglQuerySurface(
                dpy,
                surface,
                EGL_WIDTH,
                &width
            ) == EGL_FALSE ||

            eglQuerySurface(
                dpy,
                surface,
                EGL_HEIGHT,
                &height
            ) == EGL_FALSE ||

            width <= 0 ||
            height <= 0
        ) {

            resetTiming();

            drawFpsCounter();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        // ========================================================
        // GL resources
        // ========================================================

        if (
            !ensureGlResources(
                width,
                height
            )
        ) {

            resetTiming();

            applyFpsCap();

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                updateFrameStats(
                    false
                );
            }

            return result;
        }

        // ========================================================
        // Frame timing
        // ========================================================

        const int64_t hookNowNs =
            nowNs();

        int64_t frameIntervalNs =
            0;

        if (
            mLastHookStartNs != 0
        ) {

            frameIntervalNs =
                hookNowNs -
                mLastHookStartNs;
        }

        mLastHookStartNs =
            hookNowNs;

        // ========================================================
        // First frame
        // ========================================================

        if (
            !mHaveHistory
        ) {

            GlStateGuard stateGuard;

            captureCurrentFrame();

            // IMPORTANT:
            // Initialize BOTH history textures with the exact
            // current frame.
            //
            // The previous implementation could sample an
            // uninitialized mTexPrev here, producing black/
            // garbage areas.
            copyCurrentToPrevious();

            mHaveHistory =
                true;

            const int64_t timestamp =
                hookNowNs;

            mPresentationTimeFn(
                dpy,
                surface,
                timestamp
            );

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            // Counter MUST be before swap.
            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                mLastPresentedTimeNs =
                    timestamp;

                updateFrameStats(
                    false
                );

            } else {

                resetTiming();
                mHaveHistory =
                    false;
            }

            return result;
        }

        // ========================================================
        // Bad timing
        //
        // A large hitch usually means the player changed chunks,
        // perspective, opened UI, etc.
        //
        // Do NOT interpolate across a giant frame gap.
        // ========================================================

        if (
            frameIntervalNs <
                kMinimumUsefulIntervalNs ||
            frameIntervalNs >
                kMaxGenerationIntervalNs
        ) {

            GlStateGuard stateGuard;

            captureCurrentFrame();

            copyCurrentToPrevious();

            const int64_t timestamp =
                hookNowNs;

            mPresentationTimeFn(
                dpy,
                surface,
                timestamp
            );

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                mLastPresentedTimeNs =
                    timestamp;

                updateFrameStats(
                    false
                );

            } else {

                resetTiming();
                mHaveHistory =
                    false;
            }

            return result;
        }

        // ========================================================
        // Capture current real frame
        // ========================================================

        GlStateGuard stateGuard;

        captureCurrentFrame();

        // ========================================================
        // Motion
        //
        // Still intentionally zero because this implementation
        // does not use glReadPixels or CPU motion estimation.
        //
        // Perspective-change safety is handled by:
        //   - clean history initialization
        //   - no generation after large hitches
        //   - no stale timestamps
        //   - exact current-frame fallback
        // ========================================================

        constexpr float mvU =
            0.0f;

        constexpr float mvV =
            0.0f;

        // ========================================================
        // Generation
        // ========================================================

        const int steps =
            std::clamp(
                mode,
                1,
                2
            );

        const int64_t realStart =
            mLastPresentedTimeNs;

        int64_t targetRealTimestamp =
            realStart +
            frameIntervalNs;

        if (
            targetRealTimestamp <
            hookNowNs
        ) {

            targetRealTimestamp =
                hookNowNs;
        }

        // ========================================================
        // Generated frames
        // ========================================================

        bool generationFailed =
            false;

        for (
            int i = 1;
            i <= steps;
            ++i
        ) {

            const float t =
                static_cast<float>(i) /
                static_cast<float>(
                    steps + 1
                );

            const int64_t syntheticTimestamp =
                realStart +
                static_cast<int64_t>(
                    static_cast<double>(
                        targetRealTimestamp -
                        realStart
                    ) *
                    static_cast<double>(
                        t
                    )
                );

            // Never submit a timestamp older than the previous
            // presentation timestamp.
            if (
                syntheticTimestamp <=
                mLastPresentedTimeNs
            ) {

                continue;
            }

            // Never intentionally queue a frame far into the
            // future. This prevents the "jump after mode 0"
            // behaviour.
            const int64_t timestampNow =
                nowNs();

            if (
                syntheticTimestamp -
                timestampNow >
                frameIntervalNs
            ) {

                continue;
            }

            mPresentationTimeFn(
                dpy,
                surface,
                syntheticTimestamp
            );

            drawBlended(
                mvU,
                mvV,
                t
            );

            const EGLBoolean generatedResult =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                generatedResult != EGL_TRUE
            ) {

                generationFailed =
                    true;

                break;
            }

            mLastPresentedTimeNs =
                syntheticTimestamp;

            updateFrameStats(
                true
            );
        }

        if (
            generationFailed
        ) {

            resetTiming();

            mHaveHistory =
                false;

            return EGL_FALSE;
        }

        // ========================================================
        // Final REAL frame
        // ========================================================

        const int64_t finalTimestamp =
            std::max(
                targetRealTimestamp,
                mLastPresentedTimeNs + 1
            );

        mPresentationTimeFn(
            dpy,
            surface,
            finalTimestamp
        );

        // Exact current frame.
        //
        // This is especially important after perspective changes.
        // We never use a stale generated frame as the final frame.
        drawBlended(
            0.0f,
            0.0f,
            1.0f
        );

        // FPS overlay is drawn BEFORE the final swap.
        drawFpsCounter();

        const EGLBoolean result =
            mOriginalSwapBuffers(
                dpy,
                surface
            );

        if (
            result == EGL_TRUE
        ) {

            mLastPresentedTimeNs =
                finalTimestamp;

            updateFrameStats(
                false
            );

            // The captured current frame becomes the previous
            // history frame for the next real game frame.
            std::swap(
                mTexPrev,
                mTexCur
            );

        } else {

            resetTiming();

            mHaveHistory =
                false;
        }

        return result;
    }

    // ============================================================
    // GL resources
    // ============================================================

    bool ensureGlResources(
        GLsizei width,
        GLsizei height
    ) {

        if (
            mBlendProgram == 0 &&
            !buildBlendProgram()
        ) {

            return false;
        }

        if (
            !mOverlayReady &&
            !buildOverlayProgram()
        ) {

            getSelf().getLogger().error(
                "FPS overlay shader failed"
            );
        }

        if (
            mVaoSupported &&
            mOwnVao == 0
        ) {

            gGenVertexArrays(
                1,
                &mOwnVao
            );
        }

        if (
            mGlReady &&
            width == mTexWidth &&
            height == mTexHeight
        ) {

            return true;
        }

        // Surface resize / orientation change.
        //
        // Destroying old textures prevents stale dimensions from
        // being used after rotation or surface recreation.
        destroyTextures();

        mTexPrev =
            createFrameTexture(
                width,
                height
            );

        mTexCur =
            createFrameTexture(
                width,
                height
            );

        if (
            mTexPrev == 0 ||
            mTexCur == 0
        ) {

            destroyTextures();

            mGlReady =
                false;

            return false;
        }

        mTexWidth =
            width;

        mTexHeight =
            height;

        mGlReady =
            true;

        mHaveHistory =
            false;

        resetTiming();

        return true;
    }

    static GLuint createFrameTexture(
        GLsizei width,
        GLsizei height
    ) {

        GLuint tex =
            0;

        glGenTextures(
            1,
            &tex
        );

        if (
            tex == 0
        ) {

            return 0;
        }

        glBindTexture(
            GL_TEXTURE_2D,
            tex
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MIN_FILTER,
            GL_LINEAR
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_LINEAR
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_WRAP_S,
            GL_CLAMP_TO_EDGE
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_WRAP_T,
            GL_CLAMP_TO_EDGE
        );

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        const GLenum error =
            glGetError();

        if (
            error != GL_NO_ERROR
        ) {

            glDeleteTextures(
                1,
                &tex
            );

            return 0;
        }

        return tex;
    }

    // ============================================================
    // Capture
    // ============================================================

    void captureCurrentFrame() {

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glActiveTexture(
            GL_TEXTURE0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexCur
        );

        glCopyTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            0,
            mTexWidth,
            mTexHeight
        );
    }

    // ============================================================
    // Initialize previous texture
    // ============================================================

    void copyCurrentToPrevious() {

        if (
            mTexCur == 0 ||
            mTexPrev == 0
        ) {

            return;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        // The current frame is already in mTexCur.
        //
        // Copy it through the default framebuffer again into
        // mTexPrev so both history textures start identical.
        //
        // This is intentionally done only during initialization,
        // not every frame.
        glActiveTexture(
            GL_TEXTURE0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
        );

        glCopyTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            0,
            mTexWidth,
            mTexHeight
        );
    }

    // ============================================================
    // Draw blended frame
    // ============================================================

    void drawBlended(
        float mvU,
        float mvV,
        float t
    ) {

        if (
            mBlendProgram == 0
        ) {

            return;
        }

        if (
            mVaoSupported
        ) {

            gBindVertexArray(
                mOwnVao
            );
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glViewport(
            0,
            0,
            mTexWidth,
            mTexHeight
        );

        glDisable(
            GL_DEPTH_TEST
        );

        glDisable(
            GL_BLEND
        );

        glDisable(
            GL_CULL_FACE
        );

        glDisable(
            GL_SCISSOR_TEST
        );

        glDisable(
            GL_STENCIL_TEST
        );

        glColorMask(
            GL_TRUE,
            GL_TRUE,
            GL_TRUE,
            GL_TRUE
        );

        glUseProgram(
            mBlendProgram
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        );

        // Previous
        glActiveTexture(
            GL_TEXTURE0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
        );

        glUniform1i(
            mBLocPrev,
            0
        );

        // Current
        glActiveTexture(
            GL_TEXTURE1
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexCur
        );

        glUniform1i(
            mBLocCur,
            1
        );

        glUniform2f(
            mBLocMv,
            mvU,
            mvV
        );

        glUniform1f(
            mBLocT,
            std::clamp(
                t,
                0.0f,
                1.0f
            )
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mBLocPos
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mBLocPos
            )
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mBLocUV
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts + 2
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mBLocUV
            )
        );

        glDrawArrays(
            GL_TRIANGLE_STRIP,
            0,
            4
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mBLocPos
            )
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mBLocUV
            )
        );
    }

    // ============================================================
    // Shader creation
    // ============================================================

    bool buildBlendProgram() {

        GLuint program =
            0;

        if (
            !linkProgram(
                kVertexShaderSrc,
                kBlendFragmentShaderSrc,
                program
            )
        ) {

            return false;
        }

        mBlendProgram =
            program;

        mBLocPos =
            glGetAttribLocation(
                mBlendProgram,
                "aPos"
            );

        mBLocUV =
            glGetAttribLocation(
                mBlendProgram,
                "aUV"
            );

        mBLocPrev =
            glGetUniformLocation(
                mBlendProgram,
                "uPrev"
            );

        mBLocCur =
            glGetUniformLocation(
                mBlendProgram,
                "uCur"
            );

        mBLocMv =
            glGetUniformLocation(
                mBlendProgram,
                "uMv"
            );

        mBLocT =
            glGetUniformLocation(
                mBlendProgram,
                "uT"
            );

        return
            mBLocPos >= 0 &&
            mBLocUV >= 0 &&
            mBLocPrev >= 0 &&
            mBLocCur >= 0 &&
            mBLocMv >= 0 &&
            mBLocT >= 0;
    }

    bool buildOverlayProgram() {

        GLuint program =
            0;

        if (
            !linkProgram(
                kOverlayVertexShaderSrc,
                kOverlayFragmentShaderSrc,
                program
            )
        ) {

            return false;
        }

        mOverlayProgram =
            program;

        mOverlayPos =
            glGetAttribLocation(
                mOverlayProgram,
                "aPos"
            );

        mOverlayColor =
            glGetUniformLocation(
                mOverlayProgram,
                "uColor"
            );

        mOverlayReady =
            mOverlayPos >= 0 &&
            mOverlayColor >= 0;

        return mOverlayReady;
    }

    bool linkProgram(
        const char *vsSrc,
        const char *fsSrc,
        GLuint &outProgram
    ) {

        const GLuint vs =
            compileShader(
                GL_VERTEX_SHADER,
                vsSrc
            );

        const GLuint fs =
            compileShader(
                GL_FRAGMENT_SHADER,
                fsSrc
            );

        if (
            vs == 0 ||
            fs == 0
        ) {

            if (vs != 0) {
                glDeleteShader(vs);
            }

            if (fs != 0) {
                glDeleteShader(fs);
            }

            return false;
        }

        const GLuint program =
            glCreateProgram();

        if (
            program == 0
        ) {

            glDeleteShader(vs);
            glDeleteShader(fs);

            return false;
        }

        glAttachShader(
            program,
            vs
        );

        glAttachShader(
            program,
            fs
        );

        glBindAttribLocation(
            program,
            0,
            "aPos"
        );

        glLinkProgram(
            program
        );

        glDeleteShader(
            vs
        );

        glDeleteShader(
            fs
        );

        GLint linked =
            GL_FALSE;

        glGetProgramiv(
            program,
            GL_LINK_STATUS,
            &linked
        );

        if (
            linked == GL_FALSE
        ) {

            logProgramError(
                program
            );

            glDeleteProgram(
                program
            );

            return false;
        }

        outProgram =
            program;

        return true;
    }

    GLuint compileShader(
        GLenum type,
        const char *src
    ) {

        const GLuint shader =
            glCreateShader(type);

        if (
            shader == 0
        ) {

            return 0;
        }

        glShaderSource(
            shader,
            1,
            &src,
            nullptr
        );

        glCompileShader(
            shader
        );

        GLint compiled =
            GL_FALSE;

        glGetShaderiv(
            shader,
            GL_COMPILE_STATUS,
            &compiled
        );

        if (
            compiled == GL_FALSE
        ) {

            logShaderError(
                shader
            );

            glDeleteShader(
                shader
            );

            return 0;
        }

        return shader;
    }

    // ============================================================
    // Shader logs
    // ============================================================

    void logShaderError(
        GLuint shader
    ) {

        GLint len =
            0;

        glGetShaderiv(
            shader,
            GL_INFO_LOG_LENGTH,
            &len
        );

        std::string log(
            static_cast<size_t>(
                std::max(
                    len,
                    1
                )
            ),
            '\0'
        );

        glGetShaderInfoLog(
            shader,
            len,
            nullptr,
            log.data()
        );

        getSelf().getLogger().error(
            "Shader compile failed: {}",
            log
        );
    }

    void logProgramError(
        GLuint program
    ) {

        GLint len =
            0;

        glGetProgramiv(
            program,
            GL_INFO_LOG_LENGTH,
            &len
        );

        std::string log(
            static_cast<size_t>(
                std::max(
                    len,
                    1
                )
            ),
            '\0'
        );

        glGetProgramInfoLog(
            program,
            len,
            nullptr,
            log.data()
        );

        getSelf().getLogger().error(
            "Program link failed: {}",
            log
        );
    }

    // ============================================================
    // Seven segment digits
    // ============================================================

    static bool digitSegments(
        int digit,
        bool &s0,
        bool &s1,
        bool &s2,
        bool &s3,
        bool &s4,
        bool &s5,
        bool &s6
    ) {

        static constexpr uint8_t table[10] = {

            0b0111111,
            0b0000110,
            0b1011011,
            0b1001111,
            0b1100110,
            0b1101101,
            0b1111101,
            0b0000111,
            0b1111111,
            0b1101111
        };

        if (
            digit < 0 ||
            digit > 9
        ) {

            return false;
        }

        const uint8_t mask =
            table[digit];

        s0 = (mask & 0b0000001) != 0;
        s1 = (mask & 0b0000010) != 0;
        s2 = (mask & 0b0000100) != 0;
        s3 = (mask & 0b0001000) != 0;
        s4 = (mask & 0b0010000) != 0;
        s5 = (mask & 0b0100000) != 0;
        s6 = (mask & 0b1000000) != 0;

        return true;
    }

    static void addSegment(
        float *vertices,
        int &count,
        float x0,
        float y0,
        float x1,
        float y1
    ) {

        vertices[count++] = x0;
        vertices[count++] = y0;
        vertices[count++] = x1;
        vertices[count++] = y1;
    }

    static int buildDigit(
        int digit,
        float x,
        float y,
        float w,
        float h,
        float *vertices
    ) {

        bool s0;
        bool s1;
        bool s2;
        bool s3;
        bool s4;
        bool s5;
        bool s6;

        if (
            !digitSegments(
                digit,
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6
            )
        ) {

            return 0;
        }

        int count =
            0;

        const float left =
            x;

        const float right =
            x + w;

        const float top =
            y;

        const float mid =
            y - h * 0.5f;

        const float bottom =
            y - h;

        if (s0) {
            addSegment(
                vertices,
                count,
                left,
                top,
                right,
                top
            );
        }

        if (s1) {
            addSegment(
                vertices,
                count,
                right,
                top,
                right,
                mid
            );
        }

        if (s2) {
            addSegment(
                vertices,
                count,
                right,
                mid,
                right,
                bottom
            );
        }

        if (s3) {
            addSegment(
                vertices,
                count,
                left,
                bottom,
                right,
                bottom
            );
        }

        if (s4) {
            addSegment(
                vertices,
                count,
                left,
                mid,
                left,
                bottom
            );
        }

        if (s5) {
            addSegment(
                vertices,
                count,
                left,
                top,
                left,
                mid
            );
        }

        if (s6) {
            addSegment(
                vertices,
                count,
                left,
                mid,
                right,
                mid
            );
        }

        return count;
    }

    static void drawNumber(
        int value,
        float startX,
        float startY,
        float digitWidth,
        float digitHeight,
        float spacing,
        float *vertices,
        int &vertexFloatCount
    ) {

        value =
            std::clamp(
                value,
                0,
                9999
            );

        char buffer[8];

        std::snprintf(
            buffer,
            sizeof(buffer),
            "%d",
            value
        );

        const size_t length =
            std::strlen(buffer);

        float x =
            startX;

        for (
            size_t i = 0;
            i < length;
            ++i
        ) {

            const int digit =
                buffer[i] - '0';

            vertexFloatCount +=
                buildDigit(
                    digit,
                    x,
                    startY,
                    digitWidth,
                    digitHeight,
                    vertices +
                        vertexFloatCount
                );

            x +=
                digitWidth +
                spacing;
        }
    }

    // ============================================================
    // FPS overlay
    // ============================================================

    void drawFpsCounter() {

        if (
            mFpsCounter.load(
                std::memory_order_relaxed
            ) == 0
        ) {

            return;
        }

        if (
            !mOverlayReady ||
            mOverlayProgram == 0 ||
            mTexWidth <= 0 ||
            mTexHeight <= 0
        ) {

            return;
        }

        float vertices[512];

        int count =
            0;

        drawNumber(
            static_cast<int>(
                mLastRealFps
            ),
            -0.94f,
            0.92f,
            0.035f,
            0.070f,
            0.014f,
            vertices,
            count
        );

        addSegment(
            vertices,
            count,
            -0.94f,
            0.82f,
            -0.90f,
            0.82f
        );

        drawNumber(
            static_cast<int>(
                mLastGeneratedFps
            ),
            -0.94f,
            0.76f,
            0.035f,
            0.070f,
            0.014f,
            vertices,
            count
        );

        if (
            count <= 0
        ) {

            return;
        }

        GlStateGuard guard;

        if (
            mVaoSupported
        ) {

            gBindVertexArray(
                mOwnVao
            );
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glDisable(
            GL_DEPTH_TEST
        );

        glDisable(
            GL_CULL_FACE
        );

        glDisable(
            GL_SCISSOR_TEST
        );

        glDisable(
            GL_STENCIL_TEST
        );

        glDisable(
            GL_BLEND
        );

        glUseProgram(
            mOverlayProgram
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        );

        glUniform4f(
            mOverlayColor,
            1.0f,
            1.0f,
            1.0f,
            1.0f
        );

        glLineWidth(
            3.0f
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mOverlayPos
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            2 * sizeof(float),
            vertices
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mOverlayPos
            )
        );

        glDrawArrays(
            GL_LINES,
            0,
            count / 2
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mOverlayPos
            )
        );
    }

    // ============================================================
    // Cleanup
    // ============================================================

    void destroyTextures() {

        if (
            mTexPrev != 0
        ) {

            glDeleteTextures(
                1,
                &mTexPrev
            );

            mTexPrev =
                0;
        }

        if (
            mTexCur != 0
        ) {

            glDeleteTextures(
                1,
                &mTexCur
            );

            mTexCur =
                0;
        }
    }

    void destroyGlResources() {

        destroyTextures();

        if (
            mBlendProgram != 0
        ) {

            glDeleteProgram(
                mBlendProgram
            );

            mBlendProgram =
                0;
        }

        if (
            mOverlayProgram != 0
        ) {

            glDeleteProgram(
                mOverlayProgram
            );

            mOverlayProgram =
                0;
        }

        if (
            mOwnVao != 0 &&
            gDeleteVertexArrays != nullptr
        ) {

            gDeleteVertexArrays(
                1,
                &mOwnVao
            );

            mOwnVao =
                0;
        }

        mOverlayReady =
            false;

        mGlReady =
            false;

        mHaveHistory =
            false;

        mTexWidth =
            0;

        mTexHeight =
            0;

        resetTiming();
    }
};

} // namespace

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
)
