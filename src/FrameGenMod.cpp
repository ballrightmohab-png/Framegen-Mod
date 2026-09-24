// FrameGenMod.cpp
//
// FrameGen with:
//   - REAL FPS counter
//   - GENERATED frame counter
//   - FPS Counter toggle
//   - UI Safe Mode
//   - No glReadPixels
//
// Generation Mode:
//   0 = Off
//   1 = 1 generated frame
//   2 = 2 generated frames
//
// UI Safe Mode:
//   ON  = generated frames use the current frame directly.
//          This prevents UI ghosting/jitter.
//
//   OFF = normal temporal blending.
//
// IMPORTANT:
// This implementation does NOT use glReadPixels.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

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

constexpr const char *kUiSafeKey =
    "uiSafeMode";

constexpr int64_t kMinFrameIntervalNs =
    2'000'000LL;

constexpr int64_t kMaxFrameIntervalNs =
    80'000'000LL;

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
// Blend vertex shader
// ============================================================

constexpr const char *kVertexShaderSrc = R"glsl(
attribute vec2 aPos;
attribute vec2 aUV;

varying vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

// ============================================================
// Blend fragment shader
// ============================================================

constexpr const char *kBlendFragmentShaderSrc = R"glsl(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform vec2 uMv;
uniform float uT;

void main() {

    vec2 uvPrev =
        vUV - uT * uMv;

    vec2 uvCur =
        vUV + (1.0 - uT) * uMv;

    uvPrev =
        clamp(
            uvPrev,
            0.0,
            1.0
        );

    uvCur =
        clamp(
            uvCur,
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
            uT
        );
}
)glsl";

// ============================================================
// HUD vertex shader
// ============================================================

constexpr const char *kHudVertexShaderSrc = R"glsl(
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

// ============================================================
// HUD fragment shader
// ============================================================

constexpr const char *kHudFragmentShaderSrc = R"glsl(
precision mediump float;

uniform vec4 uColor;

void main() {
    gl_FragColor = uColor;
}
)glsl";

// ============================================================
// EGL / VAO function types
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
    gGenVertexArrays = nullptr;

PFN_glBindVertexArray
    gBindVertexArray = nullptr;

PFN_glDeleteVertexArrays
    gDeleteVertexArrays = nullptr;

// ============================================================
// EGL extension check
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
// GL state guard
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

        if (
            target == nullptr
        ) {

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
            >(
                presTimeSym
            );

        mTrueFrameGenAvailable =
            mPresentationTimeFn != nullptr;

        // --------------------------------------------------------
        // VAO support
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // Menu
        // --------------------------------------------------------

        const bool moduleRegistered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator"
            )
                .modId(
                    getSelf().getId()
                )
                .description(
                    "FrameGen with FPS counter and UI safe mode."
                )
                .defaultEnabled(true)
                .onToggle(
                    onModuleToggle
                )

                .config(
                    kModeKey,
                    "Generation Mode",
                    ConfigType::SliderInt,
                    std::to_string(
                        mGenerationMode.load()
                    ),
                    "0",
                    "2"
                )

                .config(
                    kPresetKey,
                    "Quality Preset",
                    ConfigType::SliderInt,
                    std::to_string(
                        mQualityPreset.load()
                    ),
                    "0",
                    "2"
                )

                .config(
                    kFpsCounterKey,
                    "FPS Counter",
                    ConfigType::Switch,
                    "1"
                )

                .config(
                    kUiSafeKey,
                    "UI Safe Mode",
                    ConfigType::Switch,
                    "1"
                )

                .onConfigChanged(
                    onConfigChanged
                )
                .registerModule();

        if (
            !moduleRegistered
        ) {

            getSelf().getLogger().error(
                "Failed to register Frame Generator menu"
            );
        }

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

        resetAllStats();

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

    pl::memory::HookHandle mSwapHook;

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

    std::atomic_bool mFpsCounter{
        true
    };

    std::atomic_bool mUiSafeMode{
        true
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
    // GL resources
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
    // HUD
    // ============================================================

    GLuint mHudProgram =
        0;

    GLint mHudLocPos =
        -1;

    GLint mHudLocColor =
        -1;

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

        if (!enabled) {

            instance().resetTiming();

            instance().resetAllStats();
        }
    }

    static bool parseBool(
        std::string_view value
    ) {

        return
            value == "1" ||
            value == "true" ||
            value == "True" ||
            value == "on" ||
            value == "ON";
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

        if (
            key == kFpsCounterKey
        ) {

            instance().mFpsCounter.store(
                parseBool(value),
                std::memory_order_relaxed
            );

            return;
        }

        if (
            key == kUiSafeKey
        ) {

            instance().mUiSafeMode.store(
                parseBool(value),
                std::memory_order_relaxed
            );

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

            instance().mGenerationMode.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                ),
                std::memory_order_relaxed
            );

            instance().resetTiming();
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
    // Time
    // ============================================================

    static int64_t nowNs() {

        return std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            std::chrono::steady_clock::now()
                .time_since_epoch()
        ).count();
    }

    void resetTiming() {

        mHaveHistory =
            false;

        mLastHookStartNs =
            0;

        mLastPresentedTimeNs =
            0;
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

        getSelf().getLogger().info(
            "FrameGen stats - REAL FPS: {} | GENERATED: {}",
            mLastRealFps,
            mLastGeneratedFps
        );

        mRealFramesThisSecond =
            0;

        mGeneratedFramesThisSecond =
            0;

        mStatsWindowStartNs =
            now;
    }

    // ============================================================
    // Main swap
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
            mGenerationMode.load(
                std::memory_order_relaxed
            );

        if (
            !mEnabled.load(
                std::memory_order_relaxed
            ) ||
            mode == 0
        ) {

            resetTiming();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        if (
            !mVaoSupported ||
            !mTrueFrameGenAvailable ||
            !hasEglExtension(
                dpy,
                "EGL_ANDROID_presentation_time"
            )
        ) {

            resetTiming();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

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

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        if (
            !ensureGlResources(
                width,
                height
            )
        ) {

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

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

        GlStateGuard stateGuard;

        captureCurrentFrame();

        // ========================================================
        // First real frame
        // ========================================================

        if (
            !mHaveHistory
        ) {

            const int64_t timestamp =
                std::max(
                    hookNowNs,
                    mLastPresentedTimeNs + 1
                );

            mPresentationTimeFn(
                dpy,
                surface,
                timestamp
            );

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                )
            ) {

                drawHud(
                    mLastRealFps,
                    mLastGeneratedFps
                );
            }

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

                mHaveHistory =
                    true;

                updateFrameStats(
                    false
                );

                std::swap(
                    mTexPrev,
                    mTexCur
                );

            } else {

                resetTiming();
            }

            return result;
        }

        // ========================================================
        // Bad interval
        // ========================================================

        if (
            frameIntervalNs <
                kMinFrameIntervalNs ||
            frameIntervalNs >
                kMaxFrameIntervalNs
        ) {

            const int64_t timestamp =
                std::max(
                    hookNowNs,
                    mLastPresentedTimeNs + 1
                );

            mPresentationTimeFn(
                dpy,
                surface,
                timestamp
            );

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                )
            ) {

                drawHud(
                    mLastRealFps,
                    mLastGeneratedFps
                );
            }

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

                std::swap(
                    mTexPrev,
                    mTexCur
                );

            } else {

                resetTiming();
            }

            return result;
        }

        // ========================================================
        // Generated frames
        // ========================================================

        const int steps =
            std::clamp(
                mode,
                1,
                2
            );

        int64_t targetRealTimestamp =
            mLastPresentedTimeNs +
            frameIntervalNs;

        if (
            targetRealTimestamp <
            hookNowNs
        ) {

            targetRealTimestamp =
                hookNowNs;
        }

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
                mLastPresentedTimeNs +
                static_cast<int64_t>(
                    static_cast<double>(
                        targetRealTimestamp -
                        mLastPresentedTimeNs
                    ) *
                    static_cast<double>(
                        t
                    )
                );

            const int64_t currentTimeNs =
                nowNs();

            if (
                syntheticTimestamp <=
                currentTimeNs
            ) {
                continue;
            }

            if (
                syntheticTimestamp <=
                mLastPresentedTimeNs
            ) {
                continue;
            }

            mPresentationTimeFn(
                dpy,
                surface,
                syntheticTimestamp
            );

            // ----------------------------------------------------
            // UI SAFE MODE
            //
            // Instead of mixing two UI states, draw the current
            // frame directly.
            //
            // This removes the UI ghosting/jitter.
            // ----------------------------------------------------

            if (
                mUiSafeMode.load(
                    std::memory_order_relaxed
                )
            ) {

                drawCurrentFrame();

            } else {

                drawBlended(
                    0.0f,
                    0.0f,
                    t
                );
            }

            // ----------------------------------------------------
            // Counter is drawn AFTER the generated frame.
            //
            // This keeps the counter itself stable instead of
            // blending it between frames.
            // ----------------------------------------------------

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                )
            ) {

                drawHud(
                    mLastRealFps,
                    mLastGeneratedFps
                );
            }

            const EGLBoolean generatedResult =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                generatedResult != EGL_TRUE
            ) {

                resetTiming();

                return generatedResult;
            }

            mLastPresentedTimeNs =
                syntheticTimestamp;

            updateFrameStats(
                true
            );
        }

        // ========================================================
        // Final real frame
        // ========================================================

        const int64_t finalTimestamp =
            std::max(
                targetRealTimestamp,
                std::max(
                    nowNs(),
                    mLastPresentedTimeNs + 1
                )
            );

        mPresentationTimeFn(
            dpy,
            surface,
            finalTimestamp
        );

        drawCurrentFrame();

        if (
            mFpsCounter.load(
                std::memory_order_relaxed
            )
        ) {

            drawHud(
                mLastRealFps,
                mLastGeneratedFps
            );
        }

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

            std::swap(
                mTexPrev,
                mTexCur
            );

        } else {

            resetTiming();
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
            mHudProgram == 0 &&
            !buildHudProgram()
        ) {

            return false;
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
    // Draw current frame
    // ============================================================

    void drawCurrentFrame() {

        drawBlended(
            0.0f,
            0.0f,
            1.0f
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
    // HUD shader
    // ============================================================

    bool buildHudProgram() {

        GLuint program =
            0;

        if (
            !linkProgram(
                kHudVertexShaderSrc,
                kHudFragmentShaderSrc,
                program
            )
        ) {

            return false;
        }

        mHudProgram =
            program;

        mHudLocPos =
            glGetAttribLocation(
                mHudProgram,
                "aPos"
            );

        mHudLocColor =
            glGetUniformLocation(
                mHudProgram,
                "uColor"
            );

        return
            mHudLocPos >= 0 &&
            mHudLocColor >= 0;
    }

    // ============================================================
    // Seven-segment renderer
    // ============================================================

    //
    // Segment layout:
    //
    //      --- A ---
    //     |         |
    //     F         B
    //     |         |
    //      --- G ---
    //     |         |
    //     E         C
    //     |         |
    //      --- D ---
    //
    // We use line segments instead of textures.
    //

    static bool digitSegment(
        char c,
        int segment
    ) {

        switch (c) {

            case '0':
                return
                    segment == 0 ||
                    segment == 1 ||
                    segment == 2 ||
                    segment == 3 ||
                    segment == 4 ||
                    segment == 5;

            case '1':
                return
                    segment == 1 ||
                    segment == 2;

            case '2':
                return
                    segment == 0 ||
                    segment == 1 ||
                    segment == 6 ||
                    segment == 4 ||
                    segment == 3;

            case '3':
                return
                    segment == 0 ||
                    segment == 1 ||
                    segment == 2 ||
                    segment == 3 ||
                    segment == 6;

            case '4':
                return
                    segment == 5 ||
                    segment == 6 ||
                    segment == 1 ||
                    segment == 2;

            case '5':
                return
                    segment == 0 ||
                    segment == 5 ||
                    segment == 6 ||
                    segment == 2 ||
                    segment == 3;

            case '6':
                return
                    segment == 0 ||
                    segment == 5 ||
                    segment == 6 ||
                    segment == 4 ||
                    segment == 3 ||
                    segment == 2;

            case '7':
                return
                    segment == 0 ||
                    segment == 1 ||
                    segment == 2;

            case '8':
                return true;

            case '9':
                return
                    segment == 0 ||
                    segment == 1 ||
                    segment == 2 ||
                    segment == 3 ||
                    segment == 5 ||
                    segment == 6;

            default:
                return false;
        }
    }

    static bool letterSegment(
        char c,
        int segment
    ) {

        // F
        if (c == 'F') {

            return
                segment == 0 ||
                segment == 5 ||
                segment == 6;
        }

        // P
        if (c == 'P') {

            return
                segment == 0 ||
                segment == 1 ||
                segment == 5 ||
                segment == 6;
        }

        // S
        if (c == 'S') {

            return
                segment == 0 ||
                segment == 5 ||
                segment == 6 ||
                segment == 2 ||
                segment == 3;
        }

        // G
        if (c == 'G') {

            return
                segment == 0 ||
                segment == 5 ||
                segment == 6 ||
                segment == 2 ||
                segment == 3 ||
                segment == 4;
        }

        // M - approximate
        if (c == 'M') {

            return
                segment == 5 ||
                segment == 1 ||
                segment == 4 ||
                segment == 2;
        }

        // A
        if (c == 'A') {

            return
                segment == 0 ||
                segment == 1 ||
                segment == 2 ||
                segment == 5 ||
                segment == 6;
        }

        // E
        if (c == 'E') {

            return
                segment == 0 ||
                segment == 5 ||
                segment == 6 ||
                segment == 4 ||
                segment == 3;
        }

        // N - approximate
        if (c == 'N') {

            return
                segment == 5 ||
                segment == 1 ||
                segment == 4 ||
                segment == 2;
        }

        // R
        if (c == 'R') {

            return
                segment == 0 ||
                segment == 1 ||
                segment == 5 ||
                segment == 6 ||
                segment == 4;
        }

        return false;
    }

    void addSegment(
        float *verts,
        int &count,
        float x,
        float y,
        float w,
        float h,
        int segment
    ) {

        float x1 = x;
        float y1 = y;

        float x2 = x;
        float y2 = y;

        const float midY =
            y + h * 0.5f;

        switch (segment) {

            case 0:
                x1 = x + w * 0.2f;
                y1 = y + h;

                x2 = x + w * 0.8f;
                y2 = y + h;
                break;

            case 1:
                x1 = x + w;
                y1 = midY + h * 0.25f;

                x2 = x + w;
                y2 = y + h * 0.9f;
                break;

            case 2:
                x1 = x + w;
                y1 = y + h * 0.1f;

                x2 = x + w;
                y2 = midY - h * 0.25f;
                break;

            case 3:
                x1 = x + w * 0.2f;
                y1 = y;

                x2 = x + w * 0.8f;
                y2 = y;
                break;

            case 4:
                x1 = x;
                y1 = y + h * 0.1f;

                x2 = x;
                y2 = midY - h * 0.25f;
                break;

            case 5:
                x1 = x;
                y1 = midY + h * 0.25f;

                x2 = x;
                y2 = y + h * 0.9f;
                break;

            case 6:
                x1 = x + w * 0.2f;
                y1 = midY;

                x2 = x + w * 0.8f;
                y2 = midY;
                break;

            default:
                return;
        }

        // Convert pixel coordinates into NDC.
        const float ndcX1 =
            (x1 /
                static_cast<float>(mTexWidth))
            * 2.0f - 1.0f;

        const float ndcY1 =
            1.0f -
            (y1 /
                static_cast<float>(mTexHeight))
            * 2.0f;

        const float ndcX2 =
            (x2 /
                static_cast<float>(mTexWidth))
            * 2.0f - 1.0f;

        const float ndcY2 =
            1.0f -
            (y2 /
                static_cast<float>(mTexHeight))
            * 2.0f;

        verts[count++] =
            ndcX1;

        verts[count++] =
            ndcY1;

        verts[count++] =
            ndcX2;

        verts[count++] =
            ndcY2;
    }

    void addCharacter(
        float *verts,
        int &count,
        char c,
        float x,
        float y,
        float w,
        float h
    ) {

        for (
            int segment = 0;
            segment < 7;
            ++segment
        ) {

            bool enabled =
                digitSegment(
                    c,
                    segment
                ) ||
                letterSegment(
                    c,
                    segment
                );

            if (
                enabled
            ) {

                addSegment(
                    verts,
                    count,
                    x,
                    y,
                    w,
                    h,
                    segment
                );
            }
        }
    }

    // ============================================================
    // Draw FPS HUD
    // ============================================================

    void drawHud(
        uint32_t realFps,
        uint32_t generatedFps
    ) {

        if (
            mHudProgram == 0 ||
            mTexWidth <= 0 ||
            mTexHeight <= 0
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
            GL_CULL_FACE
        );

        glDisable(
            GL_SCISSOR_TEST
        );

        glDisable(
            GL_STENCIL_TEST
        );

        // Alpha blending lets the counter sit cleanly over
        // Minecraft without blocking the entire background.
        glEnable(
            GL_BLEND
        );

        glBlendFunc(
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA
        );

        glUseProgram(
            mHudProgram
        );

        // --------------------------------------------------------
        // Text:
        //
        // FPS 90
        // GEN 90
        //
        // --------------------------------------------------------

        char realDigits[16];

        char genDigits[16];

        std::snprintf(
            realDigits,
            sizeof(realDigits),
            "%u",
            realFps
        );

        std::snprintf(
            genDigits,
            sizeof(genDigits),
            "%u",
            generatedFps
        );

        // Keep the HUD reasonably small on the A16.
        const float h =
            std::max(
                18.0f,
                static_cast<float>(
                    mTexHeight
                ) * 0.025f
            );

        const float w =
            h * 0.62f;

        const float gap =
            h * 0.16f;

        const float startX =
            16.0f;

        const float startY =
            18.0f;

        // Maximum buffer:
        // 2 lines x ~12 chars x 7 segments x 4 floats
        float verts[1024];

        int count =
            0;

        auto addText =
            [&](const char *label,
                const char *number,
                float y) {

                float x =
                    startX;

                // label
                for (
                    size_t i = 0;
                    label[i] != '\0';
                    ++i
                ) {

                    addCharacter(
                        verts,
                        count,
                        label[i],
                        x,
                        y,
                        w,
                        h
                    );

                    x +=
                        w + gap;
                }

                // number
                for (
                    size_t i = 0;
                    number[i] != '\0';
                    ++i
                ) {

                    addCharacter(
                        verts,
                        count,
                        number[i],
                        x,
                        y,
                        w,
                        h
                    );

                    x +=
                        w + gap;
                }
            };

        addText(
            "FPS",
            realDigits,
            startY
        );

        addText(
            "GEN",
            genDigits,
            startY +
                h +
                gap * 2.0f
        );

        if (
            count <= 0
        ) {
            return;
        }

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        );

        glUniform4f(
            mHudLocColor,
            1.0f,
            1.0f,
            1.0f,
            0.95f
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mHudLocPos
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            0,
            verts
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mHudLocPos
            )
        );

        glLineWidth(
            2.0f
        );

        glDrawArrays(
            GL_LINES,
            0,
            count / 2
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mHudLocPos
            )
        );

        glDisable(
            GL_BLEND
        );
    }

    // ============================================================
    // Shader program
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

        return true;
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

        glBindAttribLocation(
            program,
            1,
            "aUV"
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
    // Texture cleanup
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

    // ============================================================
    // GL cleanup
    // ============================================================

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
            mHudProgram != 0
        ) {

            glDeleteProgram(
                mHudProgram
            );

            mHudProgram =
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
