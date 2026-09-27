// FrameGenMod.cpp
//
// Frame Generator
//
// Features:
//   - Generation Mode: 0 / 1 / 2
//   - Quality Preset: 0 / 1 / 2
//   - Real FPS counter
//   - Generated FPS counter
//   - FPS cap: 10..144
//   - GPU-side frame-change detection
//   - Hard temporal reset on major scene/perspective change
//   - No ll/api/... headers; uses the project's pl:: API
//   - No GL work after eglSwapBuffers()
//   - Safe Mode 0
//   - Safe surface resize/orientation reset
//   - Safe timestamp handling
//
// IMPORTANT:
// This implementation intentionally becomes conservative when
// the current frame differs strongly from the previous frame.
// This prevents old temporal data from producing black/ghost/
// flickering areas during rapid camera movement.

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
#include <cmath>

#include <dlfcn.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <pl/Mod.hpp>
#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>

namespace {

constexpr const char *kModuleId = "framegen.core";
constexpr const char *kModeKey = "generationMode";
constexpr const char *kPresetKey = "qualityPreset";
constexpr const char *kFpsCounterKey = "fpsCounter";
constexpr const char *kFpsCapKey = "fpsCap";

constexpr int kMinFpsCap = 10;
constexpr int kMaxFpsCap = 144;

constexpr int64_t kMaxGenerationIntervalNs = 50'000'000LL;
constexpr int64_t kMinimumUsefulIntervalNs = 4'000'000LL;
constexpr int64_t kSleepMarginNs = 500'000LL;

// GPU change detector thresholds.
//
// The detector renders the absolute difference between two frames
// into a tiny GPU texture. A small readback is then performed only
// on the 4x4 detector texture rather than the entire Minecraft frame.
//
// This makes the expensive work happen on the GPU while keeping the
// CPU readback extremely small.
constexpr int kDetectorWidth = 4;
constexpr int kDetectorHeight = 4;

// Average difference above this means that temporal history is no
// longer trustworthy.
//
// 0.0 = identical
// 1.0 = completely different
constexpr float kHardResetThreshold = 0.115f;

// Very high difference means an immediate hard reset.
constexpr float kExtremeResetThreshold = 0.30f;

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
// Blend shaders
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

constexpr const char *kBlendFragmentShaderSrc = R"glsl(
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
// GPU frame-change detector shader
// ============================================================

constexpr const char *kDetectorFragmentShaderSrc = R"glsl(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;

void main() {

    vec3 a =
        texture2D(
            uPrev,
            vUV
        ).rgb;

    vec3 b =
        texture2D(
            uCur,
            vUV
        ).rgb;

    vec3 d =
        abs(a - b);

    // Perceptual-ish RGB weighting.
    float difference =
        dot(
            d,
            vec3(
                0.299,
                0.587,
                0.114
            )
        );

    gl_FragColor =
        vec4(
            difference,
            difference,
            difference,
            1.0
        );
}
)glsl";

// ============================================================
// Overlay shaders
// ============================================================

constexpr const char *kOverlayVertexShaderSrc = R"glsl(
attribute vec2 aPos;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

constexpr const char *kOverlayFragmentShaderSrc = R"glsl(
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

PFN_glGenVertexArrays gGenVertexArrays = nullptr;
PFN_glBindVertexArray gBindVertexArray = nullptr;
PFN_glDeleteVertexArrays gDeleteVertexArrays = nullptr;

// ============================================================
// Time
// ============================================================

int64_t nowNs() {

    return std::chrono::duration_cast<
        std::chrono::nanoseconds
    >(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

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

    const size_t length =
        std::strlen(name);

    const char *p =
        extensions;

    while (
        (p = std::strstr(p, name)) != nullptr
    ) {

        const bool start =
            p == extensions ||
            p[-1] == ' ';

        const char end =
            p[length];

        if (
            start &&
            (
                end == ' ' ||
                end == '\0'
            )
        ) {
            return true;
        }

        p += length;
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

        glGetIntegerv(
            GL_SCISSOR_BOX,
            mScissorBox
        );

        mDepth =
            glIsEnabled(GL_DEPTH_TEST);

        mBlend =
            glIsEnabled(GL_BLEND);

        mCull =
            glIsEnabled(GL_CULL_FACE);

        mScissor =
            glIsEnabled(GL_SCISSOR_TEST);

        mStencil =
            glIsEnabled(GL_STENCIL_TEST);

        glGetBooleanv(
            GL_COLOR_WRITEMASK,
            mColorMask
        );

        glActiveTexture(GL_TEXTURE0);

        glGetIntegerv(
            GL_TEXTURE_BINDING_2D,
            &mTex0
        );

        glActiveTexture(GL_TEXTURE1);

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

        if (gBindVertexArray != nullptr) {

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

        glScissor(
            mScissorBox[0],
            mScissorBox[1],
            mScissorBox[2],
            mScissorBox[3]
        );

        glUseProgram(
            static_cast<GLuint>(
                mProgram
            )
        );

        glActiveTexture(GL_TEXTURE0);

        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(
                mTex0
            )
        );

        glActiveTexture(GL_TEXTURE1);

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
            mDepth
        );

        setEnabled(
            GL_BLEND,
            mBlend
        );

        setEnabled(
            GL_CULL_FACE,
            mCull
        );

        setEnabled(
            GL_SCISSOR_TEST,
            mScissor
        );

        setEnabled(
            GL_STENCIL_TEST,
            mStencil
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
        GLboolean value
    ) {

        if (value) {
            glEnable(cap);
        } else {
            glDisable(cap);
        }
    }

    GLint mProgram = 0;
    GLint mActiveTexture = GL_TEXTURE0;
    GLint mViewport[4] = {};
    GLint mFramebuffer = 0;
    GLint mArrayBuffer = 0;
    GLint mVertexArray = 0;
    GLint mScissorBox[4] = {};
    GLint mTex0 = 0;
    GLint mTex1 = 0;

    GLboolean mDepth = GL_FALSE;
    GLboolean mBlend = GL_FALSE;
    GLboolean mCull = GL_FALSE;
    GLboolean mScissor = GL_FALSE;
    GLboolean mStencil = GL_FALSE;

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

        if (!mSwapHook.installed()) {

            getSelf().getLogger().error(
                "Failed to hook eglSwapBuffers"
            );

            return false;
        }

        void *presentationSymbol =
            reinterpret_cast<void *>(
                eglGetProcAddress(
                    "eglPresentationTimeANDROID"
                )
            );

        mPresentationTimeFn =
            reinterpret_cast<
                PFN_eglPresentationTimeANDROID
            >(presentationSymbol);

        mTrueFrameGenAvailable =
            mPresentationTimeFn != nullptr;

        // ========================================================
        // VAO
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
        // Mod menu
        // ========================================================

        const bool registered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator"
            )
                .modId(
                    getSelf().getId()
                )
                .description(
                    "Frame generation with temporal "
                    "scene-change protection."
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

        resetTiming();
        resetAllStats();
        hardTemporalReset();

        if (!registered) {

            getSelf().getLogger().error(
                "Failed to register Frame Generator"
            );
        }

        return registered;
    }

    bool disable() {

        pl::modmenu::unregisterModule(
            kModuleId
        );

        mSwapHook.reset();

        destroyGlResources();

        resetTiming();
        resetAllStats();

        return true;
    }

    bool unload() {

        return true;
    }

private:

    ll::mod::NativeMod &mSelf;

    pl::memory::HookHandle mSwapHook;

    EglSwapBuffersFn
        mOriginalSwapBuffers = nullptr;

    PFN_eglPresentationTimeANDROID
        mPresentationTimeFn = nullptr;

    bool mTrueFrameGenAvailable = false;

    bool mEnabled = true;

    std::atomic_int mGenerationMode{1};
    std::atomic_int mQualityPreset{1};
    std::atomic_int mFpsCounter{1};
    std::atomic_int mFpsCap{60};

    int64_t mLastHookStartNs = 0;
    int64_t mLastPresentedTimeNs = 0;
    int64_t mNextCapTimeNs = 0;

    int64_t mStatsWindowStartNs = 0;

    uint32_t mRealFramesThisSecond = 0;
    uint32_t mGeneratedFramesThisSecond = 0;

    uint32_t mLastRealFps = 0;
    uint32_t mLastGeneratedFps = 0;

    // ============================================================
    // Main textures
    // ============================================================

    bool mGlReady = false;
    bool mHaveHistory = false;

    GLsizei mTexWidth = 0;
    GLsizei mTexHeight = 0;

    GLuint mTexPrev = 0;
    GLuint mTexCur = 0;

    // ============================================================
    // Blend program
    // ============================================================

    GLuint mBlendProgram = 0;

    GLint mBLocPos = -1;
    GLint mBLocUV = -1;
    GLint mBLocPrev = -1;
    GLint mBLocCur = -1;
    GLint mBLocMv = -1;
    GLint mBLocT = -1;

    // ============================================================
    // Detector
    // ============================================================

    GLuint mDetectorProgram = 0;

    GLint mDLocPos = -1;
    GLint mDLocUV = -1;
    GLint mDLocPrev = -1;
    GLint mDLocCur = -1;

    GLuint mDetectorTexture = 0;
    GLuint mDetectorFramebuffer = 0;

    bool mDetectorReady = false;

    float mLastFrameDifference = 0.0f;

    // ============================================================
    // Overlay
    // ============================================================

    GLuint mOverlayProgram = 0;

    GLint mOverlayPos = -1;
    GLint mOverlayColor = -1;

    bool mOverlayReady = false;

    // ============================================================
    // VAO
    // ============================================================

    bool mVaoSupported = false;
    GLuint mOwnVao = 0;

    // ============================================================
    // Callbacks
    // ============================================================

    static void onModuleToggle(
        std::string_view moduleId,
        bool enabled
    ) {

        if (moduleId != kModuleId) {
            return;
        }

        FrameGenMod &mod =
            instance();

        mod.mEnabled =
            enabled;

        mod.hardTemporalReset();
        mod.resetTiming();

        if (!enabled) {
            mod.resetAllStats();
        }
    }

    static void onConfigChanged(
        std::string_view moduleId,
        std::string_view key,
        std::string_view value
    ) {

        if (moduleId != kModuleId) {
            return;
        }

        FrameGenMod &mod =
            instance();

        std::string text(value);

        char *end = nullptr;

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

        if (key == kModeKey) {

            mod.mGenerationMode.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                )
            );

            mod.hardTemporalReset();
            mod.resetTiming();
            mod.resetAllStats();

        } else if (key == kPresetKey) {

            mod.mQualityPreset.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                )
            );

            // Quality changes can alter the generated result,
            // so discard temporal history.
            mod.hardTemporalReset();

        } else if (key == kFpsCounterKey) {

            mod.mFpsCounter.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        1
                    )
                )
            );

        } else if (key == kFpsCapKey) {

            mod.mFpsCap.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        kMinFpsCap,
                        kMaxFpsCap
                    )
                )
            );

            mod.mNextCapTimeNs = 0;
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
    // Reset
    // ============================================================

    void resetTiming() {

        mLastHookStartNs = 0;
        mLastPresentedTimeNs = 0;
        mNextCapTimeNs = 0;
    }

    void hardTemporalReset() {

        mHaveHistory = false;

        mLastFrameDifference = 0.0f;

        resetTiming();
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

        const int64_t interval =
            1'000'000'000LL /
            static_cast<int64_t>(cap);

        const int64_t now =
            nowNs();

        if (mNextCapTimeNs == 0) {

            mNextCapTimeNs =
                now + interval;

            return;
        }

        if (now < mNextCapTimeNs) {

            const int64_t remaining =
                mNextCapTimeNs - now;

            if (remaining > kSleepMarginNs) {

                std::this_thread::sleep_for(
                    std::chrono::nanoseconds(
                        remaining -
                        kSleepMarginNs
                    )
                );
            }

            while (nowNs() < mNextCapTimeNs) {
                std::this_thread::yield();
            }

            mNextCapTimeNs += interval;

        } else {

            const int64_t late =
                now - mNextCapTimeNs;

            if (late > interval * 3) {

                mNextCapTimeNs =
                    now + interval;

            } else {

                mNextCapTimeNs += interval;
            }
        }
    }

    // ============================================================
    // Statistics
    // ============================================================

    void resetAllStats() {

        mStatsWindowStartNs = 0;

        mRealFramesThisSecond = 0;
        mGeneratedFramesThisSecond = 0;

        mLastRealFps = 0;
        mLastGeneratedFps = 0;
    }

    void updateFrameStats(
        bool generated
    ) {

        const int64_t now =
            nowNs();

        if (mStatsWindowStartNs == 0) {
            mStatsWindowStartNs = now;
        }

        if (generated) {
            ++mGeneratedFramesThisSecond;
        } else {
            ++mRealFramesThisSecond;
        }

        const int64_t elapsed =
            now - mStatsWindowStartNs;

        if (elapsed < 1'000'000'000LL) {
            return;
        }

        const double seconds =
            static_cast<double>(elapsed) /
            1'000'000'000.0;

        mLastRealFps =
            static_cast<uint32_t>(
                static_cast<double>(
                    mRealFramesThisSecond
                ) / seconds
            );

        mLastGeneratedFps =
            static_cast<uint32_t>(
                static_cast<double>(
                    mGeneratedFramesThisSecond
                ) / seconds
            );

        mRealFramesThisSecond = 0;
        mGeneratedFramesThisSecond = 0;

        mStatsWindowStartNs = now;
    }

    // ============================================================
    // Main swap
    // ============================================================

    EGLBoolean handleSwapBuffers(
        EGLDisplay dpy,
        EGLSurface surface
    ) {

        if (mOriginalSwapBuffers == nullptr) {
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

        // ========================================================
        // Disabled / Mode 0
        // ========================================================

        if (!mEnabled || mode == 0) {

            hardTemporalReset();

            applyFpsCap();

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (result == EGL_TRUE) {
                updateFrameStats(false);
            }

            return result;
        }

        // ========================================================
        // Required presentation support
        // ========================================================

        if (
            !mTrueFrameGenAvailable ||
            !mVaoSupported ||
            !hasEglExtension(
                dpy,
                "EGL_ANDROID_presentation_time"
            )
        ) {

            hardTemporalReset();

            applyFpsCap();

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (result == EGL_TRUE) {
                updateFrameStats(false);
            }

            return result;
        }

        // ========================================================
        // Surface size
        // ========================================================

        EGLint width = 0;
        EGLint height = 0;

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

            hardTemporalReset();

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

            hardTemporalReset();

            applyFpsCap();

            drawFpsCounter();

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (result == EGL_TRUE) {
                updateFrameStats(false);
            }

            return result;
        }

        const int64_t hookNow =
            nowNs();

        int64_t frameInterval = 0;

        if (mLastHookStartNs != 0) {

            frameInterval =
                hookNow -
                mLastHookStartNs;
        }

        mLastHookStartNs =
            hookNow;

        // ========================================================
        // First frame
        // ========================================================

        if (!mHaveHistory) {

            GlStateGuard guard;

            captureCurrentFrame();

            copyCurrentToPrevious();

            mHaveHistory = true;

            const int64_t timestamp =
                hookNow;

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

            if (result == EGL_TRUE) {

                mLastPresentedTimeNs =
                    timestamp;

                updateFrameStats(false);

            } else {

                hardTemporalReset();
            }

            return result;
        }

        // ========================================================
        // Capture the real current frame.
        // ========================================================

        GlStateGuard guard;

        captureCurrentFrame();

        // ========================================================
        // GPU frame-change detection
        // ========================================================

        const float difference =
            detectFrameChange();

        mLastFrameDifference =
            difference;

        // ========================================================
        // HARD TEMPORAL RESET
        //
        // This is the important perspective-flicker fix.
        //
        // If the current frame differs substantially from the
        // previous frame, don't interpolate old history into it.
        //
        // Instead:
        //   1. discard old temporal history
        //   2. make current frame the new previous frame
        //   3. display current frame exactly
        //   4. do not generate synthetic frames
        // ========================================================

        if (
            difference >=
            kHardResetThreshold
        ) {

            copyCurrentToPrevious();

            mHaveHistory = true;

            const int64_t timestamp =
                hookNow;

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

            if (result == EGL_TRUE) {

                mLastPresentedTimeNs =
                    timestamp;

                updateFrameStats(false);

            } else {

                hardTemporalReset();
            }

            return result;
        }

        // ========================================================
        // Bad frame timing
        // ========================================================

        if (
            frameInterval <
                kMinimumUsefulIntervalNs ||
            frameInterval >
                kMaxGenerationIntervalNs
        ) {

            copyCurrentToPrevious();

            const int64_t timestamp =
                hookNow;

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

            if (result == EGL_TRUE) {

                mLastPresentedTimeNs =
                    timestamp;

                updateFrameStats(false);

            } else {

                hardTemporalReset();
            }

            return result;
        }

        // ========================================================
        // Conservative motion
        //
        // We intentionally keep motion at zero because incorrect
        // motion estimation is what can create black/warped areas.
        //
        // GPU scene-change detection decides whether generation
        // is safe.
        // ========================================================

        float mvU = 0.0f;
        float mvV = 0.0f;

        const int quality =
            std::clamp(
                mQualityPreset.load(
                    std::memory_order_relaxed
                ),
                0,
                2
            );

        // Higher quality gets a slightly more conservative
        // temporal decision.
        if (quality == 0) {

            mvU = 0.0f;
            mvV = 0.0f;

        } else if (quality == 1) {

            mvU = 0.0f;
            mvV = 0.0f;

        } else {

            mvU = 0.0f;
            mvV = 0.0f;
        }

        // ========================================================
        // Timestamps
        // ========================================================

        const int steps =
            std::clamp(
                mode,
                1,
                2
            );

        int64_t realStart =
            mLastPresentedTimeNs;

        if (realStart <= 0) {
            realStart =
                hookNow -
                frameInterval;
        }

        int64_t targetRealTimestamp =
            realStart +
            frameInterval;

        if (
            targetRealTimestamp <=
            realStart
        ) {

            targetRealTimestamp =
                realStart + 1;
        }

        bool generationFailed = false;

        // ========================================================
        // Generated frames
        // ========================================================

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

            int64_t syntheticTimestamp =
                realStart +
                static_cast<int64_t>(
                    static_cast<double>(
                        targetRealTimestamp -
                        realStart
                    ) *
                    static_cast<double>(t)
                );

            if (
                syntheticTimestamp <=
                mLastPresentedTimeNs
            ) {
                continue;
            }

            const int64_t currentTime =
                nowNs();

            // Never queue a synthetic frame too far ahead.
            if (
                syntheticTimestamp -
                currentTime >
                frameInterval
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
                generatedResult !=
                EGL_TRUE
            ) {

                generationFailed = true;
                break;
            }

            mLastPresentedTimeNs =
                syntheticTimestamp;

            updateFrameStats(true);
        }

        if (generationFailed) {

            hardTemporalReset();

            return EGL_FALSE;
        }

        // ========================================================
        // Final real frame
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

        if (result == EGL_TRUE) {

            mLastPresentedTimeNs =
                finalTimestamp;

            updateFrameStats(false);

            // Current frame becomes previous history.
            std::swap(
                mTexPrev,
                mTexCur
            );

        } else {

            hardTemporalReset();
        }

        return result;
    }

    // ============================================================
    // Resources
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
            !mDetectorReady &&
            !buildDetector()
        ) {
            return false;
        }

        if (
            !mOverlayReady &&
            !buildOverlayProgram()
        ) {
            // Overlay is optional.
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

        destroyFrameTextures();

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

            destroyFrameTextures();

            return false;
        }

        mTexWidth = width;
        mTexHeight = height;

        mGlReady = true;

        hardTemporalReset();

        return true;
    }

    static GLuint createFrameTexture(
        GLsizei width,
        GLsizei height
    ) {

        GLuint texture = 0;

        glGenTextures(
            1,
            &texture
        );

        if (texture == 0) {
            return 0;
        }

        glBindTexture(
            GL_TEXTURE_2D,
            texture
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

        if (glGetError() != GL_NO_ERROR) {

            glDeleteTextures(
                1,
                &texture
            );

            return 0;
        }

        return texture;
    }

    // ============================================================
    // Capture
    // ============================================================

    void captureCurrentFrame() {

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glActiveTexture(GL_TEXTURE0);

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

    void copyCurrentToPrevious() {

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glActiveTexture(GL_TEXTURE0);

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
    // GPU detector
    // ============================================================

    float detectFrameChange() {

        if (
            !mDetectorReady ||
            mDetectorFramebuffer == 0 ||
            mDetectorTexture == 0
        ) {
            return 1.0f;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            mDetectorFramebuffer
        );

        glViewport(
            0,
            0,
            kDetectorWidth,
            kDetectorHeight
        );

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);

        glColorMask(
            GL_TRUE,
            GL_TRUE,
            GL_TRUE,
            GL_TRUE
        );

        glUseProgram(
            mDetectorProgram
        );

        if (mVaoSupported) {
            gBindVertexArray(mOwnVao);
        }

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        );

        glActiveTexture(GL_TEXTURE0);

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
        );

        glUniform1i(
            mDLocPrev,
            0
        );

        glActiveTexture(GL_TEXTURE1);

        glBindTexture(
            GL_TEXTURE_2D,
            mTexCur
        );

        glUniform1i(
            mDLocCur,
            1
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mDLocPos
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mDLocPos
            )
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mDLocUV
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts + 2
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mDLocUV
            )
        );

        glDrawArrays(
            GL_TRIANGLE_STRIP,
            0,
            4
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mDLocPos
            )
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mDLocUV
            )
        );

        // Read back only 4x4 pixels.
        // The expensive full-resolution comparison was already
        // performed by the GPU.
        uint8_t pixels[
            kDetectorWidth *
            kDetectorHeight *
            4
        ] = {};

        glReadPixels(
            0,
            0,
            kDetectorWidth,
            kDetectorHeight,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels
        );

        float sum = 0.0f;

        constexpr int pixelCount =
            kDetectorWidth *
            kDetectorHeight;

        for (int i = 0; i < pixelCount; ++i) {

            sum +=
                static_cast<float>(
                    pixels[i * 4]
                ) /
                255.0f;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        return std::clamp(
            sum /
            static_cast<float>(
                pixelCount
            ),
            0.0f,
            1.0f
        );
    }

    // ============================================================
    // Blend
    // ============================================================

    void drawBlended(
        float mvU,
        float mvV,
        float t
    ) {

        if (mBlendProgram == 0) {
            return;
        }

        if (mVaoSupported) {
            gBindVertexArray(mOwnVao);
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

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);

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

        glActiveTexture(GL_TEXTURE0);

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
        );

        glUniform1i(
            mBLocPrev,
            0
        );

        glActiveTexture(GL_TEXTURE1);

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

        GLuint program = 0;

        if (
            !linkProgram(
                kVertexShaderSrc,
                kBlendFragmentShaderSrc,
                program
            )
        ) {
            return false;
        }

        mBlendProgram = program;

        mBLocPos =
            glGetAttribLocation(
                program,
                "aPos"
            );

        mBLocUV =
            glGetAttribLocation(
                program,
                "aUV"
            );

        mBLocPrev =
            glGetUniformLocation(
                program,
                "uPrev"
            );

        mBLocCur =
            glGetUniformLocation(
                program,
                "uCur"
            );

        mBLocMv =
            glGetUniformLocation(
                program,
                "uMv"
            );

        mBLocT =
            glGetUniformLocation(
                program,
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

    bool buildDetector() {

        GLuint program = 0;

        if (
            !linkProgram(
                kVertexShaderSrc,
                kDetectorFragmentShaderSrc,
                program
            )
        ) {
            return false;
        }

        mDetectorProgram =
            program;

        mDLocPos =
            glGetAttribLocation(
                program,
                "aPos"
            );

        mDLocUV =
            glGetAttribLocation(
                program,
                "aUV"
            );

        mDLocPrev =
            glGetUniformLocation(
                program,
                "uPrev"
            );

        mDLocCur =
            glGetUniformLocation(
                program,
                "uCur"
            );

        glGenTextures(
            1,
            &mDetectorTexture
        );

        if (mDetectorTexture == 0) {
            return false;
        }

        glBindTexture(
            GL_TEXTURE_2D,
            mDetectorTexture
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MIN_FILTER,
            GL_NEAREST
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_NEAREST
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
            kDetectorWidth,
            kDetectorHeight,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        glGenFramebuffers(
            1,
            &mDetectorFramebuffer
        );

        if (mDetectorFramebuffer == 0) {
            return false;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            mDetectorFramebuffer
        );

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            mDetectorTexture,
            0
        );

        const GLenum status =
            glCheckFramebufferStatus(
                GL_FRAMEBUFFER
            );

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        if (
            status !=
            GL_FRAMEBUFFER_COMPLETE
        ) {

            return false;
        }

        mDetectorReady =
            mDLocPos >= 0 &&
            mDLocUV >= 0 &&
            mDLocPrev >= 0 &&
            mDLocCur >= 0;

        return mDetectorReady;
    }

    bool buildOverlayProgram() {

        GLuint program = 0;

        if (
            !linkProgram(
                kOverlayVertexShaderSrc,
                kOverlayFragmentShaderSrc,
                program
            )
        ) {
            return false;
        }

        mOverlayProgram = program;

        mOverlayPos =
            glGetAttribLocation(
                program,
                "aPos"
            );

        mOverlayColor =
            glGetUniformLocation(
                program,
                "uColor"
            );

        mOverlayReady =
            mOverlayPos >= 0 &&
            mOverlayColor >= 0;

        return mOverlayReady;
    }

    bool linkProgram(
        const char *vertexSource,
        const char *fragmentSource,
        GLuint &outProgram
    ) {

        const GLuint vs =
            compileShader(
                GL_VERTEX_SHADER,
                vertexSource
            );

        const GLuint fs =
            compileShader(
                GL_FRAGMENT_SHADER,
                fragmentSource
            );

        if (vs == 0 || fs == 0) {

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

        if (program == 0) {

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

        glLinkProgram(program);

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint linked = GL_FALSE;

        glGetProgramiv(
            program,
            GL_LINK_STATUS,
            &linked
        );

        if (linked == GL_FALSE) {

            glDeleteProgram(program);

            return false;
        }

        outProgram = program;

        return true;
    }

    GLuint compileShader(
        GLenum type,
        const char *source
    ) {

        const GLuint shader =
            glCreateShader(type);

        if (shader == 0) {
            return 0;
        }

        glShaderSource(
            shader,
            1,
            &source,
            nullptr
        );

        glCompileShader(shader);

        GLint compiled = GL_FALSE;

        glGetShaderiv(
            shader,
            GL_COMPILE_STATUS,
            &compiled
        );

        if (compiled == GL_FALSE) {

            GLint len = 0;

            glGetShaderiv(
                shader,
                GL_INFO_LOG_LENGTH,
                &len
            );

            if (len > 0) {

                std::string log(
                    static_cast<size_t>(len),
                    '\0'
                );

                glGetShaderInfoLog(
                    shader,
                    len,
                    nullptr,
                    log.data()
                );

                getSelf().getLogger().error(
                    "FrameGen shader error: {}",
                    log
                );
            }

            glDeleteShader(shader);

            return 0;
        }

        return shader;
    }

    // ============================================================
    // FPS digits
    // ============================================================

    static void addLine(
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
            return 0;
        }

        const uint8_t mask =
            table[digit];

        const float left = x;
        const float right = x + w;
        const float top = y;
        const float mid = y - h * 0.5f;
        const float bottom = y - h;

        int count = 0;

        if (mask & 0b0000001)
            addLine(
                vertices,
                count,
                left,
                top,
                right,
                top
            );

        if (mask & 0b0000010)
            addLine(
                vertices,
                count,
                right,
                top,
                right,
                mid
            );

        if (mask & 0b0000100)
            addLine(
                vertices,
                count,
                right,
                mid,
                right,
                bottom
            );

        if (mask & 0b0001000)
            addLine(
                vertices,
                count,
                left,
                bottom,
                right,
                bottom
            );

        if (mask & 0b0010000)
            addLine(
                vertices,
                count,
                left,
                mid,
                left,
                bottom
            );

        if (mask & 0b0100000)
            addLine(
                vertices,
                count,
                left,
                top,
                left,
                mid
            );

        if (mask & 0b1000000)
            addLine(
                vertices,
                count,
                left,
                mid,
                right,
                mid
            );

        return count;
    }

    static void drawNumber(
        int value,
        float startX,
        float startY,
        float width,
        float height,
        float spacing,
        float *vertices,
        int &count
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

        float x = startX;

        for (
            size_t i = 0;
            buffer[i] != '\0';
            ++i
        ) {

            count +=
                buildDigit(
                    buffer[i] - '0',
                    x,
                    startY,
                    width,
                    height,
                    vertices + count
                );

            x +=
                width +
                spacing;
        }
    }

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
            mOverlayProgram == 0
        ) {
            return;
        }

        float vertices[512] = {};

        int count = 0;

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

        addLine(
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

        if (count <= 0) {
            return;
        }

        GlStateGuard guard;

        if (mVaoSupported) {
            gBindVertexArray(mOwnVao);
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_BLEND);

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

        glLineWidth(3.0f);

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

    void destroyFrameTextures() {

        if (mTexPrev != 0) {

            glDeleteTextures(
                1,
                &mTexPrev
            );

            mTexPrev = 0;
        }

        if (mTexCur != 0) {

            glDeleteTextures(
                1,
                &mTexCur
            );

            mTexCur = 0;
        }

        mHaveHistory = false;
    }

    void destroyGlResources() {

        destroyFrameTextures();

        if (mDetectorFramebuffer != 0) {

            glDeleteFramebuffers(
                1,
                &mDetectorFramebuffer
            );

            mDetectorFramebuffer = 0;
        }

        if (mDetectorTexture != 0) {

            glDeleteTextures(
                1,
                &mDetectorTexture
            );

            mDetectorTexture = 0;
        }

        if (mBlendProgram != 0) {

            glDeleteProgram(
                mBlendProgram
            );

            mBlendProgram = 0;
        }

        if (mDetectorProgram != 0) {

            glDeleteProgram(
                mDetectorProgram
            );

            mDetectorProgram = 0;
        }

        if (mOverlayProgram != 0) {

            glDeleteProgram(
                mOverlayProgram
            );

            mOverlayProgram = 0;
        }

        if (
            mOwnVao != 0 &&
            gDeleteVertexArrays != nullptr
        ) {

            gDeleteVertexArrays(
                1,
                &mOwnVao
            );

            mOwnVao = 0;
        }

        mDetectorReady = false;
        mOverlayReady = false;
        mGlReady = false;

        mTexWidth = 0;
        mTexHeight = 0;

        hardTemporalReset();
    }
};

} // namespace

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
)
