// FrameGenMod.cpp
//
// Frame Generator
//
// Features:
//   - Generation Mode: 0 / 1 / 2 generated frames
//   - Quality Preset: 0 / 1 / 2
//   - REAL FPS counter
//   - GENERATED FPS counter
//   - Toggleable in-game FPS counter
//   - No glReadPixels
//   - No CPU motion estimation
//   - 0,0 motion vector
//
// IMPORTANT:
// The FPS counter is drawn directly through GLES immediately before
// eglSwapBuffers(). It is NOT the ModMenu UI.
//
// FPS counter config:
//   0 = OFF
//   1 = ON
//
// ModMenu switches are intentionally represented as SliderInt 0..1
// because this ModMenu API does not provide ConfigType::Switch.
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

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

constexpr int64_t kMinFrameIntervalNs =
    2'000'000LL;

(m)=>m+"\nconstexpr EGLint kEglRefreshRate = 0x200F;\nconstexpr int64_t kFallbackFramePeriodNs = 16'666'667LL;\n"
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
// FPS counter shader
//
// This is deliberately separate from the FrameGen shader.
//
// The counter uses simple line segments instead of a font atlas.
// Therefore it needs no external font texture or dependency.
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
// EGL / VAO
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
// EGL extension
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

        // --------------------------------------------------------
        // VAO functions
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
        // ModMenu
        //
        // IMPORTANT:
        // SliderInt is used for everything because your installed
        // ModMenu API does not contain ConfigType::Switch.
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
                    "FrameGen with real/generated "
                    "FPS counter."
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

(m)=>m+"\n    EGLDisplay mRefreshDisplay = EGL_NO_DISPLAY;\n    EGLSurface mRefreshSurface = EGL_NO_SURFACE;\n    int64_t mFramePeriodNs = kFallbackFramePeriodNs;\n"
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
    // FPS overlay program
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
    // Module callbacks
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

    int64_t framePeriodForSurface(EGLDisplay dpy, EGLSurface surface) {
        if (dpy == mRefreshDisplay && surface == mRefreshSurface) {
            return mFramePeriodNs;
        }

        mRefreshDisplay = dpy;
        mRefreshSurface = surface;
        mFramePeriodNs = kFallbackFramePeriodNs;

        EGLint surfaceConfigId = 0;
        EGLint configCount = 0;
        if (eglQuerySurface(dpy, surface, EGL_CONFIG_ID, &surfaceConfigId) == EGL_FALSE ||
            eglGetConfigs(dpy, nullptr, 0, &configCount) == EGL_FALSE ||
            configCount <= 0) {
            return mFramePeriodNs;
        }

        std::vector<EGLConfig> configs(static_cast<size_t>(configCount));
        EGLint returnedCount = 0;
        if (eglGetConfigs(dpy, configs.data(), configCount, &returnedCount) == EGL_FALSE) {
            return mFramePeriodNs;
        }

        for (EGLint i = 0; i < returnedCount; ++i) {
            EGLint configId = 0;
            EGLint refreshRate = 0;
            const EGLConfig config = configs[static_cast<size_t>(i)];
            if (eglGetConfigAttrib(dpy, config, EGL_CONFIG_ID, &configId) != EGL_FALSE &&
                configId == surfaceConfigId &&
                eglGetConfigAttrib(dpy, config, kEglRefreshRate, &refreshRate) != EGL_FALSE &&
                refreshRate > 0) {
                mFramePeriodNs = 1'000'000'000LL / refreshRate;
                break;
            }
        }
        return mFramePeriodNs;
    }

    void resetTiming() {

        mHaveHistory =
            false;

        mLastHookStartNs =
            0;

(m)=>m+"\n        mRefreshDisplay = EGL_NO_DISPLAY;\n        mRefreshSurface = EGL_NO_SURFACE;\n        mFramePeriodNs = kFallbackFramePeriodNs;\n"    }

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
            mGenerationMode.load(
                std::memory_order_relaxed
            );

        // --------------------------------------------------------
        // Completely disabled
        // --------------------------------------------------------

        if (
            !mEnabled.load(
                std::memory_order_relaxed
            ) ||
            mode == 0
        ) {

            resetTiming();

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

                drawFpsCounter();
            }

            return result;
        }

        // --------------------------------------------------------
        // Required EGL support
        // --------------------------------------------------------

        if (
            !mVaoSupported ||
            !mTrueFrameGenAvailable ||
            !hasEglExtension(
                dpy,
                "EGL_ANDROID_presentation_time"
            )
        ) {

            resetTiming();

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

                drawFpsCounter();
            }

            return result;
        }

        // --------------------------------------------------------
        // Surface size
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // GL resources
        // --------------------------------------------------------

        if (
            !ensureGlResources(
                width,
                height
            )
        ) {

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

                drawFpsCounter();
            }

            return result;
        }

        // --------------------------------------------------------
        // Frame timing
        // --------------------------------------------------------

(m)=>m+"\n        const int64_t framePeriodNs =\n            framePeriodForSurface(dpy, surface);\n"
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

        // --------------------------------------------------------
        // Capture actual game frame
        // --------------------------------------------------------

        captureCurrentFrame();

        // --------------------------------------------------------
        // First frame
        // --------------------------------------------------------

        if (
            !mHaveHistory
        ) {

            const int64_t firstTimestamp = hookNowNs + framePeriodNs;

            mPresentationTimeFn(
                dpy,
                surface,
                firstTimestamp
            );

            // Draw actual current frame.
            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                mLastPresentedTimeNs =
                    firstTimestamp;

                mHaveHistory =
                    true;

                updateFrameStats(
                    false
                );

                std::swap(
                    mTexPrev,
                    mTexCur
                );

                // FPS counter is deliberately drawn AFTER
                // FrameGen processing so it doesn't become part
                // of the captured history texture.
                drawFpsCounter();

            } else {

                resetTiming();
            }

            return result;
        }

        // --------------------------------------------------------
        // Invalid frame interval
        // --------------------------------------------------------

        if (
            frameIntervalNs <
                kMinFrameIntervalNs ||
            frameIntervalNs >
                kMaxFrameIntervalNs
        ) {

            const int64_t timestamp = std::max(hookNowNs + framePeriodNs, mLastPresentedTimeNs + framePeriodNs);

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

                drawFpsCounter();

            } else {

                resetTiming();
            }

            return result;
        }

        // ========================================================
        // Motion
        // ========================================================

        // Intentionally disabled.
        //
        // This avoids glReadPixels and CPU motion estimation.
        //
        // The generated frames therefore use a zero motion vector.

        constexpr float mvU =
            0.0f;

        constexpr float mvV =
            0.0f;

        // ========================================================
        // Generation amount
        // ========================================================

        const int availableFrameSlots = std::max(1, static_cast<int>((frameIntervalNs + framePeriodNs / 2) / framePeriodNs));
        const int steps = std::min(mode, availableFrameSlots - 1);

        // ========================================================
        // Next real timestamp
        // ========================================================

        const int64_t previousRealTimestamp = mLastPresentedTimeNs;

        int64_t targetRealTimestamp = previousRealTimestamp + frameIntervalNs;
        const int64_t workCompleteNs = nowNs();
        if (targetRealTimestamp <= workCompleteNs) {
            targetRealTimestamp = workCompleteNs + framePeriodNs;
        }

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

            const int64_t syntheticTimestamp = previousRealTimestamp + static_cast<int64_t>(static_cast<double>(targetRealTimestamp - previousRealTimestamp) * static_cast<double>(t));

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
            // Generated frame.
            //
            // IMPORTANT:
            // The counter is NOT rendered here.
            //
            // This prevents the FPS UI itself from being blended
            // between previous/current generated frames.
            // ----------------------------------------------------

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

        const int64_t finalTimestamp = std::max(targetRealTimestamp, std::max(nowNs() + framePeriodNs, mLastPresentedTimeNs + framePeriodNs));

        mPresentationTimeFn(
            dpy,
            surface,
            finalTimestamp
        );

        drawBlended(
            0.0f,
            0.0f,
            1.0f
        );

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

            // ----------------------------------------------------
            // FPS counter is rendered only on the real frame.
            //
            // This avoids the counter itself being interpolated
            // by FrameGen.
            // ----------------------------------------------------

            drawFpsCounter();

        } else {

            resetTiming();
        }

        return result;
    }

    // ============================================================
    // GL resource creation
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
    // Capture current frame
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
    // Draw FrameGen frame
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

        // Motion
        glUniform2f(
            mBLocMv,
            mvU,
            mvV
        );

        // Interpolation
        glUniform1f(
            mBLocT,
            std::clamp(
                t,
                0.0f,
                1.0f
            )
        );

        // Position
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

        // UV
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

        return true;
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
    // FPS digit renderer
    // ============================================================

    //
    // Seven-segment digit:
    //
    //       0
    //      ---
    //   5 |   | 1
    //      ---
    //   4 |   | 2
    //      ---
    //       3
    //

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

            // 0
            0b0111111,

            // 1
            0b0000110,

            // 2
            0b1011011,

            // 3
            0b1001111,

            // 4
            0b1100110,

            // 5
            0b1101101,

            // 6
            0b1111101,

            // 7
            0b0000111,

            // 8
            0b1111111,

            // 9
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

        s0 =
            (mask & 0b0000001) != 0;

        s1 =
            (mask & 0b0000010) != 0;

        s2 =
            (mask & 0b0000100) != 0;

        s3 =
            (mask & 0b0001000) != 0;

        s4 =
            (mask & 0b0010000) != 0;

        s5 =
            (mask & 0b0100000) != 0;

        s6 =
            (mask & 0b1000000) != 0;

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

        vertices[count++] =
            x0;

        vertices[count++] =
            y0;

        vertices[count++] =
            x1;

        vertices[count++] =
            y1;
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

        const float t =
            w * 0.18f;

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

        const float cx =
            x + w * 0.5f;

        // Top
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

        // Upper right
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

        // Lower right
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

        // Bottom
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

        // Lower left
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

        // Upper left
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

        // Middle
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

        (void)t;
        (void)cx;

        return count;
    }

    // ============================================================
    // Draw number
    // ============================================================

    void drawNumber(
        int value,
        float startX,
        float startY,
        float digitWidth,
        float digitHeight,
        float spacing,
        float *vertices,
        int &vertexFloatCount
    ) {

        if (
            value < 0
        ) {
            value = 0;
        }

        if (
            value > 9999
        ) {
            value = 9999;
        }

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

        //
        // The overlay is deliberately drawn AFTER the real frame
        // is submitted to the compositor.
        //
        // This prevents the FPS counter from entering the FrameGen
        // history textures.
        //
        // The counter therefore cannot be interpolated/ghosted by
        // the FrameGen shader itself.
        //

        const int realFps =
            static_cast<int>(
                mLastRealFps
            );

        const int generatedFps =
            static_cast<int>(
                mLastGeneratedFps
            );

        //
        // Only draw the REAL number and GENERATED number.
        //
        // Position: upper-left.
        //

        float vertices[512];

        int count =
            0;

        //
        // REAL FPS number
        //

        drawNumber(
            realFps,
            -0.94f,
            0.92f,
            0.035f,
            0.070f,
            0.014f,
            vertices,
            count
        );

        //
        // Small separator made from two lines.
        //

        addSegment(
            vertices,
            count,
            -0.94f,
            0.82f,
            -0.90f,
            0.82f
        );

        //
        // GENERATED FPS number below.
        //

        drawNumber(
            generatedFps,
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
