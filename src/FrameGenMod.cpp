// FrameGenMod.cpp
//
// Frame Generator
//
// Features:
//   - Generation Mode: 0 / 1 / 2 generated frames
//   - Quality Preset: 0 / 1 / 2
//   - FPS Counter
//   - FPS Cap: 10 - 144
//   - No glReadPixels
//   - No CPU motion estimation
//   - Zero motion vector
//   - Safer EGL/GL state handling
//   - Context/resize recovery
//
// IMPORTANT:
//   This implementation intentionally does NOT use
//   eglPresentationTimeANDROID for synthetic frame scheduling.
//
//   The previous implementation could produce unstable pacing
//   because multiple eglSwapBuffers calls were combined with
//   manually scheduled presentation timestamps.
//
//   The FPS counter is rendered BEFORE eglSwapBuffers() so it is
//   actually presented to the screen.
//
// ============================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

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

// Minimum accepted real-frame interval.
// 2 ms = 500 FPS.
constexpr int64_t kMinFrameIntervalNs =
    2'000'000LL;

// Maximum accepted real-frame interval.
// 80 ms = 12.5 FPS.
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
// EGL
// ============================================================

using EglSwapBuffersFn =
    decltype(&eglSwapBuffers);

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

        mAttrib0Enabled =
            getAttribEnabled(0);

        mAttrib1Enabled =
            getAttribEnabled(1);

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

        restoreAttrib(
            0,
            mAttrib0Enabled
        );

        restoreAttrib(
            1,
            mAttrib1Enabled
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

    static GLboolean getAttribEnabled(
        GLuint index
    ) {

        GLint value =
            GL_FALSE;

        glGetVertexAttribiv(
            index,
            GL_VERTEX_ATTRIB_ARRAY_ENABLED,
            &value
        );

        return value
            ? GL_TRUE
            : GL_FALSE;
    }

    static void restoreAttrib(
        GLuint index,
        GLboolean enabled
    ) {

        if (enabled) {
            glEnableVertexAttribArray(
                index
            );
        } else {
            glDisableVertexAttribArray(
                index
            );
        }
    }

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

    GLint mProgram =
        0;

    GLint mActiveTexture =
        GL_TEXTURE0;

    GLint mViewport[4] = {
        0,
        0,
        0,
        0
    };

    GLint mFramebuffer =
        0;

    GLint mArrayBuffer =
        0;

    GLint mTex0 =
        0;

    GLint mTex1 =
        0;

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

    GLboolean mAttrib0Enabled =
        GL_FALSE;

    GLboolean mAttrib1Enabled =
        GL_FALSE;
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

        // --------------------------------------------------------
        // Mod Menu
        // --------------------------------------------------------

        const bool registered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator"
            )
                .modId(
                    getSelf().getId()
                )
                .description(
                    "Frame generation, FPS counter and FPS cap."
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

        if (!registered) {

            getSelf().getLogger().error(
                "Failed to register Frame Generator"
            );

            mSwapHook.reset();

            return false;
        }

        resetTiming();

        resetStats();

        getSelf().getLogger().info(
            "Frame Generator enabled"
        );

        return true;
    }

    bool disable() {

        pl::modmenu::unregisterModule(
            kModuleId
        );

        mSwapHook.reset();

        if (
            eglGetCurrentContext() !=
            EGL_NO_CONTEXT
        ) {

            destroyGlResources();
        } else {

            invalidateGlHandles();
        }

        resetTiming();

        resetStats();

        getSelf().getLogger().info(
            "Frame Generator disabled"
        );

        return true;
    }

    bool unload() {

        return true;
    }

private:

    ll::mod::NativeMod &mSelf;

    pl::memory::HookHandle
        mSwapHook;

    EglSwapBuffersFn
        mOriginalSwapBuffers =
            nullptr;

    // ============================================================
    // Runtime settings
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

    // 10 - 144
    std::atomic_int mFpsCap{
        90
    };

    // ============================================================
    // Timing
    // ============================================================

    int64_t mLastHookNs =
        0;

    int64_t mLastCapNs =
        0;

    // ============================================================
    // Stats
    // ============================================================

    int64_t mStatsStartNs =
        0;

    uint32_t mRealFrames =
        0;

    uint32_t mGeneratedFrames =
        0;

    uint32_t mLastRealFps =
        0;

    uint32_t mLastGeneratedFps =
        0;

    // ============================================================
    // EGL/GL context
    // ============================================================

    EGLContext mContext =
        EGL_NO_CONTEXT;

    EGLSurface mSurface =
        EGL_NO_SURFACE;

    // ============================================================
    // Frame history
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

    GLint mBLocPrev =
        -1;

    GLint mBLocCur =
        -1;

    GLint mBLocMv =
        -1;

    GLint mBLocT =
        -1;

    GLuint mQuadVbo =
        0;

    // ============================================================
    // FPS program
    // ============================================================

    GLuint mOverlayProgram =
        0;

    GLint mOverlayColor =
        -1;

    GLuint mOverlayVbo =
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

        auto &self =
            instance();

        self.mEnabled.store(
            enabled,
            std::memory_order_relaxed
        );

        self.resetTiming();

        if (!enabled) {
            self.resetStats();
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

        auto &self =
            instance();

        if (
            key == kModeKey
        ) {

            self.mGenerationMode.store(
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                ),
                std::memory_order_relaxed
            );

            self.resetTiming();

        } else if (
            key == kPresetKey
        ) {

            self.mQualityPreset.store(
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

            self.mFpsCounter.store(
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
                        10,
                        144
                    )
                );

            self.mFpsCap.store(
                cap,
                std::memory_order_relaxed
            );

            self.mLastCapNs =
                nowNs();
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

    static void sleepNs(
        int64_t ns
    ) {

        if (ns <= 0) {
            return;
        }

        std::this_thread::sleep_for(
            std::chrono::nanoseconds(
                ns
            )
        );
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
                10,
                144
            );

        const int64_t intervalNs =
            1'000'000'000LL /
            static_cast<int64_t>(
                cap
            );

        const int64_t current =
            nowNs();

        if (
            mLastCapNs != 0
        ) {

            const int64_t elapsed =
                current -
                mLastCapNs;

            if (
                elapsed <
                intervalNs
            ) {

                sleepNs(
                    intervalNs -
                    elapsed
                );
            }
        }

        mLastCapNs =
            nowNs();
    }

    // ============================================================
    // Timing reset
    // ============================================================

    void resetTiming() {

        mLastHookNs =
            0;

        mLastCapNs =
            nowNs();

        mHaveHistory =
            false;
    }

    // ============================================================
    // Statistics
    // ============================================================

    void resetStats() {

        mStatsStartNs =
            nowNs();

        mRealFrames =
            0;

        mGeneratedFrames =
            0;

        mLastRealFps =
            0;

        mLastGeneratedFps =
            0;
    }

    void updateStats(
        bool generated
    ) {

        const int64_t current =
            nowNs();

        if (
            mStatsStartNs == 0
        ) {

            mStatsStartNs =
                current;
        }

        if (generated) {
            ++mGeneratedFrames;
        } else {
            ++mRealFrames;
        }

        const int64_t elapsed =
            current -
            mStatsStartNs;

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
                    mRealFrames
                ) /
                seconds
            );

        mLastGeneratedFps =
            static_cast<uint32_t>(
                static_cast<double>(
                    mGeneratedFrames
                ) /
                seconds
            );

        mRealFrames =
            0;

        mGeneratedFrames =
            0;

        mStatsStartNs =
            current;
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

        // --------------------------------------------------------
        // Apply FPS cap FIRST.
        //
        // This makes the cap apply to real Minecraft frame
        // submissions instead of only generated frames.
        // --------------------------------------------------------

        applyFpsCap();

        // --------------------------------------------------------
        // If module is disabled, pass through.
        // --------------------------------------------------------

        if (
            !mEnabled.load(
                std::memory_order_relaxed
            )
        ) {

            resetTiming();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        // --------------------------------------------------------
        // Current EGL context.
        // --------------------------------------------------------

        const EGLContext context =
            eglGetCurrentContext();

        if (
            context == EGL_NO_CONTEXT
        ) {

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        // --------------------------------------------------------
        // Detect context change.
        //
        // Android can destroy/recreate EGL contexts. Old GL
        // handles must never be reused with the new context.
        // --------------------------------------------------------

        if (
            mContext != EGL_NO_CONTEXT &&
            mContext != context
        ) {

            invalidateGlHandles();

            mHaveHistory =
                false;
        }

        mContext =
            context;

        mSurface =
            surface;

        // --------------------------------------------------------
        // Surface dimensions.
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
        // GL resources.
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // Generation mode.
        // --------------------------------------------------------

        const int mode =
            std::clamp(
                mGenerationMode.load(
                    std::memory_order_relaxed
                ),
                0,
                2
            );

        // --------------------------------------------------------
        // Generation disabled.
        // --------------------------------------------------------

        if (
            mode == 0
        ) {

            resetTiming();

            GlStateGuard guard;

            drawRealFrame();

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                ) != 0
            ) {

                updateStats(
                    false
                );

                drawFpsCounter();
            }

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                mHaveHistory =
                    false;
            }

            return result;
        }

        // --------------------------------------------------------
        // Frame interval.
        // --------------------------------------------------------

        const int64_t current =
            nowNs();

        int64_t frameIntervalNs =
            0;

        if (
            mLastHookNs != 0
        ) {

            frameIntervalNs =
                current -
                mLastHookNs;
        }

        mLastHookNs =
            current;

        // --------------------------------------------------------
        // Guard all GL changes.
        // --------------------------------------------------------

        GlStateGuard guard;

        // --------------------------------------------------------
        // Capture the clean Minecraft frame BEFORE drawing the
        // FPS counter.
        //
        // Therefore the counter never enters the history texture.
        // --------------------------------------------------------

        captureCurrentFrame();

        // --------------------------------------------------------
        // First frame / bad interval.
        // --------------------------------------------------------

        if (
            !mHaveHistory ||
            frameIntervalNs <
                kMinFrameIntervalNs ||
            frameIntervalNs >
                kMaxFrameIntervalNs
        ) {

            drawRealFrame();

            updateStats(
                false
            );

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                ) != 0
            ) {

                drawFpsCounter();
            }

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result == EGL_TRUE
            ) {

                mHaveHistory =
                    true;

                std::swap(
                    mTexPrev,
                    mTexCur
                );

            } else {

                resetTiming();
            }

            return result;
        }

        // --------------------------------------------------------
        // Zero motion vector.
        //
        // This is intentionally kept safe.
        // No glReadPixels.
        // No CPU optical-flow scan.
        // --------------------------------------------------------

        constexpr float mvU =
            0.0f;

        constexpr float mvV =
            0.0f;

        // --------------------------------------------------------
        // Generated frames.
        // --------------------------------------------------------

        const int generatedCount =
            std::clamp(
                mode,
                1,
                2
            );

        for (
            int i = 1;
            i <= generatedCount;
            ++i
        ) {

            const float t =
                static_cast<float>(i) /
                static_cast<float>(
                    generatedCount + 1
                );

            drawBlended(
                mvU,
                mvV,
                t
            );

            // Update stats before drawing the counter so the
            // displayed number is current.
            updateStats(
                true
            );

            if (
                mFpsCounter.load(
                    std::memory_order_relaxed
                ) != 0
            ) {

                drawFpsCounter();
            }

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (
                result != EGL_TRUE
            ) {

                resetTiming();

                return result;
            }
        }

        // --------------------------------------------------------
        // Final REAL Minecraft frame.
        // --------------------------------------------------------

        drawRealFrame();

        updateStats(
            false
        );

        if (
            mFpsCounter.load(
                std::memory_order_relaxed
            ) != 0
        ) {

            drawFpsCounter();
        }

        const EGLBoolean result =
            mOriginalSwapBuffers(
                dpy,
                surface
            );

        if (
            result == EGL_TRUE
        ) {

            mHaveHistory =
                true;

            // Swap the clean captured textures.
            //
            // The FPS counter was drawn AFTER capture, so it is
            // not copied into history.
            //
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
    // Ensure GL resources
    // ============================================================

    bool ensureGlResources(
        GLsizei width,
        GLsizei height
    ) {

        const EGLContext currentContext =
            eglGetCurrentContext();

        if (
            currentContext ==
            EGL_NO_CONTEXT
        ) {

            return false;
        }

        if (
            mBlendProgram == 0
        ) {

            if (
                !buildBlendProgram()
            ) {

                return false;
            }
        }

        if (
            mOverlayProgram == 0
        ) {

            if (
                !buildOverlayProgram()
            ) {

                return false;
            }
        }

        if (
            mQuadVbo == 0
        ) {

            glGenBuffers(
                1,
                &mQuadVbo
            );

            if (
                mQuadVbo == 0
            ) {

                return false;
            }

            glBindBuffer(
                GL_ARRAY_BUFFER,
                mQuadVbo
            );

            glBufferData(
                GL_ARRAY_BUFFER,
                sizeof(kQuadVerts),
                kQuadVerts,
                GL_STATIC_DRAW
            );
        }

        if (
            mOverlayVbo == 0
        ) {

            glGenBuffers(
                1,
                &mOverlayVbo
            );

            if (
                mOverlayVbo == 0
            ) {

                return false;
            }
        }

        if (
            mGlReady &&
            width == mTexWidth &&
            height == mTexHeight
        ) {

            return true;
        }

        // --------------------------------------------------------
        // Resize.
        // --------------------------------------------------------

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

        return true;
    }

    // ============================================================
    // Texture creation
    // ============================================================

    static GLuint createFrameTexture(
        GLsizei width,
        GLsizei height
    ) {

        GLuint texture =
            0;

        glGenTextures(
            1,
            &texture
        );

        if (
            texture == 0
        ) {

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

        return texture;
    }

    // ============================================================
    // Capture current Minecraft frame
    // ============================================================

    void captureCurrentFrame() {

        if (
            mTexCur == 0
        ) {

            return;
        }

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
    // Draw real frame
    // ============================================================

    void drawRealFrame() {

        drawBlended(
            0.0f,
            0.0f,
            1.0f
        );
    }

    // ============================================================
    // Draw generated/current frame
    // ============================================================

    void drawBlended(
        float mvU,
        float mvV,
        float t
    ) {

        if (
            mBlendProgram == 0 ||
            mQuadVbo == 0
        ) {

            return;
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

        // Previous frame.
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

        // Current frame.
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

        glBindBuffer(
            GL_ARRAY_BUFFER,
            mQuadVbo
        );

        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            reinterpret_cast<const void *>(
                0
            )
        );

        glEnableVertexAttribArray(
            0
        );

        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            reinterpret_cast<const void *>(
                2 * sizeof(float)
            )
        );

        glEnableVertexAttribArray(
            1
        );

        glDrawArrays(
            GL_TRIANGLE_STRIP,
            0,
            4
        );

        glDisableVertexAttribArray(
            0
        );

        glDisableVertexAttribArray(
            1
        );
    }

    // ============================================================
    // Build FrameGen shader
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

        // Explicit attribute locations.
        //
        // This removes uncertainty about where aPos/aUV land.
        //

        glBindAttribLocation(
            mBlendProgram,
            0,
            "aPos"
        );

        glBindAttribLocation(
            mBlendProgram,
            1,
            "aUV"
        );

        // The locations above must be assigned before linking,
        // so rebuild with the explicit bindings.

        glDeleteProgram(
            mBlendProgram
        );

        mBlendProgram =
            0;

        const GLuint vs =
            compileShader(
                GL_VERTEX_SHADER,
                kVertexShaderSrc
            );

        const GLuint fs =
            compileShader(
                GL_FRAGMENT_SHADER,
                kBlendFragmentShaderSrc
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

        program =
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

        mBlendProgram =
            program;

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
            mBLocPrev >= 0 &&
            mBLocCur >= 0 &&
            mBLocMv >= 0 &&
            mBLocT >= 0;
    }

    // ============================================================
    // Build overlay shader
    // ============================================================

    bool buildOverlayProgram() {

        GLuint program =
            0;

        const GLuint vs =
            compileShader(
                GL_VERTEX_SHADER,
                kOverlayVertexShaderSrc
            );

        const GLuint fs =
            compileShader(
                GL_FRAGMENT_SHADER,
                kOverlayFragmentShaderSrc
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

        program =
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

        mOverlayProgram =
            program;

        mOverlayColor =
            glGetUniformLocation(
                mOverlayProgram,
                "uColor"
            );

        return
            mOverlayColor >= 0;
    }

    // ============================================================
    // Shader helpers
    // ============================================================

    GLuint compileShader(
        GLenum type,
        const char *source
    ) {

        const GLuint shader =
            glCreateShader(
                type
            );

        if (
            shader == 0
        ) {

            return 0;
        }

        glShaderSource(
            shader,
            1,
            &source,
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

    void logShaderError(
        GLuint shader
    ) {

        GLint length =
            0;

        glGetShaderiv(
            shader,
            GL_INFO_LOG_LENGTH,
            &length
        );

        if (
            length <= 0
        ) {
            return;
        }

        std::string log(
            static_cast<size_t>(
                length
            ),
            '\0'
        );

        glGetShaderInfoLog(
            shader,
            length,
            nullptr,
            log.data()
        );

        getSelf().getLogger().error(
            "FrameGen shader error: {}",
            log
        );
    }

    void logProgramError(
        GLuint program
    ) {

        GLint length =
            0;

        glGetProgramiv(
            program,
            GL_INFO_LOG_LENGTH,
            &length
        );

        if (
            length <= 0
        ) {
            return;
        }

        std::string log(
            static_cast<size_t>(
                length
            ),
            '\0'
        );

        glGetProgramInfoLog(
            program,
            length,
            nullptr,
            log.data()
        );

        getSelf().getLogger().error(
            "FrameGen program error: {}",
            log
        );
    }

    // ============================================================
    // Seven segment digits
    // ============================================================

    static uint8_t digitMask(
        int digit
    ) {

        static constexpr uint8_t masks[10] = {

            0b0111111, // 0
            0b0000110, // 1
            0b1011011, // 2
            0b1001111, // 3
            0b1100110, // 4
            0b1101101, // 5
            0b1111101, // 6
            0b0000111, // 7
            0b1111111, // 8
            0b1101111  // 9
        };

        if (
            digit < 0 ||
            digit > 9
        ) {

            return 0;
        }

        return masks[digit];
    }

    static void addLine(
        float *vertices,
        int &count,
        float x1,
        float y1,
        float x2,
        float y2
    ) {

        vertices[count++] =
            x1;

        vertices[count++] =
            y1;

        vertices[count++] =
            x2;

        vertices[count++] =
            y2;
    }

    static int buildDigit(
        int digit,
        float x,
        float y,
        float width,
        float height,
        float *vertices
    ) {

        const uint8_t mask =
            digitMask(
                digit
            );

        int count =
            0;

        const float right =
            x + width;

        const float middle =
            y - height * 0.5f;

        const float bottom =
            y - height;

        if (mask & 0b0000001) {
            addLine(
                vertices,
                count,
                x,
                y,
                right,
                y
            );
        }

        if (mask & 0b0000010) {
            addLine(
                vertices,
                count,
                right,
                y,
                right,
                middle
            );
        }

        if (mask & 0b0000100) {
            addLine(
                vertices,
                count,
                right,
                middle,
                right,
                bottom
            );
        }

        if (mask & 0b0001000) {
            addLine(
                vertices,
                count,
                x,
                bottom,
                right,
                bottom
            );
        }

        if (mask & 0b0010000) {
            addLine(
                vertices,
                count,
                x,
                middle,
                x,
                bottom
            );
        }

        if (mask & 0b0100000) {
            addLine(
                vertices,
                count,
                x,
                y,
                x,
                middle
            );
        }

        if (mask & 0b1000000) {
            addLine(
                vertices,
                count,
                x,
                middle,
                right,
                middle
            );
        }

        return count;
    }

    static int buildNumber(
        int value,
        float x,
        float y,
        float width,
        float height,
        float spacing,
        float *vertices
    ) {

        value =
            std::clamp(
                value,
                0,
                9999
            );

        char text[16];

        std::snprintf(
            text,
            sizeof(text),
            "%d",
            value
        );

        int count =
            0;

        for (
            size_t i = 0;
            text[i] != '\0';
            ++i
        ) {

            const int digit =
                text[i] - '0';

            count +=
                buildDigit(
                    digit,
                    x,
                    y,
                    width,
                    height,
                    vertices + count
                );

            x +=
                width +
                spacing;
        }

        return count;
    }

    // ============================================================
    // FPS counter
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
            mOverlayProgram == 0 ||
            mOverlayVbo == 0
        ) {

            return;
        }

        // Enough space for both FPS values.
        float vertices[512];

        int count =
            0;

        count +=
            buildNumber(
                static_cast<int>(
                    mLastRealFps
                ),
                -0.94f,
                0.94f,
                0.035f,
                0.065f,
                0.014f,
                vertices + count
            );

        // Separator.
        addLine(
            vertices,
            count,
            -0.94f,
            0.82f,
            -0.90f,
            0.82f
        );

        count +=
            buildNumber(
                static_cast<int>(
                    mLastGeneratedFps
                ),
                -0.94f,
                0.77f,
                0.035f,
                0.065f,
                0.014f,
                vertices + count
            );

        if (
            count <= 0
        ) {

            return;
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

        glUseProgram(
            mOverlayProgram
        );

        glUniform4f(
            mOverlayColor,
            1.0f,
            1.0f,
            1.0f,
            1.0f
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            mOverlayVbo
        );

        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(
                count *
                sizeof(float)
            ),
            vertices,
            GL_DYNAMIC_DRAW
        );

        glVertexAttribPointer(
            0,
            2,
            GL_FLOAT,
            GL_FALSE,
            2 * sizeof(float),
            reinterpret_cast<const void *>(
                0
            )
        );

        glEnableVertexAttribArray(
            0
        );

        glDrawArrays(
            GL_LINES,
            0,
            count / 2
        );

        glDisableVertexAttribArray(
            0
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

        mTexWidth =
            0;

        mTexHeight =
            0;

        mHaveHistory =
            false;
    }

    // ============================================================
    // Invalidate handles after context loss
    //
    // DO NOT call glDelete* here because the old EGL context may
    // already be destroyed.
    // ============================================================

    void invalidateGlHandles() {

        mBlendProgram =
            0;

        mOverlayProgram =
            0;

        mQuadVbo =
            0;

        mOverlayVbo =
            0;

        mTexPrev =
            0;

        mTexCur =
            0;

        mTexWidth =
            0;

        mTexHeight =
            0;

        mGlReady =
            false;

        mHaveHistory =
            false;

        mContext =
            EGL_NO_CONTEXT;
    }

    // ============================================================
    // Full GL cleanup
    // ============================================================

    void destroyGlResources() {

        if (
            eglGetCurrentContext() ==
            EGL_NO_CONTEXT
        ) {

            invalidateGlHandles();

            return;
        }

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
            mQuadVbo != 0
        ) {

            glDeleteBuffers(
                1,
                &mQuadVbo
            );

            mQuadVbo =
                0;
        }

        if (
            mOverlayVbo != 0
        ) {

            glDeleteBuffers(
                1,
                &mOverlayVbo
            );

            mOverlayVbo =
                0;
        }

        mGlReady =
            false;

        mHaveHistory =
            false;

        mContext =
            EGL_NO_CONTEXT;
    }
};

} // namespace

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
)
