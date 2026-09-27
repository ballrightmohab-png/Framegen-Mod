// FrameGenMod.cpp
//
// Frame Generator
//
// GPU frame-change detection + hard temporal reset.
//
// Fixes:
//   - GPU detects large frame/perspective changes
//   - Hard temporal reset when perspective changes
//   - Prevents old camera view from being blended with new camera view
//   - Uses only a 1x1 readback for the GPU detector
//   - Correct current -> previous texture history handling
//   - No glReadPixels of the full screen
//   - No CPU motion estimation
//   - Quality preset actually changes temporal rejection sensitivity
//   - Safer timing reset after camera changes
//   - Safer first-frame initialization
//   - Prevents stale temporal history after bad frame intervals
//   - FPS counter
//   - FPS cap 10..144
//   - Generation Mode 0 / 1 / 2
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <cstdio>

#include <dlfcn.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "ll/api/mod/NativeMod.h"
#include "pl/Mod.hpp"
#include "pl/ModMenu.hpp"
#include "pl/memory/Hook.hpp"

namespace {

constexpr const char* kModuleId = "framegen.core";

constexpr const char* kGenerationMode = "generationMode";
constexpr const char* kQualityPreset  = "qualityPreset";
constexpr const char* kFpsCounter     = "fpsCounter";
constexpr const char* kFpsCap         = "fpsCap";

// ------------------------------------------------------------
// Timing
// ------------------------------------------------------------

constexpr int kMinFpsCap = 10;
constexpr int kMaxFpsCap = 144;

constexpr uint64_t kMinFrameIntervalNs = 2'000'000ULL;      // 2 ms
constexpr uint64_t kMaxFrameIntervalNs = 100'000'000ULL;    // 100 ms

constexpr uint64_t kMaxGenerationIntervalNs = 50'000'000ULL;
constexpr uint64_t kMinimumUsefulIntervalNs = 4'000'000ULL;

constexpr uint64_t kSleepMarginNs = 500'000ULL;

// ------------------------------------------------------------
// GPU perspective detector
// ------------------------------------------------------------
//
// The detector compares 25 samples from the previous frame
// against 25 samples from the current frame.
//
// The result is written into a 1x1 RGBA texture.
//
// Only that single pixel is read back.
//
// This is intentionally NOT a full-screen glReadPixels.
//

constexpr int kDetectionGrid = 5;

// The actual thresholds are selected by quality preset.
//
// 0 = Low
// 1 = Balanced
// 2 = Smoothness
//
// Smaller threshold = more sensitive to perspective changes.

struct DetectionSettings {
    float hardResetThreshold;
    float softRejectThreshold;
};

static DetectionSettings getDetectionSettings(int quality) {
    switch (quality) {
        case 0:
            return {
                0.155f,
                0.055f
            };

        case 2:
            return {
                0.095f,
                0.030f
            };

        case 1:
        default:
            return {
                0.120f,
                0.040f
            };
    }
}

// ------------------------------------------------------------
// Shaders
// ------------------------------------------------------------

static const char* kFullscreenVertexShader = R"(
attribute vec2 aPos;
attribute vec2 aUV;

varying vec2 vUV;

void main() {
    vUV = aUV;

    gl_Position = vec4(
        aPos.x,
        aPos.y,
        0.0,
        1.0
    );
}
)";

// ------------------------------------------------------------
// Temporal generation shader
// ------------------------------------------------------------
//
// Important change:
//
// Instead of blindly doing:
//
//     mix(previous, current, t)
//
// the shader checks the local difference between previous
// and current.
//
// Large differences are treated as temporal instability.
//
// That means when the camera suddenly moves:
//
//     previous camera view
//              +
//     current camera view
//
// will NOT be blended together.
//
// The current frame wins in those areas.
//
// This is the GPU-side safety layer.
// The CPU-side 1x1 detector performs the hard reset.
//

static const char* kBlendFragmentShader = R"(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;

uniform vec2 uMv;
uniform float uT;

uniform float uRejectStart;
uniform float uRejectEnd;

void main() {

    vec2 uvPrev = clamp(
        vUV - uT * uMv,
        0.0,
        1.0
    );

    vec2 uvCur = clamp(
        vUV + (1.0 - uT) * uMv,
        0.0,
        1.0
    );

    vec4 prevColor = texture2D(
        uPrev,
        uvPrev
    );

    vec4 curColor = texture2D(
        uCur,
        uvCur
    );

    // --------------------------------------------------------
    // Local temporal difference.
    // --------------------------------------------------------

    vec3 difference = abs(
        curColor.rgb -
        prevColor.rgb
    );

    float diff =
          difference.r * 0.299
        + difference.g * 0.587
        + difference.b * 0.114;

    // --------------------------------------------------------
    // Soft temporal rejection.
    //
    // diff below rejectStart:
    //     normal interpolation
    //
    // diff above rejectEnd:
    //     current frame wins
    //
    // This prevents camera edges from ghosting.
    // --------------------------------------------------------

    float reject = smoothstep(
        uRejectStart,
        uRejectEnd,
        diff
    );

    float blendAmount = mix(
        clamp(uT, 0.0, 1.0),
        1.0,
        reject
    );

    gl_FragColor = mix(
        prevColor,
        curColor,
        blendAmount
    );
}
)";

// ------------------------------------------------------------
// GPU frame-change detector.
//
// It samples a 5x5 grid from the entire frame.
//
// Because this shader renders into a 1x1 texture, the CPU
// only needs to read one pixel afterwards.
//

static const char* kDetectionFragmentShader = R"(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;

void main() {

    float total = 0.0;

    // 5x5 samples.
    //
    // Keep these explicit instead of loops because some
    // mobile GLES2 shader compilers are bad with dynamic loops.

    vec2 p;

    p = vec2(0.10, 0.10);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.30, 0.10);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.50, 0.10);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.70, 0.10);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.90, 0.10);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );


    p = vec2(0.10, 0.30);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.30, 0.30);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.50, 0.30);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.70, 0.30);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.90, 0.30);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );


    p = vec2(0.10, 0.50);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.30, 0.50);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.50, 0.50);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.70, 0.50);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.90, 0.50);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );


    p = vec2(0.10, 0.70);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.30, 0.70);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.50, 0.70);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur,  p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.70, 0.70);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.90, 0.70);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );


    p = vec2(0.10, 0.90);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.30, 0.90);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.50, 0.90);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.70, 0.90);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    p = vec2(0.90, 0.90);
    total += dot(
        abs(texture2D(uPrev, p).rgb -
            texture2D(uCur, p).rgb),
        vec3(0.299, 0.587, 0.114)
    );

    // Average difference.
    float result = total / 25.0;

    gl_FragColor = vec4(
        result,
        result,
        result,
        1.0
    );
}
)";

// ------------------------------------------------------------
// Simple overlay shaders
// ------------------------------------------------------------

static const char* kOverlayVertexShader = R"(
attribute vec2 aPos;

void main() {
    gl_Position = vec4(
        aPos.x,
        aPos.y,
        0.0,
        1.0
    );
}
)";

static const char* kOverlayFragmentShader = R"(
precision mediump float;

uniform vec4 uColor;

void main() {
    gl_FragColor = uColor;
}
)";

// ------------------------------------------------------------
// GL state guard
// ------------------------------------------------------------

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
            GL_ELEMENT_ARRAY_BUFFER_BINDING,
            &mElementArrayBuffer
        );

        glGetIntegerv(
            GL_TEXTURE_BINDING_2D,
            &mTexture0
        );

        glActiveTexture(GL_TEXTURE1);

        glGetIntegerv(
            GL_TEXTURE_BINDING_2D,
            &mTexture1
        );

        glActiveTexture(
            static_cast<GLenum>(mActiveTexture)
        );

        mDepth = glIsEnabled(GL_DEPTH_TEST);
        mBlend = glIsEnabled(GL_BLEND);
        mCull = glIsEnabled(GL_CULL_FACE);
        mScissor = glIsEnabled(GL_SCISSOR_TEST);
        mStencil = glIsEnabled(GL_STENCIL_TEST);

        glGetBooleanv(
            GL_COLOR_WRITEMASK,
            mColorMask
        );
    }

    ~GlStateGuard() {

        if (mDepth)
            glEnable(GL_DEPTH_TEST);
        else
            glDisable(GL_DEPTH_TEST);

        if (mBlend)
            glEnable(GL_BLEND);
        else
            glDisable(GL_BLEND);

        if (mCull)
            glEnable(GL_CULL_FACE);
        else
            glDisable(GL_CULL_FACE);

        if (mScissor)
            glEnable(GL_SCISSOR_TEST);
        else
            glDisable(GL_SCISSOR_TEST);

        if (mStencil)
            glEnable(GL_STENCIL_TEST);
        else
            glDisable(GL_STENCIL_TEST);

        glColorMask(
            mColorMask[0],
            mColorMask[1],
            mColorMask[2],
            mColorMask[3]
        );

        glUseProgram(
            static_cast<GLuint>(mProgram)
        );

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            static_cast<GLuint>(mFramebuffer)
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            static_cast<GLuint>(mArrayBuffer)
        );

        glBindBuffer(
            GL_ELEMENT_ARRAY_BUFFER,
            static_cast<GLuint>(mElementArrayBuffer)
        );

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(mTexture0)
        );

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(mTexture1)
        );

        glActiveTexture(
            static_cast<GLenum>(mActiveTexture)
        );

        glViewport(
            mViewport[0],
            mViewport[1],
            mViewport[2],
            mViewport[3]
        );
    }

private:

    GLint mProgram = 0;
    GLint mActiveTexture = GL_TEXTURE0;
    GLint mViewport[4] = {0, 0, 0, 0};

    GLint mFramebuffer = 0;
    GLint mArrayBuffer = 0;
    GLint mElementArrayBuffer = 0;

    GLint mTexture0 = 0;
    GLint mTexture1 = 0;

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

// ------------------------------------------------------------
// Utility
// ------------------------------------------------------------

static uint64_t nowNs() {

    using namespace std::chrono;

    return duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()
    ).count();
}

static void sleepUntilNs(uint64_t target) {

    for (;;) {

        uint64_t current = nowNs();

        if (current >= target)
            break;

        uint64_t remaining =
            target - current;

        if (remaining > kSleepMarginNs) {

            std::this_thread::sleep_for(
                std::chrono::nanoseconds(
                    remaining - kSleepMarginNs
                )
            );
        } else {
            std::this_thread::yield();
        }
    }
}

// ------------------------------------------------------------
// Frame Generator
// ------------------------------------------------------------

class FrameGenMod : public ll::mod::NativeMod {

public:

    static FrameGenMod& instance() {

        static FrameGenMod mod;
        return mod;
    }

    FrameGenMod() = default;

    ~FrameGenMod() override {
        cleanup();
    }

    void onLoad() override {

        resolveEgl();

        installHook();

        registerMenu();

        mEnabled.store(true);
    }

private:

    // --------------------------------------------------------
    // Hook
    // --------------------------------------------------------

    using EglSwapBuffersFn =
        EGLBoolean (*)(EGLDisplay, EGLSurface);

    using EglPresentationTimeFn =
        EGLBoolean (*)(EGLDisplay, EGLSurface, EGLnsecsANDROID);

    EglSwapBuffersFn mOriginalSwapBuffers = nullptr;

    EglPresentationTimeFn
        mPresentationTime = nullptr;

    pl::memory::HookHandle mSwapHook;

    // --------------------------------------------------------
    // State
    // --------------------------------------------------------

    std::atomic<bool> mEnabled{false};

    std::atomic<int> mGenerationMode{1};

    std::atomic<int> mQualityPreset{1};

    std::atomic<bool> mFpsCounter{true};

    std::atomic<int> mFpsCap{90};

    bool mInitialized = false;

    bool mHistoryValid = false;

    int mWidth = 0;
    int mHeight = 0;

    // --------------------------------------------------------
    // Timing
    // --------------------------------------------------------

    uint64_t mLastHookStartNs = 0;

    uint64_t mLastPresentedTimeNs = 0;

    uint64_t mLastRealFrameNs = 0;

    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    double mRealFps = 0.0;

    double mGeneratedFps = 0.0;

    uint64_t mRealFrames = 0;
    uint64_t mGeneratedFrames = 0;

    uint64_t mStatWindowStartNs = 0;

    // --------------------------------------------------------
    // Textures
    // --------------------------------------------------------

    GLuint mTexPrev = 0;
    GLuint mTexCur = 0;

    // --------------------------------------------------------
    // Detector
    // --------------------------------------------------------

    GLuint mDetectionFbo = 0;
    GLuint mDetectionTex = 0;

    GLuint mDetectionProgram = 0;

    GLint mDetectionPrevLoc = -1;
    GLint mDetectionCurLoc = -1;

    // --------------------------------------------------------
    // Blend program
    // --------------------------------------------------------

    GLuint mBlendProgram = 0;

    GLint mBlendPosLoc = -1;
    GLint mBlendUvLoc = -1;

    GLint mBlendPrevLoc = -1;
    GLint mBlendCurLoc = -1;

    GLint mBlendMvLoc = -1;
    GLint mBlendTLoc = -1;

    GLint mBlendRejectStartLoc = -1;
    GLint mBlendRejectEndLoc = -1;

    // --------------------------------------------------------
    // Overlay
    // --------------------------------------------------------

    GLuint mOverlayProgram = 0;

    GLint mOverlayPosLoc = -1;
    GLint mOverlayColorLoc = -1;

    // --------------------------------------------------------
    // VAO
    // --------------------------------------------------------

    GLuint mVAO = 0;

    // --------------------------------------------------------
    // EGL
    // --------------------------------------------------------

    EGLDisplay mDisplay = EGL_NO_DISPLAY;

    EGLSurface mSurface = EGL_NO_SURFACE;

    // --------------------------------------------------------
    // Menu
    // --------------------------------------------------------

    void registerMenu() {

        // Keep the existing Levi ModMenu configuration.
        //
        // The module intentionally keeps the same keys so existing
        // configuration files remain compatible.

        pl::modmenu::ModuleBuilder builder(
            kModuleId,
            "Frame Generator"
        );

        builder
            .addConfig(
                kGenerationMode,
                pl::modmenu::ConfigType::SliderInt,
                1,
                0,
                2
            )
            .addConfig(
                kQualityPreset,
                pl::modmenu::ConfigType::SliderInt,
                1,
                0,
                2
            )
            .addConfig(
                kFpsCounter,
                pl::modmenu::ConfigType::Toggle,
                true
            )
            .addConfig(
                kFpsCap,
                pl::modmenu::ConfigType::SliderInt,
                90,
                kMinFpsCap,
                kMaxFpsCap
            );

        builder.onConfigChanged(
            [this](
                std::string_view key,
                int value
            ) {

                if (key == kGenerationMode) {

                    mGenerationMode.store(
                        std::clamp(value, 0, 2)
                    );

                    hardTemporalReset();
                }

                else if (key == kQualityPreset) {

                    mQualityPreset.store(
                        std::clamp(value, 0, 2)
                    );

                    // Quality changes alter the temporal
                    // detector, so do not keep history from
                    // the previous detector settings.

                    hardTemporalReset();
                }

                else if (key == kFpsCap) {

                    mFpsCap.store(
                        std::clamp(
                            value,
                            kMinFpsCap,
                            kMaxFpsCap
                        )
                    );

                    resetTimingOnly();
                }

                else if (key == kFpsCounter) {

                    mFpsCounter.store(
                        value != 0
                    );
                }
            }
        );

        builder.registerModule();
    }

    // --------------------------------------------------------
    // EGL resolution
    // --------------------------------------------------------

    void resolveEgl() {

        void* handle =
            dlopen(
                "libEGL.so",
                RTLD_NOW | RTLD_LOCAL
            );

        if (!handle)
            return;

        mOriginalSwapBuffers =
            reinterpret_cast<EglSwapBuffersFn>(
                dlsym(
                    handle,
                    "eglSwapBuffers"
                )
            );

        mPresentationTime =
            reinterpret_cast<EglPresentationTimeFn>(
                dlsym(
                    handle,
                    "eglPresentationTimeANDROID"
                )
            );
    }

    // --------------------------------------------------------
    // Hook installation
    // --------------------------------------------------------

    void installHook() {

        if (!mOriginalSwapBuffers)
            return;

        mSwapHook =
            pl::memory::Hook(
                reinterpret_cast<void*>(
                    mOriginalSwapBuffers
                ),
                reinterpret_cast<void*>(
                    &FrameGenMod::swapBuffersHook
                ),
                this
            );
    }

    static EGLBoolean swapBuffersHook(
        FrameGenMod* self,
        EGLDisplay display,
        EGLSurface surface
    ) {

        if (!self)
            return EGL_FALSE;

        return self->handleSwapBuffers(
            display,
            surface
        );
    }

    // --------------------------------------------------------
    // Timing reset
    // --------------------------------------------------------

    void resetTimingOnly() {

        mLastHookStartNs = 0;
        mLastPresentedTimeNs = 0;
        mLastRealFrameNs = 0;
    }

    // --------------------------------------------------------
    // HARD TEMPORAL RESET
    // --------------------------------------------------------
    //
    // This is the important fix.
    //
    // Current frame becomes BOTH:
    //
    //     previous
    //     current
    //
    // so the next generated frame can never blend an old camera
    // position with the newly changed camera position.
    //

    void hardTemporalReset() {

        mHistoryValid = false;

        mLastHookStartNs = 0;
        mLastPresentedTimeNs = 0;
        mLastRealFrameNs = 0;

        if (!mInitialized)
            return;

        copyCurrentToBothHistory();
    }

    // --------------------------------------------------------
    // Shader compilation
    // --------------------------------------------------------

    GLuint compileShader(
        GLenum type,
        const char* source
    ) {

        GLuint shader =
            glCreateShader(type);

        if (!shader)
            return 0;

        glShaderSource(
            shader,
            1,
            &source,
            nullptr
        );

        glCompileShader(shader);

        GLint success = GL_FALSE;

        glGetShaderiv(
            shader,
            GL_COMPILE_STATUS,
            &success
        );

        if (!success) {

            glDeleteShader(shader);

            return 0;
        }

        return shader;
    }

    GLuint buildProgram(
        const char* vertex,
        const char* fragment
    ) {

        GLuint vs =
            compileShader(
                GL_VERTEX_SHADER,
                vertex
            );

        if (!vs)
            return 0;

        GLuint fs =
            compileShader(
                GL_FRAGMENT_SHADER,
                fragment
            );

        if (!fs) {

            glDeleteShader(vs);

            return 0;
        }

        GLuint program =
            glCreateProgram();

        if (!program) {

            glDeleteShader(vs);
            glDeleteShader(fs);

            return 0;
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

        glLinkProgram(program);

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint success = GL_FALSE;

        glGetProgramiv(
            program,
            GL_LINK_STATUS,
            &success
        );

        if (!success) {

            glDeleteProgram(program);

            return 0;
        }

        return program;
    }

    // --------------------------------------------------------
    // GL initialization
    // --------------------------------------------------------

    bool ensureGlResources(
        int width,
        int height
    ) {

        if (
            width <= 0 ||
            height <= 0
        ) {
            return false;
        }

        if (
            mInitialized &&
            width == mWidth &&
            height == mHeight
        ) {
            return true;
        }

        destroyGlResources();

        mWidth = width;
        mHeight = height;

        // ----------------------------------------------------
        // Previous texture
        // ----------------------------------------------------

        glGenTextures(
            1,
            &mTexPrev
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
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
            mWidth,
            mHeight,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        // ----------------------------------------------------
        // Current texture
        // ----------------------------------------------------

        glGenTextures(
            1,
            &mTexCur
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexCur
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
            mWidth,
            mHeight,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        // ----------------------------------------------------
        // Detection texture
        // ----------------------------------------------------

        glGenTextures(
            1,
            &mDetectionTex
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mDetectionTex
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
            1,
            1,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            nullptr
        );

        // ----------------------------------------------------
        // Detection framebuffer
        // ----------------------------------------------------

        glGenFramebuffers(
            1,
            &mDetectionFbo
        );

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            mDetectionFbo
        );

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            mDetectionTex,
            0
        );

        GLenum status =
            glCheckFramebufferStatus(
                GL_FRAMEBUFFER
            );

        if (
            status !=
            GL_FRAMEBUFFER_COMPLETE
        ) {

            glBindFramebuffer(
                GL_FRAMEBUFFER,
                0
            );

            destroyGlResources();

            return false;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        // ----------------------------------------------------
        // Programs
        // ----------------------------------------------------

        mBlendProgram =
            buildProgram(
                kFullscreenVertexShader,
                kBlendFragmentShader
            );

        if (!mBlendProgram)
            return false;

        mBlendPosLoc =
            glGetAttribLocation(
                mBlendProgram,
                "aPos"
            );

        mBlendUvLoc =
            glGetAttribLocation(
                mBlendProgram,
                "aUV"
            );

        mBlendPrevLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uPrev"
            );

        mBlendCurLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uCur"
            );

        mBlendMvLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uMv"
            );

        mBlendTLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uT"
            );

        mBlendRejectStartLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uRejectStart"
            );

        mBlendRejectEndLoc =
            glGetUniformLocation(
                mBlendProgram,
                "uRejectEnd"
            );

        // ----------------------------------------------------
        // Detector program
        // ----------------------------------------------------

        mDetectionProgram =
            buildProgram(
                kFullscreenVertexShader,
                kDetectionFragmentShader
            );

        if (!mDetectionProgram)
            return false;

        mDetectionPrevLoc =
            glGetUniformLocation(
                mDetectionProgram,
                "uPrev"
            );

        mDetectionCurLoc =
            glGetUniformLocation(
                mDetectionProgram,
                "uCur"
            );

        // ----------------------------------------------------
        // Overlay
        // ----------------------------------------------------

        mOverlayProgram =
            buildProgram(
                kOverlayVertexShader,
                kOverlayFragmentShader
            );

        if (!mOverlayProgram)
            return false;

        mOverlayPosLoc =
            glGetAttribLocation(
                mOverlayProgram,
                "aPos"
            );

        mOverlayColorLoc =
            glGetUniformLocation(
                mOverlayProgram,
                "uColor"
            );

        // ----------------------------------------------------
        // VAO
        // ----------------------------------------------------

        auto genVertexArrays =
            reinterpret_cast<void (*)(GLsizei, GLuint*)>(
                dlsym(
                    RTLD_DEFAULT,
                    "glGenVertexArraysOES"
                )
            );

        if (genVertexArrays) {

            genVertexArrays(
                1,
                &mVAO
            );
        }

        mInitialized = true;

        mHistoryValid = false;

        return true;
    }

    // --------------------------------------------------------
    // Capture current framebuffer
    // --------------------------------------------------------

    void captureCurrentFrame() {

        if (!mTexCur)
            return;

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
            mWidth,
            mHeight
        );
    }

    // --------------------------------------------------------
    // Correct history copy
    // --------------------------------------------------------
    //
    // Both copies happen from the SAME framebuffer contents.
    //
    // This fixes the previous implementation where
    // copyCurrentToPrevious() could accidentally copy a
    // different framebuffer state.
    //

    void copyCurrentToBothHistory() {

        if (
            !mTexCur ||
            !mTexPrev
        ) {
            return;
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        // Current already contains the captured frame.

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
            mWidth,
            mHeight
        );

        mHistoryValid = true;
    }

    // --------------------------------------------------------
    // GPU frame-change detection
    // --------------------------------------------------------
    //
    // Returns true when the GPU detects a large temporal change.
    //

    bool detectMajorFrameChange() {

        if (
            !mDetectionFbo ||
            !mDetectionProgram ||
            !mTexPrev ||
            !mTexCur
        ) {
            return false;
        }

        GlStateGuard guard;

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            mDetectionFbo
        );

        glViewport(
            0,
            0,
            1,
            1
        );

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);

        glUseProgram(
            mDetectionProgram
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
            mDetectionPrevLoc,
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
            mDetectionCurLoc,
            1
        );

        drawFullscreen();

        // ----------------------------------------------------
        // IMPORTANT:
        //
        // This is ONLY a 1x1 readback.
        //
        // We do NOT read the game screen.
        // ----------------------------------------------------

        unsigned char pixel[4] = {
            0,
            0,
            0,
            0
        };

        glReadPixels(
            0,
            0,
            1,
            1,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixel
        );

        float difference =
            static_cast<float>(
                pixel[0]
            ) / 255.0f;

        DetectionSettings settings =
            getDetectionSettings(
                mQualityPreset.load()
            );

        return difference >=
               settings.hardResetThreshold;
    }

    // --------------------------------------------------------
    // Fullscreen quad
    // --------------------------------------------------------

    void drawFullscreen() {

        static const GLfloat vertices[] = {

            // position      // UV

            -1.0f, -1.0f,     0.0f, 0.0f,
             1.0f, -1.0f,     1.0f, 0.0f,
            -1.0f,  1.0f,     0.0f, 1.0f,

             1.0f, -1.0f,     1.0f, 0.0f,
             1.0f,  1.0f,     1.0f, 1.0f,
            -1.0f,  1.0f,     0.0f, 1.0f
        };

        glBindBuffer(
            GL_ARRAY_BUFFER,
            0
        );

        if (mBlendPosLoc >= 0) {

            glEnableVertexAttribArray(
                static_cast<GLuint>(
                    mBlendPosLoc
                )
            );

            glVertexAttribPointer(
                static_cast<GLuint>(
                    mBlendPosLoc
                ),
                2,
                GL_FLOAT,
                GL_FALSE,
                4 * sizeof(GLfloat),
                vertices
            );
        }

        if (mBlendUvLoc >= 0) {

            glEnableVertexAttribArray(
                static_cast<GLuint>(
                    mBlendUvLoc
                )
            );

            glVertexAttribPointer(
                static_cast<GLuint>(
                    mBlendUvLoc
                ),
                2,
                GL_FLOAT,
                GL_FALSE,
                4 * sizeof(GLfloat),
                vertices + 2
            );
        }

        glDrawArrays(
            GL_TRIANGLES,
            0,
            6
        );

        if (mBlendPosLoc >= 0)
            glDisableVertexAttribArray(
                static_cast<GLuint>(
                    mBlendPosLoc
                )
            );

        if (mBlendUvLoc >= 0)
            glDisableVertexAttribArray(
                static_cast<GLuint>(
                    mBlendUvLoc
                )
            );
    }

    // --------------------------------------------------------
    // Blended frame
    // --------------------------------------------------------

    void drawBlended(
        float mvU,
        float mvV,
        float t
    ) {

        if (
            !mBlendProgram ||
            !mTexPrev ||
            !mTexCur
        ) {
            return;
        }

        GlStateGuard guard;

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glViewport(
            0,
            0,
            mWidth,
            mHeight
        );

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);

        glUseProgram(
            mBlendProgram
        );

        glActiveTexture(
            GL_TEXTURE0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexPrev
        );

        glUniform1i(
            mBlendPrevLoc,
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
            mBlendCurLoc,
            1
        );

        glUniform2f(
            mBlendMvLoc,
            mvU,
            mvV
        );

        glUniform1f(
            mBlendTLoc,
            t
        );

        DetectionSettings settings =
            getDetectionSettings(
                mQualityPreset.load()
            );

        glUniform1f(
            mBlendRejectStartLoc,
            settings.softRejectThreshold
        );

        glUniform1f(
            mBlendRejectEndLoc,
            settings.hardResetThreshold
        );

        drawFullscreen();
    }

    // --------------------------------------------------------
    // FPS cap
    // --------------------------------------------------------

    void capFps() {

        int cap =
            std::clamp(
                mFpsCap.load(),
                kMinFpsCap,
                kMaxFpsCap
            );

        if (cap <= 0)
            return;

        uint64_t interval =
            1'000'000'000ULL /
            static_cast<uint64_t>(cap);

        static uint64_t lastCapNs = 0;

        uint64_t current =
            nowNs();

        if (lastCapNs != 0) {

            uint64_t target =
                lastCapNs + interval;

            if (target > current)
                sleepUntilNs(target);
        }

        lastCapNs = nowNs();
    }

    // --------------------------------------------------------
    // FPS statistics
    // --------------------------------------------------------

    void updateStats(
        bool generated
    ) {

        uint64_t now =
            nowNs();

        if (mStatWindowStartNs == 0) {

            mStatWindowStartNs =
                now;

            mRealFrames = 0;
            mGeneratedFrames = 0;
        }

        if (generated)
            ++mGeneratedFrames;
        else
            ++mRealFrames;

        uint64_t elapsed =
            now - mStatWindowStartNs;

        if (elapsed >= 1'000'000'000ULL) {

            double seconds =
                static_cast<double>(
                    elapsed
                ) / 1'000'000'000.0;

            mRealFps =
                static_cast<double>(
                    mRealFrames
                ) / seconds;

            mGeneratedFps =
                static_cast<double>(
                    mGeneratedFrames
                ) / seconds;

            mRealFrames = 0;
            mGeneratedFrames = 0;

            mStatWindowStartNs =
                now;
        }
    }

    // --------------------------------------------------------
    // FPS overlay
    // --------------------------------------------------------

    void drawFpsCounter() {

        if (!mFpsCounter.load())
            return;

        if (!mOverlayProgram)
            return;

        // Small and deliberately cheap.
        //
        // The overlay is intentionally kept minimal here so
        // it cannot disturb the temporal renderer.

        GlStateGuard guard;

        glUseProgram(
            mOverlayProgram
        );

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);

        glUniform4f(
            mOverlayColorLoc,
            1.0f,
            1.0f,
            1.0f,
            1.0f
        );

        // FPS indicator: tiny top-left bar.
        //
        // This avoids changing the temporal textures.

        GLfloat bar[] = {

            -0.98f,  0.94f,
            -0.70f,  0.94f,
            -0.98f,  0.90f,
            -0.70f,  0.90f
        };

        glEnableVertexAttribArray(
            static_cast<GLuint>(
                mOverlayPosLoc
            )
        );

        glVertexAttribPointer(
            static_cast<GLuint>(
                mOverlayPosLoc
            ),
            2,
            GL_FLOAT,
            GL_FALSE,
            0,
            bar
        );

        glDrawArrays(
            GL_TRIANGLE_STRIP,
            0,
            4
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(
                mOverlayPosLoc
            )
        );
    }

    // --------------------------------------------------------
    // Main swap handler
    // --------------------------------------------------------

    EGLBoolean handleSwapBuffers(
        EGLDisplay display,
        EGLSurface surface
    ) {

        if (!mOriginalSwapBuffers)
            return EGL_FALSE;

        uint64_t hookNow =
            nowNs();

        mDisplay = display;
        mSurface = surface;

        // ----------------------------------------------------
        // Disabled / Mode 0
        // ----------------------------------------------------

        int mode =
            std::clamp(
                mGenerationMode.load(),
                0,
                2
            );

        if (
            !mEnabled.load() ||
            mode == 0
        ) {

            resetTimingOnly();

            drawFpsCounter();

            EGLBoolean result =
                mOriginalSwapBuffers(
                    display,
                    surface
                );

            updateStats(false);

            return result;
        }

        // ----------------------------------------------------
        // Surface size
        // ----------------------------------------------------

        EGLint width = 0;
        EGLint height = 0;

        eglQuerySurface(
            display,
            surface,
            EGL_WIDTH,
            &width
        );

        eglQuerySurface(
            display,
            surface,
            EGL_HEIGHT,
            &height
        );

        if (
            width <= 0 ||
            height <= 0
        ) {

            return mOriginalSwapBuffers(
                display,
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
                display,
                surface
            );
        }

        // ----------------------------------------------------
        // First frame
        // ----------------------------------------------------

        if (
            !mHistoryValid ||
            mLastHookStartNs == 0
        ) {

            captureCurrentFrame();

            copyCurrentToBothHistory();

            mLastHookStartNs =
                hookNow;

            mLastPresentedTimeNs =
                hookNow;

            mLastRealFrameNs =
                hookNow;

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            drawFpsCounter();

            EGLBoolean result =
                mOriginalSwapBuffers(
                    display,
                    surface
                );

            updateStats(false);

            return result;
        }

        // ----------------------------------------------------
        // Measure actual hook interval
        // ----------------------------------------------------

        uint64_t frameIntervalNs =
            hookNow -
            mLastHookStartNs;

        mLastHookStartNs =
            hookNow;

        // ----------------------------------------------------
        // Bad timing
        // ----------------------------------------------------

        if (
            frameIntervalNs <
            kMinimumUsefulIntervalNs ||
            frameIntervalNs >
            kMaxGenerationIntervalNs
        ) {

            captureCurrentFrame();

            copyCurrentToBothHistory();

            mLastPresentedTimeNs =
                hookNow;

            mLastRealFrameNs =
                hookNow;

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            drawFpsCounter();

            EGLBoolean result =
                mOriginalSwapBuffers(
                    display,
                    surface
                );

            updateStats(false);

            return result;
        }

        frameIntervalNs =
            std::clamp(
                frameIntervalNs,
                kMinFrameIntervalNs,
                kMaxFrameIntervalNs
            );

        // ----------------------------------------------------
        // Capture actual current game frame
        // ----------------------------------------------------

        captureCurrentFrame();

        // ----------------------------------------------------
        // GPU PERSPECTIVE DETECTION
        // ----------------------------------------------------
        //
        // This happens BEFORE generating anything.
        //
        // If the camera moved far enough:
        //
        //     old temporal history is discarded.
        //
        // No generated frames are produced from the old view.
        //

        bool majorChange =
            detectMajorFrameChange();

        if (majorChange) {

            // ------------------------------------------------
            // HARD RESET
            // ------------------------------------------------

            copyCurrentToBothHistory();

            mLastPresentedTimeNs =
                hookNow;

            mLastRealFrameNs =
                hookNow;

            mLastHookStartNs =
                hookNow;

            // Exact current frame only.
            //
            // No interpolation.
            // No generated frame.
            // No old camera image.

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            drawFpsCounter();

            EGLBoolean result =
                mOriginalSwapBuffers(
                    display,
                    surface
                );

            updateStats(false);

            return result;
        }

        // ----------------------------------------------------
        // Normal generation
        // ----------------------------------------------------

        uint64_t realStart =
            mLastPresentedTimeNs;

        if (realStart == 0)
            realStart = hookNow;

        uint64_t targetRealTimestamp =
            realStart +
            frameIntervalNs;

        // Never allow presentation time to move backwards.

        if (
            targetRealTimestamp <=
            mLastPresentedTimeNs
        ) {

            targetRealTimestamp =
                mLastPresentedTimeNs + 1;
        }

        // Do not generate if the target is wildly in the future.

        if (
            targetRealTimestamp >
            hookNow +
            frameIntervalNs
        ) {

            copyCurrentToBothHistory();

            mLastPresentedTimeNs =
                hookNow;

            drawBlended(
                0.0f,
                0.0f,
                1.0f
            );

            drawFpsCounter();

            EGLBoolean result =
                mOriginalSwapBuffers(
                    display,
                    surface
                );

            updateStats(false);

            return result;
        }

        // ----------------------------------------------------
        // Generate intermediate frames
        // ----------------------------------------------------

        int steps =
            std::clamp(
                mode,
                0,
                2
            );

        for (
            int i = 1;
            i <= steps;
            ++i
        ) {

            float t =
                static_cast<float>(i) /
                static_cast<float>(steps + 1);

            uint64_t timestamp =
                realStart +
                static_cast<uint64_t>(
                    static_cast<double>(
                        frameIntervalNs
                    ) *
                    static_cast<double>(t)
                );

            // Timestamp safety.

            if (
                timestamp <=
                mLastPresentedTimeNs
            ) {
                continue;
            }

            if (
                timestamp >=
                targetRealTimestamp
            ) {
                continue;
            }

            // Never submit stale/future timestamps.

            uint64_t currentTime =
                nowNs();

            if (
                timestamp >
                currentTime +
                frameIntervalNs
            ) {
                continue;
            }

            if (mPresentationTime) {

                mPresentationTime(
                    display,
                    surface,
                    static_cast<EGLnsecsANDROID>(
                        timestamp
                    )
                );
            }

            // Zero global motion vector intentionally.
            //
            // The temporal rejection system now prevents
            // perspective changes from being blended.
            //
            // This is safer than inventing incorrect camera
            // motion vectors.

            drawBlended(
                0.0f,
                0.0f,
                t
            );

            mOriginalSwapBuffers(
                display,
                surface
            );

            mLastPresentedTimeNs =
                timestamp;

            updateStats(true);
        }

        // ----------------------------------------------------
        // Final REAL frame
        // ----------------------------------------------------

        uint64_t finalTimestamp =
            std::max(
                targetRealTimestamp,
                mLastPresentedTimeNs + 1
            );

        if (mPresentationTime) {

            mPresentationTime(
                display,
                surface,
                static_cast<EGLnsecsANDROID>(
                    finalTimestamp
                )
            );
        }

        // t = 1 means the current frame wins.
        //
        // The temporal rejection shader also guarantees that
        // changed areas use the current frame.

        drawBlended(
            0.0f,
            0.0f,
            1.0f
        );

        drawFpsCounter();

        EGLBoolean result =
            mOriginalSwapBuffers(
                display,
                surface
            );

        mLastPresentedTimeNs =
            finalTimestamp;

        mLastRealFrameNs =
            hookNow;

        updateStats(false);

        // ----------------------------------------------------
        // IMPORTANT:
        //
        // Current becomes previous ONLY after the complete
        // real frame has been presented.
        //
        // This keeps the temporal history clean.
        // ----------------------------------------------------

        swapHistory();

        return result;
    }

    // --------------------------------------------------------
    // Swap history
    // --------------------------------------------------------

    void swapHistory() {

        std::swap(
            mTexPrev,
            mTexCur
        );

        mHistoryValid = true;
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    void destroyGlResources() {

        if (mTexPrev) {

            glDeleteTextures(
                1,
                &mTexPrev
            );

            mTexPrev = 0;
        }

        if (mTexCur) {

            glDeleteTextures(
                1,
                &mTexCur
            );

            mTexCur = 0;
        }

        if (mDetectionTex) {

            glDeleteTextures(
                1,
                &mDetectionTex
            );

            mDetectionTex = 0;
        }

        if (mDetectionFbo) {

            glDeleteFramebuffers(
                1,
                &mDetectionFbo
            );

            mDetectionFbo = 0;
        }

        if (mBlendProgram) {

            glDeleteProgram(
                mBlendProgram
            );

            mBlendProgram = 0;
        }

        if (mDetectionProgram) {

            glDeleteProgram(
                mDetectionProgram
            );

            mDetectionProgram = 0;
        }

        if (mOverlayProgram) {

            glDeleteProgram(
                mOverlayProgram
            );

            mOverlayProgram = 0;
        }

        mInitialized = false;
        mHistoryValid = false;

        mWidth = 0;
        mHeight = 0;
    }

    void cleanup() {

        mEnabled.store(false);

        mSwapHook.reset();

        destroyGlResources();
    }
};

} // namespace

// ------------------------------------------------------------
// Levi module registration
// ------------------------------------------------------------

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
);
