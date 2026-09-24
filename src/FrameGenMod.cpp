// FrameGenMod.cpp
//
// Stable FrameGen diagnostic build for LeviLaunchroid.
//
// IMPORTANT:
// This version intentionally DOES NOT use glReadPixels.
// Motion estimation is disabled and motion vector is always 0,0.
//
// The goal is to test whether the previous synchronous GPU->CPU
// readback / motion estimation was causing jitter.
//
// Features:
//   - 1x or 2x generated frames
//   - Safer frame timing
//   - Presentation timestamps set BEFORE eglSwapBuffers
//   - Skips synthetic frames if their target time is already late
//   - Private VAO isolation
//   - GL state restoration
//   - No glReadPixels
//   - No CPU/GPU readback
//
// Generation Mode:
//   0 = Off
//   1 = 1 generated frame
//   2 = 2 generated frames
//
// Quality Preset is kept in the menu for compatibility, but motion
// estimation is disabled in this diagnostic build.

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

constexpr const char *kModuleId = "framegen.core";
constexpr const char *kModeKey = "generationMode";
constexpr const char *kPresetKey = "qualityPreset";

constexpr int64_t kMinFrameIntervalNs = 2'000'000LL;   // 2 ms
constexpr int64_t kMaxFrameIntervalNs = 80'000'000LL;  // 80 ms

// Fullscreen quad: x, y, u, v
constexpr float kQuadVerts[16] = {
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
     1.0f,  1.0f, 1.0f, 1.0f
};

constexpr const char *kVertexShaderSrc = R"glsl(
attribute vec2 aPos;
attribute vec2 aUV;

varying vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)glsl";

// Motion vector is intentionally still supported by the shader,
// but this diagnostic build always supplies 0,0.
//
// Explicit UV clamping prevents sampling outside the valid image.
constexpr const char *kBlendFragmentShaderSrc = R"glsl(
precision mediump float;

varying vec2 vUV;

uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform vec2 uMv;
uniform float uT;

void main() {

    vec2 uvPrev = vUV - uT * uMv;
    vec2 uvCur  = vUV + (1.0 - uT) * uMv;

    uvPrev = clamp(uvPrev, 0.0, 1.0);
    uvCur  = clamp(uvCur, 0.0, 1.0);

    vec4 prevColor = texture2D(uPrev, uvPrev);
    vec4 curColor  = texture2D(uCur, uvCur);

    gl_FragColor = mix(prevColor, curColor, uT);
}
)glsl";

using EglSwapBuffersFn = decltype(&eglSwapBuffers);

using PFN_eglPresentationTimeANDROID =
    EGLBoolean(EGLAPIENTRY *)(EGLDisplay,
                               EGLSurface,
                               int64_t);

// VAO functions.
using PFN_glGenVertexArrays =
    void (*)(GLsizei, GLuint *);

using PFN_glBindVertexArray =
    void (*)(GLuint);

using PFN_glDeleteVertexArrays =
    void (*)(GLsizei, const GLuint *);

constexpr GLenum kVertexArrayBindingPname = 0x85B5;

PFN_glGenVertexArrays gGenVertexArrays = nullptr;
PFN_glBindVertexArray gBindVertexArray = nullptr;
PFN_glDeleteVertexArrays gDeleteVertexArrays = nullptr;

bool hasEglExtension(EGLDisplay display, const char *name) {
    const char *extensions = eglQueryString(display, EGL_EXTENSIONS);

    if (extensions == nullptr || name == nullptr || name[0] == '\0') {
        return false;
    }

    const size_t nameLength = std::strlen(name);
    const char *match = extensions;

    while ((match = std::strstr(match, name)) != nullptr) {

        const bool startsAtToken =
            match == extensions || match[-1] == ' ';

        const char terminator = match[nameLength];

        if (startsAtToken &&
            (terminator == ' ' || terminator == '\0')) {
            return true;
        }

        match += nameLength;
    }

    return false;
}

class GlStateGuard {
public:

    GlStateGuard() {

        glGetIntegerv(GL_CURRENT_PROGRAM, &mProgram);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &mActiveTexture);
        glGetIntegerv(GL_VIEWPORT, mViewport);
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &mFramebuffer);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &mArrayBuffer);

        glGetIntegerv(
            kVertexArrayBindingPname,
            &mVertexArray
        );

        mDepthTest = glIsEnabled(GL_DEPTH_TEST);
        mBlend = glIsEnabled(GL_BLEND);
        mCullFace = glIsEnabled(GL_CULL_FACE);
        mScissorTest = glIsEnabled(GL_SCISSOR_TEST);
        mStencilTest = glIsEnabled(GL_STENCIL_TEST);

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
            static_cast<GLenum>(mActiveTexture)
        );
    }

    ~GlStateGuard() {

        if (gBindVertexArray != nullptr) {
            gBindVertexArray(
                static_cast<GLuint>(mVertexArray)
            );
        }

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            static_cast<GLuint>(mFramebuffer)
        );

        glBindBuffer(
            GL_ARRAY_BUFFER,
            static_cast<GLuint>(mArrayBuffer)
        );

        glViewport(
            mViewport[0],
            mViewport[1],
            mViewport[2],
            mViewport[3]
        );

        glUseProgram(
            static_cast<GLuint>(mProgram)
        );

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(mTex0)
        );

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(
            GL_TEXTURE_2D,
            static_cast<GLuint>(mTex1)
        );

        glActiveTexture(
            static_cast<GLenum>(mActiveTexture)
        );

        setEnabled(GL_DEPTH_TEST, mDepthTest);
        setEnabled(GL_BLEND, mBlend);
        setEnabled(GL_CULL_FACE, mCullFace);
        setEnabled(GL_SCISSOR_TEST, mScissorTest);
        setEnabled(GL_STENCIL_TEST, mStencilTest);

        glColorMask(
            mColorMask[0],
            mColorMask[1],
            mColorMask[2],
            mColorMask[3]
        );
    }

    GlStateGuard(const GlStateGuard &) = delete;
    GlStateGuard &operator=(const GlStateGuard &) = delete;

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
    GLint mActiveTexture = GL_TEXTURE0;

    GLint mViewport[4] = {
        0, 0, 0, 0
    };

    GLint mFramebuffer = 0;
    GLint mArrayBuffer = 0;
    GLint mVertexArray = 0;

    GLint mTex0 = 0;
    GLint mTex1 = 0;

    GLboolean mDepthTest = GL_FALSE;
    GLboolean mBlend = GL_FALSE;
    GLboolean mCullFace = GL_FALSE;
    GLboolean mScissorTest = GL_FALSE;
    GLboolean mStencilTest = GL_FALSE;

    GLboolean mColorMask[4] = {
        GL_TRUE,
        GL_TRUE,
        GL_TRUE,
        GL_TRUE
    };
};

class FrameGenMod {
public:

    static FrameGenMod &instance() {
        static FrameGenMod mod;
        return mod;
    }

    FrameGenMod()
        : mSelf(*ll::mod::NativeMod::current()) {}

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

        mSwapHook = pl::memory::HookHandle(
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

        if (mTrueFrameGenAvailable) {

            getSelf().getLogger().info(
                "eglPresentationTimeANDROID available"
            );

        } else {

            getSelf().getLogger().info(
                "eglPresentationTimeANDROID unavailable"
            );
        }

        // Try OES VAO first.
        gGenVertexArrays =
            reinterpret_cast<PFN_glGenVertexArrays>(
                eglGetProcAddress(
                    "glGenVertexArraysOES"
                )
            );

        gBindVertexArray =
            reinterpret_cast<PFN_glBindVertexArray>(
                eglGetProcAddress(
                    "glBindVertexArrayOES"
                )
            );

        gDeleteVertexArrays =
            reinterpret_cast<PFN_glDeleteVertexArrays>(
                eglGetProcAddress(
                    "glDeleteVertexArraysOES"
                )
            );

        // Try GLES3 core functions.
        if (gGenVertexArrays == nullptr ||
            gBindVertexArray == nullptr ||
            gDeleteVertexArrays == nullptr) {

            gGenVertexArrays =
                reinterpret_cast<PFN_glGenVertexArrays>(
                    eglGetProcAddress(
                        "glGenVertexArrays"
                    )
                );

            gBindVertexArray =
                reinterpret_cast<PFN_glBindVertexArray>(
                    eglGetProcAddress(
                        "glBindVertexArray"
                    )
                );

            gDeleteVertexArrays =
                reinterpret_cast<PFN_glDeleteVertexArrays>(
                    eglGetProcAddress(
                        "glDeleteVertexArrays"
                    )
                );
        }

        mVaoSupported =
            gGenVertexArrays != nullptr &&
            gBindVertexArray != nullptr &&
            gDeleteVertexArrays != nullptr;

        getSelf().getLogger().info(
            mVaoSupported
                ? "VAO isolation available"
                : "VAO isolation unavailable"
        );

        const bool moduleRegistered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator"
            )
                .modId(getSelf().getId())
                .description(
                    "Stable frame generation test build. "
                    "Motion readback disabled."
                )
                .defaultEnabled(true)
                .onToggle(onModuleToggle)

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

                .onConfigChanged(
                    onConfigChanged
                )
                .registerModule();

        if (!moduleRegistered) {

            getSelf().getLogger().error(
                "Failed to register Frame Generator menu module"
            );
        }

        getSelf().getLogger().info(
            "Frame Generator enabled - NO glReadPixels"
        );

        return moduleRegistered;
    }

    bool disable() {

        pl::modmenu::unregisterModule(
            kModuleId
        );

        mSwapHook.reset();

        destroyGlResources();

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

    pl::memory::HookHandle mSwapHook;

    EglSwapBuffersFn
        mOriginalSwapBuffers = nullptr;

    PFN_eglPresentationTimeANDROID
        mPresentationTimeFn = nullptr;

    bool mTrueFrameGenAvailable = false;

    // Previous hook entry time.
    //
    // IMPORTANT:
    // This is updated at the START of each real game frame.
    // It is NOT processing time.
    int64_t mLastHookStartNs = 0;

    // Last timestamp that we actually submitted.
    int64_t mLastPresentedTimeNs = 0;

    std::atomic_bool mEnabled{true};

    std::atomic_int mGenerationMode{1};

    std::atomic_int mQualityPreset{1};

    // GL resources.
    bool mGlReady = false;
    bool mHaveHistory = false;

    GLsizei mTexWidth = 0;
    GLsizei mTexHeight = 0;

    GLuint mTexPrev = 0;
    GLuint mTexCur = 0;

    GLuint mBlendProgram = 0;

    GLint mBLocPos = -1;
    GLint mBLocUV = -1;
    GLint mBLocPrev = -1;
    GLint mBLocCur = -1;
    GLint mBLocMv = -1;
    GLint mBLocT = -1;

    bool mVaoSupported = false;

    GLuint mOwnVao = 0;

    static void onModuleToggle(
        std::string_view moduleId,
        bool enabled
    ) {

        if (moduleId == kModuleId) {

            instance().mEnabled.store(
                enabled,
                std::memory_order_relaxed
            );

            if (!enabled) {
                instance().resetTiming();
            }
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

        const std::string text(value);

        char *end = nullptr;

        const long parsed =
            std::strtol(
                text.c_str(),
                &end,
                10
            );

        if (end == text.c_str()) {
            return;
        }

        if (key == kModeKey) {

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

        } else if (key == kPresetKey) {

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

    static int64_t nowNs() {

        return std::chrono::duration_cast<
            std::chrono::nanoseconds
        >(
            std::chrono::steady_clock::now()
                .time_since_epoch()
        ).count();
    }

    void resetTiming() {

        mHaveHistory = false;

        mLastHookStartNs = 0;
        mLastPresentedTimeNs = 0;
    }

    EGLBoolean handleSwapBuffers(
        EGLDisplay dpy,
        EGLSurface surface
    ) {

        if (mOriginalSwapBuffers == nullptr) {
            return EGL_FALSE;
        }

        const int mode =
            mGenerationMode.load(
                std::memory_order_relaxed
            );

        if (!mEnabled.load(
                std::memory_order_relaxed
            ) ||
            mode == 0) {

            resetTiming();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        // We require presentation timestamps and VAO isolation.
        if (!mVaoSupported ||
            !mTrueFrameGenAvailable ||
            !hasEglExtension(
                dpy,
                "EGL_ANDROID_presentation_time"
            )) {

            resetTiming();

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        EGLint width = 0;
        EGLint height = 0;

        if (eglQuerySurface(
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
            height <= 0) {

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        if (!ensureGlResources(
                width,
                height
            )) {

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }

        // Capture the actual start time of THIS real game frame.
        const int64_t hookNowNs = nowNs();

        // Calculate REAL frame-to-frame interval.
        //
        // This is intentionally calculated before doing our own
        // rendering work. That prevents our processing time from
        // being mistaken for the game's frame interval.
        int64_t frameIntervalNs = 0;

        if (mLastHookStartNs != 0) {

            frameIntervalNs =
                hookNowNs -
                mLastHookStartNs;
        }

        // Update immediately for the NEXT real frame.
        mLastHookStartNs = hookNowNs;

        GlStateGuard stateGuard;

        // Copy the game's newly rendered frame.
        captureCurrentFrame();

        // First frame:
        // We have no previous frame to interpolate from.
        if (!mHaveHistory) {

            const int64_t firstTimestamp =
                std::max(
                    hookNowNs,
                    mLastPresentedTimeNs + 1
                );

            mPresentationTimeFn(
                dpy,
                surface,
                firstTimestamp
            );

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (result == EGL_TRUE) {

                mLastPresentedTimeNs =
                    firstTimestamp;

                mHaveHistory = true;

                std::swap(
                    mTexPrev,
                    mTexCur
                );
            } else {

                resetTiming();
            }

            return result;
        }

        // If the real frame interval is crazy,
        // don't try to interpolate across it.
        if (frameIntervalNs <
                kMinFrameIntervalNs ||
            frameIntervalNs >
                kMaxFrameIntervalNs) {

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

            const EGLBoolean result =
                mOriginalSwapBuffers(
                    dpy,
                    surface
                );

            if (result == EGL_TRUE) {

                mLastPresentedTimeNs =
                    timestamp;

                std::swap(
                    mTexPrev,
                    mTexCur
                );

            } else {

                resetTiming();
            }

            return result;
        }

        // ------------------------------------------------------------
        // NO MOTION ESTIMATION
        // ------------------------------------------------------------
        //
        // This is the important test.
        //
        // Previous version:
        //
        //     downsample -> glReadPixels -> CPU SAD search
        //
        // New version:
        //
        //     motion = 0,0
        //
        // This completely removes the synchronous GPU->CPU readback.
        //
        const float mvU = 0.0f;
        const float mvV = 0.0f;

        const int steps = std::clamp(
            mode,
            1,
            2
        );

        // Planned timestamp for the next REAL frame.
        int64_t targetRealTimestamp =
            mLastPresentedTimeNs +
            frameIntervalNs;

        // Never request a timestamp in the past.
        if (targetRealTimestamp < hookNowNs) {
            targetRealTimestamp = hookNowNs;
        }

        // Generate intermediate frames.
        for (int i = 1; i <= steps; ++i) {

            const float t =
                static_cast<float>(i) /
                static_cast<float>(steps + 1);

            const int64_t syntheticTimestamp =
                mLastPresentedTimeNs +
                static_cast<int64_t>(
                    static_cast<double>(
                        targetRealTimestamp -
                        mLastPresentedTimeNs
                    ) *
                    static_cast<double>(t)
                );

            // If the timestamp is already late, don't submit
            // another synthetic frame.
            const int64_t currentTimeNs = nowNs();

            if (syntheticTimestamp <=
                currentTimeNs) {

                continue;
            }

            if (syntheticTimestamp <=
                mLastPresentedTimeNs) {

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

            if (generatedResult != EGL_TRUE) {

                resetTiming();

                return generatedResult;
            }

            mLastPresentedTimeNs =
                syntheticTimestamp;
        }

        // Final current frame.
        //
        // We still render the current texture into the backbuffer
        // because the synthetic swaps changed which surface buffer
        // is currently being displayed.
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

        if (result == EGL_TRUE) {

            mLastPresentedTimeNs =
                finalTimestamp;

            std::swap(
                mTexPrev,
                mTexCur
            );

        } else {

            resetTiming();
        }

        return result;
    }

    bool ensureGlResources(
        GLsizei width,
        GLsizei height
    ) {

        if (mBlendProgram == 0 &&
            !buildBlendProgram()) {

            return false;
        }

        if (mVaoSupported &&
            mOwnVao == 0) {

            gGenVertexArrays(
                1,
                &mOwnVao
            );
        }

        if (mGlReady &&
            width == mTexWidth &&
            height == mTexHeight) {

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

        if (mTexPrev == 0 ||
            mTexCur == 0) {

            destroyTextures();

            mGlReady = false;

            return false;
        }

        mTexWidth = width;
        mTexHeight = height;

        mGlReady = true;

        resetTiming();

        return true;
    }

    static GLuint createFrameTexture(
        GLsizei width,
        GLsizei height
    ) {

        GLuint tex = 0;

        glGenTextures(
            1,
            &tex
        );

        if (tex == 0) {
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

    void captureCurrentFrame() {

        glBindFramebuffer(
            GL_FRAMEBUFFER,
            0
        );

        glBindTexture(
            GL_TEXTURE_2D,
            mTexCur
        );

        // Unlike glCopyTexImage2D, this does NOT redefine/
        // reallocate the texture every frame.
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

    void drawBlended(
        float mvU,
        float mvV,
        float t
    ) {

        if (mVaoSupported) {

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
            static_cast<GLuint>(mBLocPos),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(mBLocPos)
        );

        glVertexAttribPointer(
            static_cast<GLuint>(mBLocUV),
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * sizeof(float),
            kQuadVerts + 2
        );

        glEnableVertexAttribArray(
            static_cast<GLuint>(mBLocUV)
        );

        glDrawArrays(
            GL_TRIANGLE_STRIP,
            0,
            4
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(mBLocPos)
        );

        glDisableVertexAttribArray(
            static_cast<GLuint>(mBLocUV)
        );
    }

    bool buildBlendProgram() {

        GLuint program = 0;

        if (!linkProgram(
                kVertexShaderSrc,
                kBlendFragmentShaderSrc,
                program
            )) {

            return false;
        }

        mBlendProgram = program;

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

        if (vs == 0 ||
            fs == 0) {

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

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint linked = GL_FALSE;

        glGetProgramiv(
            program,
            GL_LINK_STATUS,
            &linked
        );

        if (linked == GL_FALSE) {

            logProgramError(
                program
            );

            glDeleteProgram(
                program
            );

            return false;
        }

        outProgram = program;

        return true;
    }

    GLuint compileShader(
        GLenum type,
        const char *src
    ) {

        const GLuint shader =
            glCreateShader(type);

        glShaderSource(
            shader,
            1,
            &src,
            nullptr
        );

        glCompileShader(
            shader
        );

        GLint compiled = GL_FALSE;

        glGetShaderiv(
            shader,
            GL_COMPILE_STATUS,
            &compiled
        );

        if (compiled == GL_FALSE) {

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

    void logShaderError(
        GLuint shader
    ) {

        GLint len = 0;

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

        GLint len = 0;

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

    void destroyTextures() {

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
    }

    void destroyGlResources() {

        destroyTextures();

        if (mBlendProgram != 0) {

            glDeleteProgram(
                mBlendProgram
            );

            mBlendProgram = 0;
        }

        if (mOwnVao != 0 &&
            gDeleteVertexArrays != nullptr) {

            gDeleteVertexArrays(
                1,
                &mOwnVao
            );

            mOwnVao = 0;
        }

        mGlReady = false;
        mHaveHistory = false;

        mTexWidth = 0;
        mTexHeight = 0;

        resetTiming();
    }
};

} // namespace

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
)
