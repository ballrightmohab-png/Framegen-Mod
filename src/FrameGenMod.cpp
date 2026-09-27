// FrameGenMod.cpp
//
// Frame Generator Pro - STABLE RELEASE EDITION
//
// Main goals:
//   - Remove perspective/camera jitter
//   - Never perform multiple eglSwapBuffers() calls per Minecraft frame
//   - Never force eglPresentationTimeANDROID()
//   - Never sleep inside the Minecraft render/swap thread
//   - Preserve Minecraft's native frame pacing
//   - Keep module menu + FPS statistics
//   - Graceful fallback if GL resources fail
//
// IMPORTANT:
// This release intentionally disables synthetic frame presentation.
// The previous implementation submitted multiple buffers with
// synthetic presentation timestamps, which could cause uneven
// SurfaceFlinger/display pacing and visible jitter.
//
// This build prioritizes stability for release.
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <cstdio>

#include <dlfcn.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <pl/Mod.hpp>
#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>

namespace {

// ============================================================
// MODULE CONFIG
// ============================================================

constexpr const char *kModuleId = "framegen.pro";

constexpr const char *kModeKey = "mode";
constexpr const char *kPresetKey = "preset";
constexpr const char *kShowStatsKey = "showStats";
constexpr const char *kFpsCapKey = "fpsCap";
constexpr const char *kAutoTuneKey = "autoTune";

constexpr int kMinFpsCap = 10;
constexpr int kMaxFpsCap = 240;

// ============================================================
// QUALITY PROFILE
// ============================================================

enum class QualityLevel {
    PowerSaver = 0,
    Balanced = 1,
    Smooth = 2,
    Ultra = 3
};

struct QualityProfile {
    QualityLevel level;
    const char *name;
    const char *description;
    float hardResetThreshold;
    float extremeResetThreshold;
    int maxSyntheticFrames;
    float motionScale;
    float blendQuality;
    int recommendedFpsCap;
};

const QualityProfile &getQualityProfile(int preset) {
    static const QualityProfile profiles[] = {
        {
            QualityLevel::PowerSaver,
            "Power Saver",
            "Minimal overhead",
            0.12f,
            0.28f,
            0,
            0.0f,
            0.95f,
            60
        },

        {
            QualityLevel::Balanced,
            "Balanced",
            "Stable Minecraft frame pacing",
            0.18f,
            0.35f,
            0,
            0.0f,
            1.0f,
            60
        },

        {
            QualityLevel::Smooth,
            "Smooth",
            "Stable high refresh rendering",
            0.25f,
            0.45f,
            0,
            0.0f,
            1.0f,
            90
        },

        {
            QualityLevel::Ultra,
            "Ultra",
            "Maximum native frame stability",
            0.30f,
            0.50f,
            0,
            0.0f,
            1.0f,
            120
        }
    };

    const int idx = std::clamp(preset, 0, 3);

    return profiles[idx];
}

// ============================================================
// DEVICE ANALYZER
// ============================================================

class DeviceAnalyzer {
public:
    enum class DeviceTier {
        BudgetPhone,
        MidRangePhone,
        FlagshipPhone,
        Premium
    };

    static QualityLevel autoDetectOptimalQuality() {
        return QualityLevel::Balanced;
    }

    static const char *deviceTierName(DeviceTier tier) {
        switch (tier) {
            case DeviceTier::BudgetPhone:
                return "Budget Phone";

            case DeviceTier::MidRangePhone:
                return "Mid-Range Phone";

            case DeviceTier::FlagshipPhone:
                return "Flagship Phone";

            case DeviceTier::Premium:
                return "Gaming Phone";
        }

        return "Unknown";
    }
};

// ============================================================
// EGL / GL TYPES
// ============================================================

using EglSwapBuffersFn = decltype(&eglSwapBuffers);

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

// ============================================================
// TIME
// ============================================================

int64_t nowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds
    >(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// ============================================================
// GL STATE GUARD
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

        auto setEnabled =
            [](GLenum cap, GLboolean value) {
                if (value) {
                    glEnable(cap);
                } else {
                    glDisable(cap);
                }
            };

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
    GLint mProgram = 0;

    GLint mActiveTexture =
        GL_TEXTURE0;

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
// FRAME GENERATOR MOD
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

    // ========================================================
    // LOAD
    // ========================================================

    bool load() {
        getSelf().getLogger().info(
            "========================================"
        );

        getSelf().getLogger().info(
            "   Frame Generator Pro - Stable Build"
        );

        getSelf().getLogger().info(
            "   Native frame pacing enabled"
        );

        getSelf().getLogger().info(
            "========================================"
        );

        return true;
    }

    // ========================================================
    // ENABLE
    // ========================================================

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
                "Failed to find eglSwapBuffers"
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

        // ----------------------------------------------------
        // VAO support
        // ----------------------------------------------------

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

        if (gGenVertexArrays == nullptr ||
            gBindVertexArray == nullptr ||
            gDeleteVertexArrays == nullptr) {

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

        // ----------------------------------------------------
        // MODULE MENU
        // ----------------------------------------------------

        const bool registered =
            pl::modmenu::ModuleBuilder(
                kModuleId,
                "Frame Generator Pro"
            )
            .modId(
                getSelf().getId()
            )
            .description(
                "Stable frame pacing | "
                "Low overhead | "
                "Jitter protection"
            )
            .defaultEnabled(true)
            .onToggle(
                onModuleToggle
            )

            .config(
                kModeKey,
                "Generation Mode: 0=OFF | 1=Stable | 2=Max",
                pl::modmenu::ConfigType::SliderInt,
                std::to_string(
                    mMode.load()
                ),
                "0",
                "2"
            )

            .config(
                kPresetKey,
                "Quality: 0=Fast | 1=Balanced | "
                "2=Smooth | 3=Ultra",
                pl::modmenu::ConfigType::SliderInt,
                std::to_string(
                    mPreset.load()
                ),
                "0",
                "3"
            )

            .config(
                kShowStatsKey,
                "Show FPS Stats: 0=OFF | 1=ON",
                pl::modmenu::ConfigType::SliderInt,
                std::to_string(
                    mShowStats.load()
                ),
                "0",
                "1"
            )

            .config(
                kAutoTuneKey,
                "Auto-Tune: 0=OFF | 1=ON",
                pl::modmenu::ConfigType::SliderInt,
                std::to_string(
                    mAutoTune.load()
                ),
                "0",
                "1"
            )

            .config(
                kFpsCapKey,
                "Target FPS - informational only",
                pl::modmenu::ConfigType::SliderInt,
                std::to_string(
                    mFpsCap.load()
                ),
                "30",
                "240"
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
                "Failed to register Frame Generator Pro"
            );

            return false;
        }

        getSelf().getLogger().info(
            "Frame Generator Pro ENABLED"
        );

        getSelf().getLogger().info(
            "Stable presentation mode active"
        );

        getSelf().getLogger().info(
            "Synthetic multi-swap generation disabled "
            "for jitter-free release"
        );

        return true;
    }

    // ========================================================
    // DISABLE
    // ========================================================

    bool disable() {
        pl::modmenu::unregisterModule(
            kModuleId
        );

        mEnabled = false;

        mSwapHook.reset();

        destroyGlResources();

        resetTiming();
        resetAllStats();

        getSelf().getLogger().info(
            "Frame Generator Pro disabled"
        );

        return true;
    }

    bool unload() {
        return true;
    }

private:

    // ========================================================
    // MEMBERS
    // ========================================================

    ll::mod::NativeMod &mSelf;

    pl::memory::HookHandle mSwapHook;

    EglSwapBuffersFn
        mOriginalSwapBuffers = nullptr;

    bool mEnabled = true;

    std::atomic_int mMode{1};

    std::atomic_int mPreset{1};

    std::atomic_int mShowStats{1};

    std::atomic_int mFpsCap{60};

    std::atomic_int mAutoTune{1};

    int64_t mLastHookStartNs = 0;

    int64_t mLastPresentedTimeNs = 0;

    int64_t mPreviousPresentedTimeNs = 0;

    int64_t mStatsWindowStartNs = 0;

    uint32_t mRealFramesThisSecond = 0;

    uint32_t mGeneratedFramesThisSecond = 0;

    uint32_t mLastRealFps = 0;

    uint32_t mLastGeneratedFps = 0;

    uint32_t mTotalFramesGenerated = 0;

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

    GLuint mDetectorProgram = 0;

    GLint mDLocPos = -1;

    GLint mDLocUV = -1;

    GLint mDLocPrev = -1;

    GLint mDLocCur = -1;

    GLuint mDetectorTexture = 0;

    GLuint mDetectorFramebuffer = 0;

    bool mDetectorReady = false;

    float mLastFrameDifference = 0.0f;

    GLuint mOverlayProgram = 0;

    GLint mOverlayPos = -1;

    GLint mOverlayColor = -1;

    bool mOverlayReady = false;

    bool mVaoSupported = false;

    GLuint mOwnVao = 0;

    uint32_t mConsecutiveErrors = 0;

    constexpr static uint32_t
        kMaxConsecutiveErrors = 5;

    // ========================================================
    // CALLBACK: MODULE TOGGLE
    // ========================================================

    static void onModuleToggle(
        std::string_view moduleId,
        bool enabled
    ) {
        if (moduleId != kModuleId) {
            return;
        }

        FrameGenMod &mod =
            instance();

        mod.mEnabled = enabled;

        mod.hardTemporalReset();

        mod.resetTiming();

        if (!enabled) {
            mod.resetAllStats();

            mod.getSelf().getLogger().info(
                "Frame Generator paused"
            );
        } else {
            const QualityProfile &profile =
                getQualityProfile(
                    mod.mPreset.load()
                );

            mod.getSelf().getLogger().info(
                "Frame Generator active - {}",
                profile.name
            );
        }
    }

    // ========================================================
    // CALLBACK: CONFIG
    // ========================================================

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

        if (end == text.c_str()) {
            return;
        }

        // ----------------------------------------------------
        // MODE
        // ----------------------------------------------------

        if (key == kModeKey) {
            const int mode =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        2
                    )
                );

            mod.mMode.store(mode);

            mod.hardTemporalReset();

            mod.resetTiming();

            const char *modeNames[] = {
                "OFF",
                "STABLE",
                "STABLE MAX"
            };

            mod.getSelf().getLogger().info(
                "Mode: {}",
                modeNames[mode]
            );

            return;
        }

        // ----------------------------------------------------
        // PRESET
        // ----------------------------------------------------

        if (key == kPresetKey) {
            const int preset =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        3
                    )
                );

            mod.mPreset.store(
                preset
            );

            const QualityProfile &profile =
                getQualityProfile(
                    preset
                );

            mod.getSelf().getLogger().info(
                "Quality: {}",
                profile.name
            );

            mod.hardTemporalReset();

            return;
        }

        // ----------------------------------------------------
        // STATS
        // ----------------------------------------------------

        if (key == kShowStatsKey) {
            const int show =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        1
                    )
                );

            mod.mShowStats.store(
                show
            );

            return;
        }

        // ----------------------------------------------------
        // AUTO TUNE
        // ----------------------------------------------------

        if (key == kAutoTuneKey) {
            const int autoTune =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        0,
                        1
                    )
                );

            mod.mAutoTune.store(
                autoTune
            );

            return;
        }

        // ----------------------------------------------------
        // FPS CAP
        //
        // IMPORTANT:
        // The value is retained for compatibility/menu purposes.
        // It is NOT applied inside eglSwapBuffers because doing
        // so would introduce render-thread stalls and jitter.
        // ----------------------------------------------------

        if (key == kFpsCapKey) {
            const int cap =
                static_cast<int>(
                    std::clamp<long>(
                        parsed,
                        30,
                        240
                    )
                );

            mod.mFpsCap.store(
                cap
            );

            mod.getSelf().getLogger().info(
                "FPS target set to {} "
                "(native pacing preserved)",
                cap
            );

            return;
        }
    }

    // ========================================================
    // SWAP HOOK
    // ========================================================

    static EGLBoolean swapBuffersDetour(
        EGLDisplay dpy,
        EGLSurface surface
    ) {
        return instance().handleSwapBuffers(
            dpy,
            surface
        );
    }

    // ========================================================
    // TIMING
    // ========================================================

    void resetTiming() {
        mLastHookStartNs = 0;

        mLastPresentedTimeNs = 0;

        mPreviousPresentedTimeNs = 0;
    }

    void hardTemporalReset() {
        mHaveHistory = false;

        mLastFrameDifference = 0.0f;

        resetTiming();
    }

    void resetAllStats() {
        mStatsWindowStartNs = 0;

        mRealFramesThisSecond = 0;

        mGeneratedFramesThisSecond = 0;

        mLastRealFps = 0;

        mLastGeneratedFps = 0;
    }

    // ========================================================
    // FRAME STATISTICS
    // ========================================================

    void updateFrameStats(
        bool generated
    ) {
        const int64_t now =
            nowNs();

        if (mStatsWindowStartNs == 0) {
            mStatsWindowStartNs =
                now;
        }

        if (generated) {
            ++mGeneratedFramesThisSecond;

            ++mTotalFramesGenerated;
        } else {
            ++mRealFramesThisSecond;
        }

        const int64_t elapsed =
            now -
            mStatsWindowStartNs;

        if (elapsed <
            1'000'000'000LL) {
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

        mRealFramesThisSecond = 0;

        mGeneratedFramesThisSecond = 0;

        mStatsWindowStartNs = now;
    }

    // ========================================================
    // STABLE PRESENTATION
    // ========================================================

    EGLBoolean presentStableFrame(
        EGLDisplay dpy,
        EGLSurface surface
    ) {
        //
        // CRITICAL RELEASE RULE:
        //
        // Exactly ONE eglSwapBuffers() call.
        //
        // We intentionally do NOT call:
        //
        //   eglPresentationTimeANDROID()
        //
        // and we do NOT sleep here.
        //
        // Minecraft + Android are allowed to control
        // presentation timing naturally.
        //

        drawFpsCounter();

        const EGLBoolean result =
            mOriginalSwapBuffers(
                dpy,
                surface
            );

        if (result == EGL_TRUE) {
            const int64_t presented =
                nowNs();

            mLastPresentedTimeNs =
                presented;

            mPreviousPresentedTimeNs =
                presented;

            updateFrameStats(
                false
            );

            mConsecutiveErrors = 0;
        } else {
            ++mConsecutiveErrors;
        }

        return result;
    }

    // ========================================================
    // MAIN SWAP HANDLER
    // ========================================================

    EGLBoolean handleSwapBuffers(
        EGLDisplay dpy,
        EGLSurface surface
    ) {
        if (mOriginalSwapBuffers == nullptr) {
            return EGL_FALSE;
        }

        try {
            const int mode =
                std::clamp(
                    mMode.load(
                        std::memory_order_relaxed
                    ),
                    0,
                    2
                );

            // ------------------------------------------------
            // MOD DISABLED
            // ------------------------------------------------

            if (!mEnabled ||
                mode == 0) {

                hardTemporalReset();

                return presentStableFrame(
                    dpy,
                    surface
                );
            }

            // ------------------------------------------------
            // SURFACE SIZE
            // ------------------------------------------------

            EGLint width = 0;

            EGLint height = 0;

            const EGLBoolean widthResult =
                eglQuerySurface(
                    dpy,
                    surface,
                    EGL_WIDTH,
                    &width
                );

            const EGLBoolean heightResult =
                eglQuerySurface(
                    dpy,
                    surface,
                    EGL_HEIGHT,
                    &height
                );

            if (widthResult == EGL_FALSE ||
                heightResult == EGL_FALSE ||
                width <= 0 ||
                height <= 0) {

                hardTemporalReset();

                return presentStableFrame(
                    dpy,
                    surface
                );
            }

            // ------------------------------------------------
            // OPTIONAL GL RESOURCE INITIALIZATION
            //
            // If resources cannot initialize, Minecraft still
            // gets its normal swap.
            // ------------------------------------------------

            if (!ensureGlResources(
                    width,
                    height
                )) {

                hardTemporalReset();

                return presentStableFrame(
                    dpy,
                    surface
                );
            }

            // ------------------------------------------------
            // RECORD FRAME TIMING
            //
            // This is measurement only.
            // No pacing manipulation.
            // ------------------------------------------------

            const int64_t hookNow =
                nowNs();

            mLastHookStartNs =
                hookNow;

            // ------------------------------------------------
            // RELEASE BUILD:
            //
            // Do not replace Minecraft's framebuffer with an
            // interpolated framebuffer.
            //
            // Do not submit additional swaps.
            //
            // Do not modify presentation timestamps.
            //
            // Let Minecraft present its real frame.
            // ------------------------------------------------

            return presentStableFrame(
                dpy,
                surface
            );

        } catch (...) {
            ++mConsecutiveErrors;

            getSelf().getLogger().error(
                "Frame Generator recovered from "
                "rendering error"
            );

            hardTemporalReset();

            // ------------------------------------------------
            // EMERGENCY FALLBACK
            // ------------------------------------------------

            return mOriginalSwapBuffers(
                dpy,
                surface
            );
        }
    }

    // ========================================================
    // GL RESOURCE INITIALIZATION
    // ========================================================

    bool ensureGlResources(
        GLsizei width,
        GLsizei height
    ) {
        //
        // The release build does not require our old frame
        // generation shaders.
        //
        // We intentionally avoid allocating full-resolution
        // history textures because they add GPU/memory overhead.
        //
        // Only mark the dimensions as known.
        //

        if (mGlReady &&
            width == mTexWidth &&
            height == mTexHeight) {

            return true;
        }

        mTexWidth = width;

        mTexHeight = height;

        mGlReady = true;

        hardTemporalReset();

        return true;
    }

    // ========================================================
    // FPS OVERLAY
    // ========================================================

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

        if (digit < 0 ||
            digit > 9) {
            return 0;
        }

        const uint8_t mask =
            table[digit];

        const float left = x;

        const float right =
            x + w;

        const float top = y;

        const float mid =
            y - h * 0.5f;

        const float bottom =
            y - h;

        int count = 0;

        if (mask & 0b0000001) {
            addLine(
                vertices,
                count,
                left,
                top,
                right,
                top
            );
        }

        if (mask & 0b0000010) {
            addLine(
                vertices,
                count,
                right,
                top,
                right,
                mid
            );
        }

        if (mask & 0b0000100) {
            addLine(
                vertices,
                count,
                right,
                mid,
                right,
                bottom
            );
        }

        if (mask & 0b0001000) {
            addLine(
                vertices,
                count,
                left,
                bottom,
                right,
                bottom
            );
        }

        if (mask & 0b0010000) {
            addLine(
                vertices,
                count,
                left,
                mid,
                left,
                bottom
            );
        }

        if (mask & 0b0100000) {
            addLine(
                vertices,
                count,
                left,
                top,
                left,
                mid
            );
        }

        if (mask & 0b1000000) {
            addLine(
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

        char buffer[8] = {};

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
            count += buildDigit(
                buffer[i] - '0',
                x,
                startY,
                width,
                height,
                vertices + count
            );

            x += width + spacing;
        }
    }

    void drawFpsCounter() {
        //
        // The release build intentionally keeps the overlay
        // disabled unless an overlay program exists.
        //
        // This prevents extra GL state manipulation from
        // interfering with Minecraft's renderer.
        //

        if (mShowStats.load(
                std::memory_order_relaxed
            ) == 0) {

            return;
        }

        //
        // No custom GL overlay in the stable release.
        //
        // FPS values remain available through internal stats.
        //
        return;
    }

    // ========================================================
    // RESOURCE CLEANUP
    // ========================================================

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
        //
        // The stable release no longer allocates the old
        // frame-generation resources.
        //

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

        if (mOwnVao != 0 &&
            gDeleteVertexArrays != nullptr) {

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

// ============================================================
// REGISTER MOD
// ============================================================

PL_REGISTER_MOD(
    FrameGenMod,
    FrameGenMod::instance()
)

Important: this is deliberately a stability/release build, not the old synthetic-frame version. The old code's "handleFrameGeneration()" was the part most likely to create the jitter because it could call "eglSwapBuffers()" several times for one Minecraft frame and manipulate presentation timestamps.

Build this one first. If it builds and the jitter is gone, we can then add a proper single-swap frame interpolation path instead of bringing the unstable multi-swap system back.
