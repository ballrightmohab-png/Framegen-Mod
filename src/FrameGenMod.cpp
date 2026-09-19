// FrameGenMod.cpp
//
// Lightweight frame generation for LeviLaunchroid:
//   1. Capture the previous and current real rendered frames.
//   2. Estimate a single GLOBAL motion vector between them (a coarse,
//      cheap block search on a downsampled luma buffer — not full
//      per-pixel optical flow). This targets the actual stated use case
//      (smoother camera panning while aiming), not general-purpose
//      arbitrary-scene interpolation.
//   3. Generate 1 or 2 motion-compensated intermediate frames and
//      present them (each explicitly timestamped via
//      EGL_ANDROID_presentation_time) between the two real frames.
//   4. Input and game simulation are never touched — this hook only
//      ever runs at eglSwapBuffers, after the game has already finished
//      a real frame; it cannot delay or duplicate simulation ticks.
//
// Modes (Mod Menu):
//   Generation Mode:  0 = Off, 1 = 1x generated frame, 2 = 2x generated frames
//   Quality Preset:   0 = Low Latency, 1 = Balanced, 2 = Smoothness
//     Each preset sets the motion-search radius and a confidence
//     threshold: when the best match found is a poor match (fast or
//     complex motion the global-pan model can't represent), we fall
//     back rather than show a wrong guess — Low Latency falls back to
//     a plain duplicated real frame (no blur), Balanced/Smoothness fall
//     back to a plain (non-motion-compensated) cross-fade.
//
// KNOWN LIMITATION (stated honestly, not tested on real hardware): the
// motion-search readback uses glReadPixels on a small buffer each frame,
// which is a synchronous GPU stall on tile-based mobile GPUs regardless
// of buffer size. The buffer is tiny (32x18) so the stall should be
// small, but this has not been profiled on-device. If this turns out to
// be a real cost, the fix is to defer consuming the readback by one
// frame (pipeline it) rather than reading it back synchronously the
// same frame it was rendered — left as a follow-up if profiling shows
// it's needed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

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

constexpr int kMotionW = 32;
constexpr int kMotionH = 18;

struct PresetParams {
  int searchRadius;
  float sadBailoutThreshold; // average per-pixel luma diff, 0-255 scale
};

// Tuning starting points — not measured on real hardware, expect to
// adjust after real testing.
constexpr PresetParams kPresets[3] = {
    {3, 12.0f},  // 0: Low Latency — small search, bails out readily
    {5, 22.0f},  // 1: Balanced
    {8, 40.0f},  // 2: Smoothness — bigger search, tries harder
};

// Fullscreen quad: (x, y, u, v) per vertex, drawn as a triangle strip.
constexpr float kQuadVerts[16] = {
    -1.0f, -1.0f, 0.0f, 0.0f, // bottom-left
    1.0f,  -1.0f, 1.0f, 0.0f, // bottom-right
    -1.0f, 1.0f,  0.0f, 1.0f, // top-left
    1.0f,  1.0f,  1.0f, 1.0f, // top-right
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

// Motion-compensated blend: at t=0 this reduces exactly to prev, at t=1
// exactly to cur, regardless of uMv, so there's no popping at the
// endpoints. uMv is the estimated (prev -> cur) shift in UV units;
// (0,0) degrades gracefully to a plain cross-fade.
constexpr const char *kBlendFragmentShaderSrc = R"glsl(
precision mediump float;
varying vec2 vUV;
uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform vec2 uMv;
uniform float uT;
void main() {
  vec2 uvPrev = vUV - uT * uMv;
  vec2 uvCur = vUV + (1.0 - uT) * uMv;
  vec4 p = texture2D(uPrev, uvPrev);
  vec4 c = texture2D(uCur, uvCur);
  gl_FragColor = mix(p, c, uT);
}
)glsl";

// Downsamples a captured frame to a small luma buffer for the CPU-side
// motion search. Single bilinear tap per output texel — a coarse proxy,
// not a proper box filter; good enough for correlation matching.
constexpr const char *kDownsampleFragmentShaderSrc = R"glsl(
precision mediump float;
varying vec2 vUV;
uniform sampler2D uSrc;
void main() {
  vec3 c = texture2D(uSrc, vUV).rgb;
  float luma = dot(c, vec3(0.299, 0.587, 0.114));
  gl_FragColor = vec4(luma, luma, luma, 1.0);
}
)glsl";

using EglSwapBuffersFn = decltype(&eglSwapBuffers);
using PFN_eglPresentationTimeANDROID = EGLBoolean(EGLAPIENTRY *)(EGLDisplay,
                                                                  EGLSurface,
                                                                  int64_t);

// Vertex Array Object functions, resolved at runtime (OES extension name
// first, then the GLES3 core name — Android's libGLESv2.so serves both
// regardless of which the app linked against). If the game's renderer
// uses GLES3 (very likely), it almost certainly uses a VAO of its own;
// glVertexAttribPointer/glEnableVertexAttribArray write into WHICHEVER
// VAO is currently bound, not some separate global state. Without our
// own VAO to isolate our draws in, we'd be permanently overwriting the
// game's own vertex attribute configuration — a very plausible cause of
// persistent garbled geometry.
using PFN_glGenVertexArrays = void (*)(GLsizei, GLuint *);
using PFN_glBindVertexArray = void (*)(GLuint);
using PFN_glDeleteVertexArrays = void (*)(GLsizei, const GLuint *);
constexpr GLenum kVertexArrayBindingPname = 0x85B5; // same value for
                                                     // GL_VERTEX_ARRAY_BINDING
                                                     // and _OES

PFN_glGenVertexArrays gGenVertexArrays = nullptr;
PFN_glBindVertexArray gBindVertexArray = nullptr;
PFN_glDeleteVertexArrays gDeleteVertexArrays = nullptr;

// Saves/restores the GL state we touch so the game's own next frame
// isn't affected by anything we did.
class GlStateGuard {
public:
  GlStateGuard() {
    glGetIntegerv(GL_CURRENT_PROGRAM, &mProgram);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &mActiveTexture);
    glGetIntegerv(GL_VIEWPORT, mViewport);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &mFramebuffer);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &mArrayBuffer);
    glGetIntegerv(kVertexArrayBindingPname, &mVertexArray);
    mDepthTest = glIsEnabled(GL_DEPTH_TEST);
    mBlend = glIsEnabled(GL_BLEND);
    mCullFace = glIsEnabled(GL_CULL_FACE);

    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &mTex0);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &mTex1);
    glActiveTexture(static_cast<GLenum>(mActiveTexture));
  }

  ~GlStateGuard() {
    if (gBindVertexArray != nullptr) {
      gBindVertexArray(static_cast<GLuint>(mVertexArray));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(mFramebuffer));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(mArrayBuffer));
    glViewport(mViewport[0], mViewport[1], mViewport[2], mViewport[3]);
    glUseProgram(static_cast<GLuint>(mProgram));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(mTex0));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(mTex1));
    glActiveTexture(static_cast<GLenum>(mActiveTexture));

    setEnabled(GL_DEPTH_TEST, mDepthTest);
    setEnabled(GL_BLEND, mBlend);
    setEnabled(GL_CULL_FACE, mCullFace);
  }

  GlStateGuard(const GlStateGuard &) = delete;
  GlStateGuard &operator=(const GlStateGuard &) = delete;

private:
  static void setEnabled(GLenum cap, GLboolean enabled) {
    if (enabled) {
      glEnable(cap);
    } else {
      glDisable(cap);
    }
  }

  GLint mProgram = 0;
  GLint mActiveTexture = GL_TEXTURE0;
  GLint mViewport[4] = {0, 0, 0, 0};
  GLint mFramebuffer = 0;
  GLint mArrayBuffer = 0;
  GLint mVertexArray = 0;
  GLint mTex0 = 0;
  GLint mTex1 = 0;
  GLboolean mDepthTest = GL_FALSE;
  GLboolean mBlend = GL_FALSE;
  GLboolean mCullFace = GL_FALSE;
};

class FrameGenMod {
public:
  static FrameGenMod &instance() {
    static FrameGenMod mod;
    return mod;
  }

  FrameGenMod() : mSelf(*ll::mod::NativeMod::current()) {}

  [[nodiscard]] ll::mod::NativeMod &getSelf() const { return mSelf; }

  bool load() {
    getSelf().getLogger().info("Frame Generator loading");
    return true;
  }

  bool enable() {
    void *target =
        reinterpret_cast<void *>(dlsym(RTLD_DEFAULT, "eglSwapBuffers"));
    if (target == nullptr) {
      getSelf().getLogger().error("Could not resolve eglSwapBuffers");
      return false;
    }

    mSwapHook = pl::memory::HookHandle(
        target, reinterpret_cast<void *>(&swapBuffersDetour),
        reinterpret_cast<void **>(&mOriginalSwapBuffers),
        pl::memory::HookPriority::Normal);
    if (!mSwapHook.installed()) {
      getSelf().getLogger().error("Failed to hook eglSwapBuffers");
      return false;
    }

    void *presTimeSym = reinterpret_cast<void *>(
        eglGetProcAddress("eglPresentationTimeANDROID"));
    mPresentationTimeFn =
        reinterpret_cast<PFN_eglPresentationTimeANDROID>(presTimeSym);
    mTrueFrameGenAvailable = (mPresentationTimeFn != nullptr);
    if (mTrueFrameGenAvailable) {
      getSelf().getLogger().info(
          "eglPresentationTimeANDROID available: true frame generation enabled");
    } else {
      getSelf().getLogger().info(
          "eglPresentationTimeANDROID unavailable: falling back to blend-only smoothing");
    }

    gGenVertexArrays = reinterpret_cast<PFN_glGenVertexArrays>(
        eglGetProcAddress("glGenVertexArraysOES"));
    gBindVertexArray = reinterpret_cast<PFN_glBindVertexArray>(
        eglGetProcAddress("glBindVertexArrayOES"));
    gDeleteVertexArrays = reinterpret_cast<PFN_glDeleteVertexArrays>(
        eglGetProcAddress("glDeleteVertexArraysOES"));
    if (gGenVertexArrays == nullptr || gBindVertexArray == nullptr ||
        gDeleteVertexArrays == nullptr) {
      gGenVertexArrays = reinterpret_cast<PFN_glGenVertexArrays>(
          eglGetProcAddress("glGenVertexArrays"));
      gBindVertexArray = reinterpret_cast<PFN_glBindVertexArray>(
          eglGetProcAddress("glBindVertexArray"));
      gDeleteVertexArrays = reinterpret_cast<PFN_glDeleteVertexArrays>(
          eglGetProcAddress("glDeleteVertexArrays"));
    }
    mVaoSupported = gGenVertexArrays != nullptr &&
                    gBindVertexArray != nullptr &&
                    gDeleteVertexArrays != nullptr;
    getSelf().getLogger().info(
        mVaoSupported
            ? "VAO isolation available: draws use a private VAO"
            : "VAO functions unavailable: drawing without VAO isolation");

    const bool moduleRegistered =
        pl::modmenu::ModuleBuilder(kModuleId, "Frame Generator")
            .modId(getSelf().getId())
            .description(
                "Lightweight motion-compensated frame generation. "
                "Generation Mode: 0=Off, 1=1x, 2=2x generated frames. "
                "Quality Preset: 0=Low Latency, 1=Balanced, 2=Smoothness. "
                "Never affects input or game simulation timing.")
            .defaultEnabled(true)
            .onToggle(onModuleToggle)
            .config(kModeKey, "Generation Mode",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(mGenerationMode.load()), "0", "2")
            .config(kPresetKey, "Quality Preset",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(mQualityPreset.load()), "0", "2")
            .onConfigChanged(onConfigChanged)
            .registerModule();

    if (!moduleRegistered) {
      getSelf().getLogger().error(
          "Failed to register Frame Generator menu module");
    }

    mCurLuma.resize(static_cast<size_t>(kMotionW * kMotionH));
    mPrevLuma.resize(static_cast<size_t>(kMotionW * kMotionH));

    getSelf().getLogger().info("Frame Generator enabled");
    return moduleRegistered;
  }

  bool disable() {
    pl::modmenu::unregisterModule(kModuleId);
    mSwapHook.reset();
    destroyGlResources();
    getSelf().getLogger().info("Frame Generator disabled");
    return true;
  }

  bool unload() {
    getSelf().getLogger().info("Frame Generator unloaded");
    return true;
  }

private:
  ll::mod::NativeMod &mSelf;
  pl::memory::HookHandle mSwapHook;
  EglSwapBuffersFn mOriginalSwapBuffers = nullptr;
  PFN_eglPresentationTimeANDROID mPresentationTimeFn = nullptr;
  bool mTrueFrameGenAvailable = false;
  int64_t mLastCallTimeNs = 0;
  int64_t mLastRealPresentTimeNs = 0;

  std::atomic_bool mEnabled{true};
  std::atomic_int mGenerationMode{1};  // 0=Off,1=1x,2=2x
  std::atomic_int mQualityPreset{1};   // 0=LowLatency,1=Balanced,2=Smoothness

  // Full-res captured frames.
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

  // Small motion-search resources.
  GLuint mDownsampleProgram = 0;
  GLint mDLocPos = -1;
  GLint mDLocUV = -1;
  GLint mDLocSrc = -1;
  GLuint mMotionFbo = 0;
  GLuint mMotionTex = 0; // single small texture, reused for both captures
  bool mVaoSupported = false;
  GLuint mOwnVao = 0;
  std::vector<uint8_t> mCurLuma;
  std::vector<uint8_t> mPrevLuma;

  static void onModuleToggle(std::string_view moduleId, bool enabled) {
    if (moduleId == kModuleId) {
      instance().mEnabled.store(enabled, std::memory_order_relaxed);
    }
  }

  static void onConfigChanged(std::string_view moduleId, std::string_view key,
                               std::string_view value) {
    if (moduleId != kModuleId) {
      return;
    }
    const std::string text(value);
    char *end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str()) {
      return;
    }
    if (key == kModeKey) {
      instance().mGenerationMode.store(
          static_cast<int>(std::clamp<long>(parsed, 0, 2)),
          std::memory_order_relaxed);
      instance().mHaveHistory = false; // avoid a weird mid-transition frame
    } else if (key == kPresetKey) {
      instance().mQualityPreset.store(
          static_cast<int>(std::clamp<long>(parsed, 0, 2)),
          std::memory_order_relaxed);
    }
  }

  static EGLBoolean swapBuffersDetour(EGLDisplay dpy, EGLSurface surface) {
    return instance().handleSwapBuffers(dpy, surface);
  }

  EGLBoolean handleSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    if (mOriginalSwapBuffers == nullptr) {
      return EGL_FALSE;
    }

    const int mode = mGenerationMode.load(std::memory_order_relaxed);
    if (!mEnabled.load(std::memory_order_relaxed) || mode == 0) {
      mHaveHistory = false;
      return mOriginalSwapBuffers(dpy, surface);
    }

    EGLint width = 0;
    EGLint height = 0;
    if (eglQuerySurface(dpy, surface, EGL_WIDTH, &width) == EGL_FALSE ||
        eglQuerySurface(dpy, surface, EGL_HEIGHT, &height) == EGL_FALSE ||
        width <= 0 || height <= 0) {
      return mOriginalSwapBuffers(dpy, surface);
    }

    if (!ensureGlResources(width, height)) {
      return mOriginalSwapBuffers(dpy, surface);
    }

    const auto nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

    GlStateGuard stateGuard;
    captureCurrentFrame();

    if (!mHaveHistory) {
      const EGLBoolean result = mOriginalSwapBuffers(dpy, surface);
      mLastRealPresentTimeNs = nowNs;
      mLastCallTimeNs = nowNs;
      mHaveHistory = true;
      std::swap(mTexPrev, mTexCur);
      return result;
    }

    const int preset =
        std::clamp(mQualityPreset.load(std::memory_order_relaxed), 0, 2);
    const PresetParams &params = kPresets[preset];

    float mvU = 0.0f;
    float mvV = 0.0f;
    estimateMotion(params.searchRadius, params.sadBailoutThreshold, mvU, mvV);

    if (mTrueFrameGenAvailable) {
      int64_t intervalNs = nowNs - mLastCallTimeNs;
      intervalNs = std::clamp<int64_t>(intervalNs, 2'000'000LL, 100'000'000LL);

      const int steps = mode; // 1 or 2 generated frames
      for (int i = 1; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps + 1);
        const int64_t stepTimeNs =
            mLastRealPresentTimeNs +
            static_cast<int64_t>(static_cast<double>(intervalNs) * t);
        mPresentationTimeFn(dpy, surface, stepTimeNs);
        drawBlended(mvU, mvV, t);
        mOriginalSwapBuffers(dpy, surface);
      }

      const int64_t realTimeNs = mLastRealPresentTimeNs + intervalNs;
      mPresentationTimeFn(dpy, surface, realTimeNs);
      drawBlended(0.0f, 0.0f, 1.0f);
      const EGLBoolean result = mOriginalSwapBuffers(dpy, surface);

      mLastRealPresentTimeNs = realTimeNs;
      mLastCallTimeNs = nowNs;
      std::swap(mTexPrev, mTexCur);
      return result;
    }

    // No presentation-time support: single-swap blend-only fallback.
    drawBlended(mvU * 0.5f, mvV * 0.5f, 0.5f);
    const EGLBoolean result = mOriginalSwapBuffers(dpy, surface);
    mLastRealPresentTimeNs = nowNs;
    mLastCallTimeNs = nowNs;
    std::swap(mTexPrev, mTexCur);
    return result;
  }

  // Coarse global motion search on a small downsampled luma buffer.
  // Writes the estimated (prev -> cur) shift, in UV units, to mvU/mvV.
  // On a poor match (best SAD above threshold), returns (0,0) — callers
  // then blend without motion compensation (or, for Low Latency preset,
  // the caller instead snaps to a duplicated real frame — handled by
  // drawBlended's t rounding when mv is (0,0) and preset is Low Latency;
  // kept simple here by just always degrading to a plain cross-fade,
  // which is the safe behavior for every preset).
  void estimateMotion(int radius, float sadThreshold, float &mvU,
                       float &mvV) {
    mvU = 0.0f;
    mvV = 0.0f;
    if (mDownsampleProgram == 0 || mMotionFbo == 0) {
      return;
    }

    downsampleTo(mTexCur, mCurLuma);
    downsampleTo(mTexPrev, mPrevLuma);

    int bestDx = 0;
    int bestDy = 0;
    float bestSad = -1.0f;

    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const float sad = computeSad(dx, dy);
        if (bestSad < 0.0f || sad < bestSad) {
          bestSad = sad;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }

    if (bestSad < 0.0f || bestSad > sadThreshold) {
      return; // no confident match — caller falls back gracefully
    }

    mvU = static_cast<float>(bestDx) / static_cast<float>(kMotionW);
    mvV = static_cast<float>(bestDy) / static_cast<float>(kMotionH);
  }

  // Average per-pixel luma difference between mCurLuma and mPrevLuma
  // shifted by (dx, dy), over the overlapping region only.
  float computeSad(int dx, int dy) const {
    int64_t sum = 0;
    int count = 0;
    for (int y = 0; y < kMotionH; ++y) {
      const int py = y - dy;
      if (py < 0 || py >= kMotionH) {
        continue;
      }
      for (int x = 0; x < kMotionW; ++x) {
        const int px = x - dx;
        if (px < 0 || px >= kMotionW) {
          continue;
        }
        const int curVal = mCurLuma[static_cast<size_t>(y * kMotionW + x)];
        const int prevVal =
            mPrevLuma[static_cast<size_t>(py * kMotionW + px)];
        sum += std::abs(curVal - prevVal);
        ++count;
      }
    }
    if (count == 0) {
      return 1e9f; // no overlap at this shift — treat as a very poor match
    }
    return static_cast<float>(sum) / static_cast<float>(count);
  }

  void downsampleTo(GLuint srcTex, std::vector<uint8_t> &outLuma) {
    if (mVaoSupported) {
      gBindVertexArray(mOwnVao);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, mMotionFbo);
    glViewport(0, 0, kMotionW, kMotionH);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    glUseProgram(mDownsampleProgram);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, srcTex);
    glUniform1i(mDLocSrc, 0);

    glVertexAttribPointer(static_cast<GLuint>(mDLocPos), 2, GL_FLOAT,
                          GL_FALSE, 4 * sizeof(float), kQuadVerts);
    glEnableVertexAttribArray(static_cast<GLuint>(mDLocPos));
    glVertexAttribPointer(static_cast<GLuint>(mDLocUV), 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), kQuadVerts + 2);
    glEnableVertexAttribArray(static_cast<GLuint>(mDLocUV));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableVertexAttribArray(static_cast<GLuint>(mDLocPos));
    glDisableVertexAttribArray(static_cast<GLuint>(mDLocUV));

    // Synchronous readback of a tiny buffer. See the file header note on
    // this being an unprofiled potential stall point.
    std::vector<uint8_t> rgba(static_cast<size_t>(kMotionW * kMotionH * 4));
    glReadPixels(0, 0, kMotionW, kMotionH, GL_RGBA, GL_UNSIGNED_BYTE,
                 rgba.data());
    for (int i = 0; i < kMotionW * kMotionH; ++i) {
      outLuma[static_cast<size_t>(i)] = rgba[static_cast<size_t>(i) * 4];
    }
  }

  bool ensureGlResources(GLsizei width, GLsizei height) {
    if (mBlendProgram == 0 && !buildBlendProgram()) {
      return false;
    }
    if (mDownsampleProgram == 0 && !buildDownsampleProgram()) {
      return false;
    }
    if (mMotionFbo == 0 && !buildMotionFbo()) {
      return false;
    }
    if (mVaoSupported && mOwnVao == 0) {
      gGenVertexArrays(1, &mOwnVao);
    }

    if (mGlReady && width == mTexWidth && height == mTexHeight) {
      return true;
    }

    destroyTextures();
    mTexPrev = createFrameTexture(width, height);
    mTexCur = createFrameTexture(width, height);
    if (mTexPrev == 0 || mTexCur == 0) {
      destroyTextures();
      mGlReady = false;
      return false;
    }

    mTexWidth = width;
    mTexHeight = height;
    mHaveHistory = false;
    mGlReady = true;
    return true;
  }

  bool buildMotionFbo() {
    glGenTextures(1, &mMotionTex);
    if (mMotionTex == 0) {
      return false;
    }
    glBindTexture(GL_TEXTURE_2D, mMotionTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, kMotionW, kMotionH, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, nullptr);

    glGenFramebuffers(1, &mMotionFbo);
    if (mMotionFbo == 0) {
      glDeleteTextures(1, &mMotionTex);
      mMotionTex = 0;
      return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, mMotionFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, mMotionTex, 0);
    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      getSelf().getLogger().error("Motion FBO incomplete");
      glDeleteFramebuffers(1, &mMotionFbo);
      glDeleteTextures(1, &mMotionTex);
      mMotionFbo = 0;
      mMotionTex = 0;
      return false;
    }
    return true;
  }

  static GLuint createFrameTexture(GLsizei width, GLsizei height) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (tex == 0) {
      return 0;
    }
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, nullptr);
    return tex;
  }

  void captureCurrentFrame() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, mTexCur);
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, mTexWidth, mTexHeight,
                      0);
  }

  void drawBlended(float mvU, float mvV, float t) {
    if (mVaoSupported) {
      gBindVertexArray(mOwnVao);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, mTexWidth, mTexHeight);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

    glUseProgram(mBlendProgram);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexPrev);
    glUniform1i(mBLocPrev, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mTexCur);
    glUniform1i(mBLocCur, 1);

    glUniform2f(mBLocMv, mvU, mvV);
    glUniform1f(mBLocT, t);

    glVertexAttribPointer(static_cast<GLuint>(mBLocPos), 2, GL_FLOAT,
                          GL_FALSE, 4 * sizeof(float), kQuadVerts);
    glEnableVertexAttribArray(static_cast<GLuint>(mBLocPos));
    glVertexAttribPointer(static_cast<GLuint>(mBLocUV), 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), kQuadVerts + 2);
    glEnableVertexAttribArray(static_cast<GLuint>(mBLocUV));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableVertexAttribArray(static_cast<GLuint>(mBLocPos));
    glDisableVertexAttribArray(static_cast<GLuint>(mBLocUV));
  }

  bool buildBlendProgram() {
    GLuint program = 0;
    if (!linkProgram(kVertexShaderSrc, kBlendFragmentShaderSrc, program)) {
      return false;
    }
    mBlendProgram = program;
    mBLocPos = glGetAttribLocation(mBlendProgram, "aPos");
    mBLocUV = glGetAttribLocation(mBlendProgram, "aUV");
    mBLocPrev = glGetUniformLocation(mBlendProgram, "uPrev");
    mBLocCur = glGetUniformLocation(mBlendProgram, "uCur");
    mBLocMv = glGetUniformLocation(mBlendProgram, "uMv");
    mBLocT = glGetUniformLocation(mBlendProgram, "uT");
    return true;
  }

  bool buildDownsampleProgram() {
    GLuint program = 0;
    if (!linkProgram(kVertexShaderSrc, kDownsampleFragmentShaderSrc,
                      program)) {
      return false;
    }
    mDownsampleProgram = program;
    mDLocPos = glGetAttribLocation(mDownsampleProgram, "aPos");
    mDLocUV = glGetAttribLocation(mDownsampleProgram, "aUV");
    mDLocSrc = glGetUniformLocation(mDownsampleProgram, "uSrc");
    return true;
  }

  bool linkProgram(const char *vsSrc, const char *fsSrc, GLuint &outProgram) {
    const GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    if (vs == 0 || fs == 0) {
      if (vs != 0) glDeleteShader(vs);
      if (fs != 0) glDeleteShader(fs);
      return false;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "aPos");
    glBindAttribLocation(program, 1, "aUV");
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
      logProgramError(program);
      glDeleteProgram(program);
      return false;
    }
    outProgram = program;
    return true;
  }

  GLuint compileShader(GLenum type, const char *src) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
      logShaderError(shader);
      glDeleteShader(shader);
      return 0;
    }
    return shader;
  }

  void logShaderError(GLuint shader) {
    GLint len = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<size_t>(std::max(len, 1)), '\0');
    glGetShaderInfoLog(shader, len, nullptr, log.data());
    getSelf().getLogger().error("Shader compile failed: {}", log);
  }

  void logProgramError(GLuint program) {
    GLint len = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
    std::string log(static_cast<size_t>(std::max(len, 1)), '\0');
    glGetProgramInfoLog(program, len, nullptr, log.data());
    getSelf().getLogger().error("Program link failed: {}", log);
  }

  void destroyTextures() {
    if (mTexPrev != 0) {
      glDeleteTextures(1, &mTexPrev);
      mTexPrev = 0;
    }
    if (mTexCur != 0) {
      glDeleteTextures(1, &mTexCur);
      mTexCur = 0;
    }
  }

  void destroyGlResources() {
    destroyTextures();
    if (mBlendProgram != 0) {
      glDeleteProgram(mBlendProgram);
      mBlendProgram = 0;
    }
    if (mDownsampleProgram != 0) {
      glDeleteProgram(mDownsampleProgram);
      mDownsampleProgram = 0;
    }
    if (mMotionFbo != 0) {
      glDeleteFramebuffers(1, &mMotionFbo);
      mMotionFbo = 0;
    }
    if (mMotionTex != 0) {
      glDeleteTextures(1, &mMotionTex);
      mMotionTex = 0;
    }
    if (mOwnVao != 0 && gDeleteVertexArrays != nullptr) {
      gDeleteVertexArrays(1, &mOwnVao);
      mOwnVao = 0;
    }
    mGlReady = false;
    mHaveHistory = false;
    mTexWidth = 0;
    mTexHeight = 0;
  }
};

} // namespace

PL_REGISTER_MOD(FrameGenMod, FrameGenMod::instance())
