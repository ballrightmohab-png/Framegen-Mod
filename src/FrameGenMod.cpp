// FrameGenMod.cpp
//
// An enhanced LeviLaunchroid native mod implementing true frame generation,
// V-Sync uncap, and complete screen-flicker prevention.
//
// Key Fixes in this version:
// 1. Fixed Screen Flickering: Fragment shader explicitly forces Alpha to 1.0.
//    In Minecraft Bedrock, the backbuffer alpha channel contains arbitrary depth/masking
//    data. Passing alpha < 1.0 to Android's SurfaceFlinger caused rapid transparent
//    compositing flicker.
// 2. High-Performance Captures: Replaced glCopyTexImage2D with glCopyTexSubImage2D
//    to avoid GPU memory allocation churn on every frame.
// 3. UI State & VAO Guard: Added Vertex Array Object (VAO) saving/restoring and
//    VBO unbinding to ensure UI menus and inventory slots render without colored quad leakage.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include <dlfcn.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <pl/Mod.hpp>
#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>

namespace {

constexpr const char *kModuleId = "framegen.core";
constexpr const char *kBlendKey = "blendStrength";
constexpr const char *kVSyncKey = "disableVSync";
constexpr const char *kMultiplierKey = "frameMultiplier";

// Fullscreen quad: (x, y, u, v) per vertex, drawn as a triangle strip.
constexpr float kQuadVerts[16] = {
    -1.0f, -1.0f, 0.0f, 0.0f, // bottom-left
     1.0f, -1.0f, 1.0f, 0.0f, // bottom-right
    -1.0f,  1.0f, 0.0f, 1.0f, // top-left
     1.0f,  1.0f, 1.0f, 1.0f, // top-right
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

// Note: Alpha is strictly forced to 1.0 to prevent Android SurfaceFlinger compositor flickering!
constexpr const char *kFragmentShaderSrc = R"glsl(
precision mediump float;
varying vec2 vUV;
uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform float uMix;
void main() {
  vec3 p = texture2D(uPrev, vUV).rgb;
  vec3 c = texture2D(uCur, vUV).rgb;
  gl_FragColor = vec4(mix(p, c, uMix), 1.0);
}
)glsl";

using EglSwapBuffersFn = decltype(&eglSwapBuffers);
using EglSwapIntervalFn = decltype(&eglSwapInterval);
using PfnGlBindVertexArray = void (GL_APIENTRY *)(GLuint array);

inline PfnGlBindVertexArray gGlBindVertexArray = nullptr;

inline void initGlExtensions() {
  if (!gGlBindVertexArray) {
    gGlBindVertexArray = reinterpret_cast<PfnGlBindVertexArray>(
        eglGetProcAddress("glBindVertexArray"));
    if (!gGlBindVertexArray) {
      gGlBindVertexArray = reinterpret_cast<PfnGlBindVertexArray>(
          eglGetProcAddress("glBindVertexArrayOES"));
    }
  }
}

// Comprehensive state guard capturing graphics state to avoid leaking into Minecraft Bedrock's UI
class GlStateGuard {
public:
  GlStateGuard() {
    initGlExtensions();

    glGetIntegerv(GL_CURRENT_PROGRAM, &mProgram);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &mActiveTexture);
    glGetIntegerv(GL_VIEWPORT, mViewport);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &mFramebuffer);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &mArrayBuffer);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &mElementArrayBuffer);

    // Save Vertex Array Object (VAO) binding
    glGetIntegerv(0x85B5 /* GL_VERTEX_ARRAY_BINDING */, &mVao);

    mDepthTest = glIsEnabled(GL_DEPTH_TEST);
    mBlend = glIsEnabled(GL_BLEND);
    mCullFace = glIsEnabled(GL_CULL_FACE);
    mScissorTest = glIsEnabled(GL_SCISSOR_TEST);
    mStencilTest = glIsEnabled(GL_STENCIL_TEST);

    glGetBooleanv(GL_COLOR_WRITEMASK, mColorMask);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &mDepthMask);

    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &mTex0);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &mTex1);
    glActiveTexture(static_cast<GLenum>(mActiveTexture));

    // Unbind VAO and VBOs to prevent corrupting game UI vertex attributes
    if (gGlBindVertexArray && mVao != 0) {
      gGlBindVertexArray(0);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  ~GlStateGuard() {
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(mFramebuffer));
    glViewport(mViewport[0], mViewport[1], mViewport[2], mViewport[3]);
    glUseProgram(static_cast<GLuint>(mProgram));

    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(mArrayBuffer));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLuint>(mElementArrayBuffer));

    if (gGlBindVertexArray && mVao != 0) {
      gGlBindVertexArray(static_cast<GLuint>(mVao));
    }

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(mTex0));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(mTex1));
    glActiveTexture(static_cast<GLenum>(mActiveTexture));

    setEnabled(GL_DEPTH_TEST, mDepthTest);
    setEnabled(GL_BLEND, mBlend);
    setEnabled(GL_CULL_FACE, mCullFace);
    setEnabled(GL_SCISSOR_TEST, mScissorTest);
    setEnabled(GL_STENCIL_TEST, mStencilTest);

    glColorMask(mColorMask[0], mColorMask[1], mColorMask[2], mColorMask[3]);
    glDepthMask(mDepthMask);
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
  GLint mElementArrayBuffer = 0;
  GLint mVao = 0;
  GLint mTex0 = 0;
  GLint mTex1 = 0;
  GLboolean mDepthTest = GL_FALSE;
  GLboolean mBlend = GL_FALSE;
  GLboolean mCullFace = GL_FALSE;
  GLboolean mScissorTest = GL_FALSE;
  GLboolean mStencilTest = GL_FALSE;
  GLboolean mColorMask[4] = {GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE};
  GLboolean mDepthMask = GL_TRUE;
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
    getSelf().getLogger().info("Enhanced Frame Generator loading");
    return true;
  }

  bool enable() {
    void *swapTarget = reinterpret_cast<void *>(dlsym(RTLD_DEFAULT, "eglSwapBuffers"));
    void *intervalTarget = reinterpret_cast<void *>(dlsym(RTLD_DEFAULT, "eglSwapInterval"));

    if (swapTarget == nullptr) {
      getSelf().getLogger().error("Could not resolve eglSwapBuffers");
      return false;
    }

    mSwapHook = pl::memory::HookHandle(
        swapTarget, reinterpret_cast<void *>(&swapBuffersDetour),
        reinterpret_cast<void **>(&mOriginalSwapBuffers),
        pl::memory::HookPriority::Normal);

    if (!mSwapHook.installed()) {
      getSelf().getLogger().error("Failed to hook eglSwapBuffers");
      return false;
    }

    if (intervalTarget != nullptr) {
      mIntervalHook = pl::memory::HookHandle(
          intervalTarget, reinterpret_cast<void *>(&swapIntervalDetour),
          reinterpret_cast<void **>(&mOriginalSwapInterval),
          pl::memory::HookPriority::Normal);
    } else {
      getSelf().getLogger().warn("Could not resolve eglSwapInterval; V-Sync uncap might be restricted.");
    }

    const bool moduleRegistered =
        pl::modmenu::ModuleBuilder(kModuleId, "Frame Generator & V-Sync")
            .modId(getSelf().getId())
            .description(
                "Enhances game performance with frame multiplication (2x/3x FPS) "
                "and V-Sync disabling for uncapped framerates.")
            .defaultEnabled(true)
            .onToggle(onModuleToggle)
            .config(kVSyncKey, "Disable V-Sync (Uncap FPS)",
                    pl::modmenu::ConfigType::Toggle, "true")
            .config(kMultiplierKey, "FPS Multiplier (1x=Off, 2x, 3x)",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(mFrameMultiplier.load()), "1", "3")
            .config(kBlendKey, "Interpolation Blend Strength",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(mBlendPercent.load()), "0", "100")
            .onConfigChanged(onConfigChanged)
            .registerModule();

    if (!moduleRegistered) {
      getSelf().getLogger().error("Failed to register Frame Generator menu module");
    }

    getSelf().getLogger().info("Enhanced Frame Generator enabled");
    return moduleRegistered;
  }

  bool disable() {
    pl::modmenu::unregisterModule(kModuleId);
    mSwapHook.reset();
    mIntervalHook.reset();
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
  pl::memory::HookHandle mIntervalHook;
  EglSwapBuffersFn mOriginalSwapBuffers = nullptr;
  EglSwapIntervalFn mOriginalSwapInterval = nullptr;

  std::atomic_bool mEnabled{true};
  std::atomic_bool mDisableVSync{true};
  std::atomic_int mFrameMultiplier{2};
  std::atomic_int mBlendPercent{50};

  bool mGlReady = false;
  bool mHaveHistory = false;
  GLsizei mTexWidth = 0;
  GLsizei mTexHeight = 0;
  GLuint mTexPrev = 0;
  GLuint mTexCur = 0;
  GLuint mProgram = 0;
  GLint mLocPos = -1;
  GLint mLocUV = -1;
  GLint mLocPrev = -1;
  GLint mLocCur = -1;
  GLint mLocMix = -1;

  static void onModuleToggle(std::string_view moduleId, bool enabled) {
    if (moduleId == kModuleId) {
      instance().mEnabled.store(enabled, std::memory_order_relaxed);
    }
  }

  static void onConfigChanged(std::string_view moduleId, std::string_view key,
                               std::string_view value) {
    if (moduleId != kModuleId) return;

    if (key == kVSyncKey) {
      instance().mDisableVSync.store(value == "true" || value == "1", std::memory_order_relaxed);
    } else if (key == kMultiplierKey) {
      char *end = nullptr;
      const long parsed = std::strtol(std::string(value).c_str(), &end, 10);
      if (end != value.data()) {
        instance().mFrameMultiplier.store(
            static_cast<int>(std::clamp<long>(parsed, 1, 3)),
            std::memory_order_relaxed);
      }
    } else if (key == kBlendKey) {
      char *end = nullptr;
      const long parsed = std::strtol(std::string(value).c_str(), &end, 10);
      if (end != value.data()) {
        instance().mBlendPercent.store(
            static_cast<int>(std::clamp<long>(parsed, 0, 100)),
            std::memory_order_relaxed);
      }
    }
  }

  static EGLBoolean swapIntervalDetour(EGLDisplay dpy, EGLint interval) {
    if (instance().mEnabled.load(std::memory_order_relaxed) &&
        instance().mDisableVSync.load(std::memory_order_relaxed)) {
      if (instance().mOriginalSwapInterval != nullptr) {
        return instance().mOriginalSwapInterval(dpy, 0); // Force interval = 0 (Uncapped)
      }
      return EGL_TRUE;
    }
    if (instance().mOriginalSwapInterval != nullptr) {
      return instance().mOriginalSwapInterval(dpy, interval);
    }
    return EGL_FALSE;
  }

  static EGLBoolean swapBuffersDetour(EGLDisplay dpy, EGLSurface surface) {
    return instance().handleSwapBuffers(dpy, surface);
  }

  EGLBoolean handleSwapBuffers(EGLDisplay dpy, EGLSurface surface) {
    if (mOriginalSwapBuffers == nullptr) {
      return EGL_FALSE;
    }

    if (!mEnabled.load(std::memory_order_relaxed)) {
      mHaveHistory = false;
      return mOriginalSwapBuffers(dpy, surface);
    }

    if (mDisableVSync.load(std::memory_order_relaxed) && mOriginalSwapInterval != nullptr) {
      mOriginalSwapInterval(dpy, 0);
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

    GlStateGuard stateGuard;
    captureCurrentFrame();

    const int multiplier = mFrameMultiplier.load(std::memory_order_relaxed);
    const float strength = static_cast<float>(std::clamp(mBlendPercent.load(), 0, 100)) / 100.0f;

    // Smooth Frame Generation Pass:
    // Insert intermediate frame(s) before swapping
    if (mHaveHistory && multiplier > 1) {
      for (int step = 1; step < multiplier; ++step) {
        const float t = static_cast<float>(step) / static_cast<float>(multiplier);
        const float mix = t * strength + (1.0f - strength);

        drawBlended(mix);
        mOriginalSwapBuffers(dpy, surface); // Swap intermediate frame
      }
    }

    // Present current real frame (mix = 1.0 ensures clean crisp image on real frame)
    if (mHaveHistory) {
      drawBlended(1.0f);
    }

    const EGLBoolean result = mOriginalSwapBuffers(dpy, surface);

    mHaveHistory = true;
    std::swap(mTexPrev, mTexCur);
    return result;
  }

  bool ensureGlResources(GLsizei width, GLsizei height) {
    if (mProgram == 0 && !buildProgram()) {
      return false;
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
    // glCopyTexSubImage2D reuses allocated GPU texture memory, avoiding allocation latency
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, mTexWidth, mTexHeight);
  }

  void drawBlended(float mix) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, mTexWidth, mTexHeight);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);

    glUseProgram(mProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexPrev);
    glUniform1i(mLocPrev, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, mTexCur);
    glUniform1i(mLocCur, 1);

    glUniform1f(mLocMix, mix);

    glVertexAttribPointer(static_cast<GLuint>(mLocPos), 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), kQuadVerts);
    glEnableVertexAttribArray(static_cast<GLuint>(mLocPos));

    glVertexAttribPointer(static_cast<GLuint>(mLocUV), 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), kQuadVerts + 2);
    glEnableVertexAttribArray(static_cast<GLuint>(mLocUV));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glDisableVertexAttribArray(static_cast<GLuint>(mLocPos));
    glDisableVertexAttribArray(static_cast<GLuint>(mLocUV));
  }

  bool buildProgram() {
    const GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShaderSrc);
    const GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShaderSrc);
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

    mProgram = program;
    mLocPos = glGetAttribLocation(mProgram, "aPos");
    mLocUV = glGetAttribLocation(mProgram, "aUV");
    mLocPrev = glGetUniformLocation(mProgram, "uPrev");
    mLocCur = glGetUniformLocation(mProgram, "uCur");
    mLocMix = glGetUniformLocation(mProgram, "uMix");
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
    if (mProgram != 0) {
      glDeleteProgram(mProgram);
      mProgram = 0;
    }
    mGlReady = false;
    mHaveHistory = false;
    mTexWidth = 0;
    mTexHeight = 0;
  }
};

} // namespace

PL_REGISTER_MOD(FrameGenMod, FrameGenMod::instance())
