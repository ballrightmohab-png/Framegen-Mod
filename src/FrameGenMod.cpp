// FrameGenMod.cpp
//
// A LeviLaunchroid native mod that implements temporal frame blending:
// it hooks eglSwapBuffers and, on each real swap, blends the just-finished
// frame with the previous one before presenting it — one swap in, one
// swap out, exactly matching the game's normal swap cadence.
//
// NOTE ON SCOPE: an earlier version of this mod tried to *double* the
// number of presented frames by calling eglSwapBuffers twice per real
// frame (present a blended frame, then re-present the real one). That
// fought with Android's buffer queue / VSYNC timing and caused visible
// flicker. This version deliberately does not do that: it never changes
// how many times eglSwapBuffers is called, so it can't perturb frame
// pacing the same way. The cost is that this is no longer true frame
// generation (it does not increase displayed FPS) — it's a temporal
// smoothing filter: transitions between frames look softer, at the cost
// of a small amount of ghosting on fast motion.
//
// This is intentionally a simple, dependency-light technique: it only
// needs the frames that are already on screen, not engine internals, so
// it hooks a stable system entry point (eglSwapBuffers) rather than a
// Minecraft-internal function that would need per-version signatures.
//
// Build/package conventions follow examples/mod-menu-api in the
// LeviLaunchroid repo.

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

// Fullscreen quad: (x, y, u, v) per vertex, drawn as a triangle strip.
// NOTE: texture-coordinate orientation only needs to be *consistent*
// between the prev/cur captures and both draw passes below; it does not
// need to match "correct" screen orientation, since we never look at the
// texture except through this same quad.
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

constexpr const char *kFragmentShaderSrc = R"glsl(
precision mediump float;
varying vec2 vUV;
uniform sampler2D uPrev;
uniform sampler2D uCur;
uniform float uMix;
void main() {
  vec4 p = texture2D(uPrev, vUV);
  vec4 c = texture2D(uCur, vUV);
  gl_FragColor = mix(p, c, uMix);
}
)glsl";

using EglSwapBuffersFn = decltype(&eglSwapBuffers);

// Saves a small set of GL state we touch and restores it on scope exit,
// so we don't leave the pipeline in a surprising state for the game's
// own next frame.
class GlStateGuard {
public:
  GlStateGuard() {
    glGetIntegerv(GL_CURRENT_PROGRAM, &mProgram);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &mActiveTexture);
    glGetIntegerv(GL_VIEWPORT, mViewport);
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &mFramebuffer);
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
    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(mFramebuffer));
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
    void *target = reinterpret_cast<void *>(dlsym(RTLD_DEFAULT, "eglSwapBuffers"));
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

    const bool moduleRegistered =
        pl::modmenu::ModuleBuilder(kModuleId, "Frame Generator")
            .modId(getSelf().getId())
            .description(
                "Smooths transitions by blending each frame with the "
                "previous one before it's shown. Does not increase FPS; "
                "may cause slight ghosting on fast motion. Set strength "
                "to 0 to disable the effect entirely.")
            .defaultEnabled(true)
            .onToggle(onModuleToggle)
            .config(kBlendKey, "Interpolation Strength",
                    pl::modmenu::ConfigType::SliderInt,
                    std::to_string(mBlendPercent.load()), "0", "100")
            .onConfigChanged(onConfigChanged)
            .registerModule();

    if (!moduleRegistered) {
      getSelf().getLogger().error("Failed to register Frame Generator menu module");
    }

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

  std::atomic_bool mEnabled{true};
  std::atomic_int mBlendPercent{50};

  // GL state; only ever touched from the render thread inside the hook.
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
    if (moduleId != kModuleId || key != kBlendKey) {
      return;
    }
    const std::string text(value);
    char *end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end != text.c_str()) {
      instance().mBlendPercent.store(
          static_cast<int>(std::clamp<long>(parsed, 0, 100)),
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

    if (!mEnabled.load(std::memory_order_relaxed)) {
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

    GlStateGuard stateGuard;
    captureCurrentFrame();

    if (mHaveHistory) {
      const float mix =
          static_cast<float>(std::clamp(mBlendPercent.load(), 0, 100)) /
          100.0f;
      // mix == 0 means "all previous frame"; we want mix == 0 to mean
      // "no effect", so blend toward the current frame using
      // (1 - strength) as how much of the previous frame survives.
      drawBlended(1.0f - mix * 0.5f);
    }

    // Exactly one real swap per call, always — this never changes the
    // game's swap cadence, which is what caused the earlier flicker.
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
    mHaveHistory = false; // freshly (re)allocated textures hold no history
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
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, mTexWidth, mTexHeight,
                      0);
  }

  void drawBlended(float mix) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, mTexWidth, mTexHeight);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);

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
