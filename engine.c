// The llama.cpp engine, loaded at run time from the directory Godwit
// verified (models.load's assets.engine): nothing links against libllama, so
// the plugin library itself loads on any machine and the engine version is the
// one the registry pins. Function types come from the vendored llama.h of the
// same build (third_party/llama/VERSION).
#include <stdio.h>
#include <string.h>
#include "engine.h"
#include "third_party/llama/llama.h"

#ifdef _WIN32
#include <windows.h>
#define LIBNAME "llama.dll"
#else
#include <dlfcn.h>
#ifdef __APPLE__
#define LIBNAME "libllama.0.dylib"
#else
#define LIBNAME "libllama.so.0"
#endif
#endif

static struct {
    void (*backend_init)(void);
    void (*log_set)(ggml_log_callback, void *);
    struct llama_model_params (*model_default_params)(void);
    struct llama_model *(*model_load_from_file)(const char *, struct llama_model_params);
    void (*model_free)(struct llama_model *);
    struct llama_context_params (*context_default_params)(void);
    struct llama_context *(*init_from_model)(struct llama_model *, struct llama_context_params);
    void (*free)(struct llama_context *);
    const struct llama_vocab *(*model_get_vocab)(const struct llama_model *);
    int32_t (*tokenize)(const struct llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool);
    struct llama_batch (*batch_get_one)(llama_token *, int32_t);
    int32_t (*decode)(struct llama_context *, struct llama_batch);
    float *(*get_logits_ith)(struct llama_context *, int32_t);
    llama_memory_t (*get_memory)(const struct llama_context *);
    void (*memory_clear)(llama_memory_t, bool);
    const char *(*print_system_info)(void);
} api;

static int opened;

// Only warnings and errors: plugin-host's stderr is Godwit's log.
static void quiet(enum ggml_log_level level, const char *text, void *data) {
    (void)data;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

static void set_err(char *err, size_t cap, const char *what) {
    if (err && cap) snprintf(err, cap, "%s", what);
}

int sl_open(const char *dir, char *err, size_t cap) {
    if (opened) return 0;
    char path[4096];
    if (snprintf(path, sizeof path, "%s/%s", dir, LIBNAME) >= (int)sizeof path) {
        set_err(err, cap, "the engine directory path is too long");
        return -1;
    }
#ifdef _WIN32
    wchar_t wdir[4096];
    MultiByteToWideChar(CP_UTF8, 0, dir, -1, wdir, 4096);
    AddDllDirectory(wdir); // its ggml*.dll and libomp.dll beside it
    wchar_t wpath[4096];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 4096);
    HMODULE h = LoadLibraryExW(wpath, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) {
        snprintf(err, cap, "cannot load %s (error %lu)", path, GetLastError());
        return -1;
    }
#define SYM(name) GetProcAddress(h, name)
    // GetProcAddress does not search dependencies: the backend loader is ggml.dll's.
    void (*load_all)(const char *) = (void (*)(const char *))GetProcAddress(GetModuleHandleA("ggml.dll"), "ggml_backend_load_all_from_path");
#else
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        snprintf(err, cap, "cannot load %s: %s", path, dlerror());
        return -1;
    }
#define SYM(name) dlsym(h, name)
    void (*load_all)(const char *) = (void (*)(const char *))dlsym(h, "ggml_backend_load_all_from_path");
#endif
#define BIND(field, name)                                   \
    do {                                                    \
        *(void **)(&api.field) = (void *)SYM(name);         \
        if (!api.field) {                                   \
            snprintf(err, cap, "the engine has no %s", name); \
            return -1;                                      \
        }                                                   \
    } while (0)
    BIND(backend_init, "llama_backend_init");
    BIND(log_set, "llama_log_set");
    BIND(model_default_params, "llama_model_default_params");
    BIND(model_load_from_file, "llama_model_load_from_file");
    BIND(model_free, "llama_model_free");
    BIND(context_default_params, "llama_context_default_params");
    BIND(init_from_model, "llama_init_from_model");
    BIND(free, "llama_free");
    BIND(model_get_vocab, "llama_model_get_vocab");
    BIND(tokenize, "llama_tokenize");
    BIND(batch_get_one, "llama_batch_get_one");
    BIND(decode, "llama_decode");
    BIND(get_logits_ith, "llama_get_logits_ith");
    BIND(get_memory, "llama_get_memory");
    BIND(memory_clear, "llama_memory_clear");
    BIND(print_system_info, "llama_print_system_info");
    api.log_set(quiet, NULL);
    // Linux and Windows packages build their backends as modules (one CPU
    // backend per instruction set); macOS links them in, and this is a no-op.
    if (load_all) load_all(dir);
    api.backend_init();
    opened = 1;
    return 0;
}

const char *sl_system_info(void) { return opened ? api.print_system_info() : ""; }

void *sl_load_model(const char *path, char *err, size_t cap) {
    struct llama_model_params p = api.model_default_params();
    p.n_gpu_layers = 99;
    struct llama_model *m = api.model_load_from_file(path, p);
    if (!m) set_err(err, cap, "the engine cannot load the model");
    return m;
}

void sl_free_model(void *m) { api.model_free(m); }

int sl_tokenize(void *model, const char *text, int len, int32_t *out, int cap) {
    // add_special false, parse_special true: the prompt carries its own
    // <|im_start|> tokens, as the reference tokenizer reads them.
    return api.tokenize(api.model_get_vocab(model), text, len, out, cap, false, true);
}

void *sl_new_context(void *model, int n, char *err, size_t cap) {
    struct llama_context_params p = api.context_default_params();
    // The whole prompt in one micro-batch: the model's full-attention layers
    // read it bidirectionally (the GGUF says attention.causal = false).
    p.n_ctx = p.n_batch = p.n_ubatch = n;
    p.n_seq_max = 1;
    struct llama_context *c = api.init_from_model(model, p);
    if (!c) set_err(err, cap, "the engine cannot make a context (not enough memory?)");
    return c;
}

void sl_free_context(void *c) { api.free(c); }

int sl_eval(void *c, int32_t *tokens, int n, const int32_t *letters, int k, float *out) {
    api.memory_clear(api.get_memory(c), true); // every prompt alone, nothing cached
    if (api.decode(c, api.batch_get_one(tokens, n)) != 0) return -1;
    float *logits = api.get_logits_ith(c, -1);
    if (!logits) return -1;
    for (int i = 0; i < k; i++) out[i] = logits[letters[i]];
    return 0;
}
