#ifndef SL_ENGINE_H
#define SL_ENGINE_H

#include <stddef.h>
#include <stdint.h>

int sl_open(const char *dir, char *err, size_t cap);
const char *sl_system_info(void);
void *sl_load_model(const char *path, char *err, size_t cap);
void sl_free_model(void *m);
int sl_tokenize(void *model, const char *text, int len, int32_t *out, int cap);
void *sl_new_context(void *model, int n, char *err, size_t cap);
void sl_free_context(void *c);
int sl_eval(void *c, int32_t *tokens, int n, const int32_t *letters, int k, float *out);

#endif
