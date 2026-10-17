package main

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo !windows LDFLAGS: -ldl
#include <stdlib.h>
#include "engine.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// contextTiers are the context sizes a prompt is read in: the smallest that
// holds it. llama.cpp reserves a compute buffer of about 1 GB per 1K, which
// Windows commits in full, so a short prompt must not need a 4K context.
var contextTiers = []int{1024, 2048, 4096}

// maxPromptTokens leaves room under the largest tier.
const maxPromptTokens = 4096 - 16

const errCap = 1024

func cErr(buf []C.char) error { return errors.New(C.GoString(&buf[0])) }

// openEngine loads libllama from dir, once per process.
func openEngine(dir string) error {
	cdir := C.CString(dir)
	defer C.free(unsafe.Pointer(cdir))
	buf := make([]C.char, errCap)
	if C.sl_open(cdir, &buf[0], errCap) != 0 {
		return cErr(buf)
	}
	return nil
}

// model is a loaded GGUF and the context it reads prompts in; not safe for
// concurrent use (the plugin's lock serializes it).
type model struct {
	m       unsafe.Pointer
	ctx     unsafe.Pointer
	ctxSize int
}

func loadModel(path string) (*model, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	buf := make([]C.char, errCap)
	m := C.sl_load_model(cpath, &buf[0], errCap)
	if m == nil {
		return nil, cErr(buf)
	}
	return &model{m: m}, nil
}

func (m *model) free() {
	if m.ctx != nil {
		C.sl_free_context(m.ctx)
	}
	C.sl_free_model(m.m)
}

// tokenize reads text with the GGUF's own vocabulary.
func (m *model) tokenize(text string) ([]int32, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	out := make([]int32, len(text)+16)
	n := C.sl_tokenize(m.m, ctext, C.int(len(text)), (*C.int32_t)(&out[0]), C.int(len(out)))
	if n < 0 { // more tokens than bytes cannot happen, but the API says how many it needs
		out = make([]int32, -n)
		n = C.sl_tokenize(m.m, ctext, C.int(len(text)), (*C.int32_t)(&out[0]), C.int(len(out)))
	}
	if n < 0 {
		return nil, errors.New("the engine cannot tokenize the prompt")
	}
	return out[:n], nil
}

// letterLogits reads tokens as one prompt and returns the next-token logits
// of letters. The context grows to the tier the prompt needs.
func (m *model) letterLogits(tokens, letters []int32) ([]float32, error) {
	if len(tokens) == 0 || len(tokens) > maxPromptTokens {
		return nil, fmt.Errorf("a prompt of %d tokens", len(tokens))
	}
	need := contextTiers[len(contextTiers)-1]
	for _, t := range contextTiers {
		if len(tokens) <= t {
			need = t
			break
		}
	}
	if m.ctxSize < need {
		if m.ctx != nil {
			C.sl_free_context(m.ctx)
			m.ctx, m.ctxSize = nil, 0
		}
		buf := make([]C.char, errCap)
		c := C.sl_new_context(m.m, C.int(need), &buf[0], errCap)
		if c == nil {
			return nil, cErr(buf)
		}
		m.ctx, m.ctxSize = c, need
	}
	out := make([]float32, len(letters))
	if C.sl_eval(m.ctx, (*C.int32_t)(&tokens[0]), C.int(len(tokens)), (*C.int32_t)(&letters[0]), C.int(len(letters)), (*C.float)(&out[0])) != 0 {
		return nil, errors.New("the engine failed to read the prompt")
	}
	return out, nil
}
