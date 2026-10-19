package main

// The plugin's models (docs/superpowers/specs/2026-10-17-plugin-model-provider-design.md 7):
// two StartLux-Decision Instant builds as judge models, each one variant of
// the registry's "model" asset; the engine is the fixed "engine" asset. One
// model is loaded at a time, and one prompt is read at a time.

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

type catalogModel struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Transport   string            `json:"transport"`
	Operation   string            `json:"operation"`
	Assets      map[string]string `json:"assets"`
	Description map[string]string `json:"description,omitempty"`
}

// catalog is the models on this platform. Only Apple silicon has a GPU back
// end; elsewhere the CPU reads a 2K-token mail in tens of seconds a question
// (docs/parity.md), so only 2B is offered there, and it says it is slow.
var catalog = catalogFor(runtime.GOOS + "-" + runtime.GOARCH)

func catalogFor(platform string) []catalogModel {
	m2b := catalogModel{ID: "decision-2b-q8", Name: "StartLux Decision 2B Q8", Transport: "plugin", Operation: "judge", Assets: map[string]string{"model": "2b-q8"}}
	if platform != "darwin-arm64" {
		m2b.Description = map[string]string{"default": "Slow on this computer; best for labeling new mail only", "zh-CN": "在这台电脑上较慢，建议只给新邮件打标签"}
		return []catalogModel{m2b}
	}
	return []catalogModel{
		{ID: "decision-4b-q4", Name: "StartLux Decision 4B Q4", Transport: "plugin", Operation: "judge", Assets: map[string]string{"model": "4b-q4"}},
		m2b,
	}
}

var errBadConfig = errors.New("decision_config.json is not a StartLux-Decision read-out config")

// loaded is the plugin's one model; mu serializes everything that uses it.
var loaded struct {
	mu        sync.Mutex
	engineDir string
	id        string
	m         *model
	cfg       decisionConfig
}

// load makes modelID the loaded model from the verified directories, and
// reads one question with it: it is ready only if that works.
func load(modelID, engineDir, modelDir string) *pluginError {
	if !filepath.IsAbs(engineDir) || !filepath.IsAbs(modelDir) || !known(modelID) {
		return failure("invalid_request", "Unknown model or asset directories.", false)
	}
	loaded.mu.Lock()
	defer loaded.mu.Unlock()
	if loaded.engineDir != "" && loaded.engineDir != engineDir {
		// shortcut: one engine per process; Godwit starts a new process when the engine asset changes.
		return failure("internal", "The engine changed; the plugin must restart.", false)
	}
	if err := openEngine(engineDir); err != nil {
		return failure("internal", err.Error(), false)
	}
	loaded.engineDir = engineDir
	cfg, err := readDecisionConfig(modelDir)
	if err != nil {
		return failure("internal", "The model's decision config cannot be read: "+err.Error(), false)
	}
	if _, err := os.Stat(filepath.Join(modelDir, "model.gguf")); err != nil {
		return failure("internal", "The model file is missing.", false)
	}
	if loaded.m != nil { // one model at a time
		loaded.m.free()
		loaded.m, loaded.id = nil, ""
	}
	m, err := loadModel(filepath.Join(modelDir, "model.gguf"))
	if err != nil {
		return failure("internal", err.Error(), false)
	}
	probe := question{Name: "probe", Type: "predicate", Instructions: "Is this a test message?"}
	a, perr := judgeOne(m, cfg, "This is a test message.", probe)
	if perr != nil || a.Probability == nil || math.IsNaN(*a.Probability) {
		m.free()
		return failure("internal", "The model loaded but does not answer.", false)
	}
	loaded.id, loaded.m, loaded.cfg = modelID, m, cfg
	return nil
}

func known(id string) bool {
	for _, c := range catalog {
		if c.ID == id {
			return true
		}
	}
	return false
}

// judge answers questions about input with the loaded model, one prompt each.
func judge(modelID, input string, qs []question) ([]answer, *pluginError) {
	loaded.mu.Lock()
	defer loaded.mu.Unlock()
	if loaded.m == nil || loaded.id != modelID {
		return nil, failure("model_not_ready", "The model is not loaded.", true)
	}
	out := make([]answer, 0, len(qs))
	for _, q := range qs {
		a, err := judgeOne(loaded.m, loaded.cfg, input, q)
		if err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, nil
}

// judgeOne reads one question. Evidence too long for the model is cut (at a
// character, three tries at most); a question too long even without evidence
// is an invalid request, never a refusal, which a label classifier reads as no.
func judgeOne(m *model, cfg decisionConfig, input string, q question) (answer, *pluginError) {
	r, ok := render(input, q)
	if !ok {
		return answer{Type: "refusal", Name: q.Name}, nil
	}
	toks, err := m.tokenize(promptText(r))
	if err != nil {
		return answer{}, failure("internal", err.Error(), false)
	}
	if len(toks) > maxPromptTokens {
		bare := r
		bare.state = ""
		fixed, err := m.tokenize(promptText(bare))
		if err != nil {
			return answer{}, failure("internal", err.Error(), false)
		}
		if len(fixed) >= maxPromptTokens {
			return answer{}, failure("invalid_request", fmt.Sprintf("Question %s is too long for the model even without the evidence.", q.Name), false)
		}
		runes := []rune(r.state)
		for try := 0; try < 3 && len(toks) > maxPromptTokens; try++ {
			keep := int(float64(len(runes)) * float64(maxPromptTokens-len(fixed)) / float64(len(toks)-len(fixed)) * 0.95)
			runes = runes[:max(0, min(keep, len(runes)-1))]
			r.state = string(runes)
			if toks, err = m.tokenize(promptText(r)); err != nil {
				return answer{}, failure("internal", err.Error(), false)
			}
		}
		if len(toks) > maxPromptTokens {
			return answer{}, failure("invalid_request", "The evidence cannot be cut to fit the model.", false)
		}
	}
	z, err := m.letterLogits(toks, cfg.LetterTokenIDs[:len(r.options)])
	if err != nil {
		return answer{}, failure("internal", err.Error(), false)
	}
	return toAnswer(q, r, probabilities(z, cfg.TemperatureByType[r.typ])), nil
}
