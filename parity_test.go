package main

// Parity with the reference implementation (scripts/reference.py, docs/parity.md).
// The prompt and read-out tests need only the fixtures; the engine tests also
// need the pinned llama.cpp build and the model they were made with:
//
//	STARTLUX_ENGINE=<dir with libllama> STARTLUX_MODEL=<model dir: model.gguf, decision_config.json> \
//	STARTLUX_REF=testdata/ref/2b-q8.json go test -run Parity

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
)

type parityCase struct {
	Name     string   `json:"name"`
	State    string   `json:"state"`
	Question question `json:"question"`
}

type parityRef struct {
	Name          string    `json:"name"`
	Text          string    `json:"text"`
	IDs           []int32   `json:"ids"`
	LogProbs      []float32 `json:"logprobs"`
	Probabilities []float64 `json:"probabilities"`
}

func readJSON[T any](t *testing.T, path string) T {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var v T
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func parityFixtures(t *testing.T) ([]parityCase, map[string]parityRef, string) {
	t.Helper()
	cases, refPath := "testdata/cases.json", os.Getenv("STARTLUX_REF")
	if refPath == "" {
		refPath = "testdata/ref/2b-q8.json"
	}
	if c := os.Getenv("STARTLUX_CASES"); c != "" {
		cases = c
	}
	refs := map[string]parityRef{}
	for _, r := range readJSON[[]parityRef](t, refPath) {
		refs[r.Name] = r
	}
	return readJSON[[]parityCase](t, cases), refs, refPath
}

// The prompt is the reference's byte for byte.
func TestParityPrompt(t *testing.T) {
	cases, refs, _ := parityFixtures(t)
	for _, c := range cases {
		r, ok := render(c.State, c.Question)
		if !ok {
			t.Fatalf("%s: not rendered", c.Name)
		}
		if got, want := promptText(r), refs[c.Name].Text; got != want {
			t.Errorf("%s: prompt differs\n got: %q\nwant: %q", c.Name, got, want)
		}
	}
}

// The read-out math is the reference's: same letter log-probabilities, same option probabilities.
func TestParityReadout(t *testing.T) {
	cases, refs, refPath := parityFixtures(t)
	cfg := testConfig(t, refPath)
	for _, c := range cases {
		r, _ := render(c.State, c.Question)
		ref := refs[c.Name]
		p := probabilities(ref.LogProbs, cfg.TemperatureByType[r.typ])
		for i := range p {
			if math.Abs(p[i]-ref.Probabilities[i]) > 1e-6 { // logprobs travel as float32
				t.Errorf("%s: option %d: %.9f, reference %.9f", c.Name, i, p[i], ref.Probabilities[i])
			}
		}
	}
}

// testConfig is the decision config the reference fixture was made with.
func testConfig(t *testing.T, refPath string) decisionConfig {
	t.Helper()
	dir := os.Getenv("STARTLUX_MODEL")
	if dir == "" {
		dir = "testdata/config/" + map[string]string{"testdata/ref/2b-q8.json": "2b-q8", "testdata/ref/4b-q4.json": "4b-q4"}[refPath]
	}
	cfg, err := readDecisionConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// Read in-process, the engine gives the reference's tokens and probabilities,
// whichever context tier a prompt lands in.
func TestParityEngine(t *testing.T) {
	engine, dir := os.Getenv("STARTLUX_ENGINE"), os.Getenv("STARTLUX_MODEL")
	if engine == "" || dir == "" {
		t.Skip("STARTLUX_ENGINE and STARTLUX_MODEL not set")
	}
	cases, refs, refPath := parityFixtures(t)
	cfg := testConfig(t, refPath)
	if err := openEngine(engine); err != nil {
		t.Fatal(err)
	}
	m, err := loadModel(dir + "/model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer m.free()
	worst := 0.0
	first := map[string][]float64{}
	check := func(c parityCase) []float64 {
		r, _ := render(c.State, c.Question)
		ref := refs[c.Name]
		toks, err := m.tokenize(promptText(r))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(toks, ref.IDs) {
			t.Errorf("%s: %d tokens differ from the reference's %d", c.Name, len(toks), len(ref.IDs))
			return nil
		}
		z, err := m.letterLogits(toks, cfg.LetterTokenIDs[:len(r.options)])
		if err != nil {
			t.Fatal(err)
		}
		p := probabilities(z, cfg.TemperatureByType[r.typ])
		for i := range p {
			d := math.Abs(p[i] - ref.Probabilities[i])
			worst = max(worst, d)
			if d > 1e-5 {
				t.Errorf("%s: option %d: %.6f, reference %.6f", c.Name, i, p[i], ref.Probabilities[i])
			}
		}
		return p
	}
	slices.SortStableFunc(cases, func(a, b parityCase) int { return len(a.State) - len(b.State) }) // small tiers first
	for _, c := range cases {
		first[c.Name] = check(c)
	}
	t.Logf("%d cases, largest difference %.2e, context grew to %d", len(cases), worst, m.ctxSize)
	// The short prompts again, now in the largest context: the tier changes nothing.
	for _, c := range cases[:5] {
		again := check(c)
		for i := range again {
			if math.Abs(again[i]-first[c.Name][i]) > 1e-6 {
				t.Errorf("%s: %d context gives %.6f, smaller one %.6f", c.Name, m.ctxSize, again[i], first[c.Name][i])
			}
		}
	}
}

// Evidence too long for the model is cut and still answered; a question too
// long even without it is an invalid request, not a refusal.
func TestLongPrompts(t *testing.T) {
	engine, dir := os.Getenv("STARTLUX_ENGINE"), os.Getenv("STARTLUX_MODEL")
	if engine == "" || dir == "" {
		t.Skip("STARTLUX_ENGINE and STARTLUX_MODEL not set")
	}
	if err := openEngine(engine); err != nil {
		t.Fatal(err)
	}
	cfg, err := readDecisionConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := loadModel(dir + "/model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer m.free()
	q := question{Name: "bill", Type: "predicate", Instructions: "Answer yes if this email is a bill."}
	long := strings.Repeat("Your invoice for October is ready; the amount due is 42.00 USD. 账单已出。", 1500)
	a, perr := judgeOne(m, cfg, long, q)
	if perr != nil || a.Probability == nil || *a.Probability < 0 || *a.Probability > 1 {
		t.Fatalf("%+v %+v", a, perr)
	}
	q.Instructions = strings.Repeat("Answer yes only if this is a bill. ", 1500)
	if _, perr := judgeOne(m, cfg, "x", q); perr == nil || perr.Code != "invalid_request" {
		t.Fatalf("%+v", perr)
	}
}
