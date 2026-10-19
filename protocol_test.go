package main

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func call(t *testing.T, method, req string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(dispatch(method, []byte(req)), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func errCode(out map[string]any) string {
	e, _ := out["error"].(map[string]any)
	s, _ := e["code"].(string)
	return s
}

func TestDescribe(t *testing.T) {
	out := call(t, "describe", `{}`)
	r := out["result"].(map[string]any)
	if r["name"] != "startlux" || r["abi"] != 1.0 || r["capabilities"].([]any)[0] != "models" {
		t.Fatalf("describe: %v", out)
	}
	if len(r["methods"].([]any)) != 3 {
		t.Fatalf("methods: %v", r["methods"])
	}
}

func TestConfigureTakesNothing(t *testing.T) {
	if out := call(t, "configure", `{"config":{}}`); out["ok"] != true {
		t.Fatalf("%v", out)
	}
	if out := call(t, "configure", `{"config":{},"assets":{"engine":"/x"}}`); errCode(out) != "invalid_request" {
		t.Fatalf("assets come with models.load: %v", out)
	}
}

func TestModelsList(t *testing.T) {
	ms := call(t, "models.list", `{}`)["result"].(map[string]any)["models"].([]any)
	if len(ms) != len(catalog) {
		t.Fatalf("%v", ms)
	}
	for _, m := range ms {
		m := m.(map[string]any)
		if m["transport"] != "plugin" || m["operation"] != "judge" || m["assets"].(map[string]any)["model"] == "" {
			t.Fatalf("%v", m)
		}
	}
}

func TestLoadRejectsBadRequests(t *testing.T) {
	for _, req := range []string{
		`{"modelId":"decision-4b-q4","assets":{"engine":"/e"}}`,
		`{"modelId":"nope","assets":{"engine":"/e","model":"/m"}}`,
		`{"modelId":"decision-4b-q4","assets":{"engine":"relative","model":"/m"}}`,
	} {
		if out := call(t, "models.load", req); errCode(out) != "invalid_request" {
			t.Fatalf("%s: %v", req, out)
		}
	}
}

func TestJudgeNeedsLoadedModel(t *testing.T) {
	out := call(t, "models.judge", `{"modelId":"decision-4b-q4","input":"x","questions":[{"name":"a","type":"predicate","instructions":"q"}]}`)
	if errCode(out) != "model_not_ready" {
		t.Fatalf("%v", out)
	}
	if out := call(t, "models.judge", `{"modelId":"decision-4b-q4","input":"x","questions":[]}`); errCode(out) != "invalid_request" {
		t.Fatalf("%v", out)
	}
}

// What one pass of the format cannot hold is a refusal, never a guess.
func TestRenderLimits(t *testing.T) {
	q := question{Name: "c", Type: "choice", Instructions: "Pick one."}
	for i := range 27 {
		q.Choices = append(q.Choices, struct {
			Value       string `json:"value"`
			Description string `json:"description"`
		}{Value: "v" + itoa(i)})
	}
	if _, ok := render("s", q); ok {
		t.Fatal("27 options rendered")
	}
	q.Choices = q.Choices[:2]
	q.Choices[1].Value = "two\nlines"
	if _, ok := render("s", q); ok {
		t.Fatal("a multi-line id rendered")
	}
	if _, ok := render("s", question{Name: "x", Type: "unknown", Instructions: "q"}); ok {
		t.Fatal("an unknown type rendered")
	}
}

func TestPredicateDefaultsAndNone(t *testing.T) {
	r, ok := render("  \n", question{Name: "p", Type: "predicate", Instructions: "  "})
	if !ok {
		t.Fatal("not rendered")
	}
	text := promptText(r)
	if !strings.Contains(text, "Evidence:\n(none)\n") || !strings.Contains(text, "Question: "+noulInstructions) ||
		!strings.Contains(text, "A) yes\nB) no") {
		t.Fatalf("%q", text)
	}
}

func TestScoreAnswer(t *testing.T) {
	q := question{Name: "s", Type: "score"}
	r := row{typ: "score", options: []option{{"0", ""}, {"1", ""}, {"2", ""}}}
	a := toAnswer(q, r, []float64{0.2, 0.3, 0.5})
	if math.Abs(*a.Score-1.3) > 1e-12 || len(a.Probabilities) != 3 || a.Probabilities[2].Value != 2 {
		t.Fatalf("%+v", a)
	}
	c := toAnswer(question{Name: "c", Type: "choice"}, row{typ: "choice", options: []option{{"a", ""}, {"b", ""}}}, []float64{0.5, 0.5})
	if c.Choice != "a" {
		t.Fatalf("a tie goes to the first option: %+v", c)
	}
}

// Only Apple silicon offers 4B; elsewhere 2B alone, marked slow.
func TestCatalogFor(t *testing.T) {
	if c := catalogFor("darwin-arm64"); len(c) != 2 || c[1].Description != nil {
		t.Fatalf("%+v", c)
	}
	for _, p := range []string{"darwin-amd64", "linux-amd64", "linux-arm64", "windows-amd64", "windows-arm64"} {
		if c := catalogFor(p); len(c) != 1 || c[0].ID != "decision-2b-q8" || c[0].Description["default"] == "" {
			t.Fatalf("%s: %+v", p, c)
		}
	}
}
