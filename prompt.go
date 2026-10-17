package main

// The prompt format StartLux-Decision was trained on, ported from the
// reference startlux_decide/prompt.py (its from_systemone and messages): every
// string here is part of that format, and changing one changes the answers.
// Questions arrive in Godwit's shape and are read as Godwit's System One
// client sends them (core/ai/judge_systemone.go), so a plugin answer and a
// System One answer to the same question see the same prompt.

import (
	"regexp"
	"strings"
)

const (
	systemLine       = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
	noulInstructions = "Which answer fits the evidence?"
	maxOptions       = 26 // one letter per option, A to Z
	maxLevels        = 10
	letters          = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

// question is Godwit's ai.Question as JSON.
type question struct {
	Name         string `json:"name"`
	Type         string `json:"type"` // predicate, choice or score
	Instructions string `json:"instructions"`
	Choices      []struct {
		Value       string `json:"value"`
		Description string `json:"description"`
	} `json:"choices"`
	Levels []struct {
		Label       string `json:"label"`
		Description string `json:"description"`
	} `json:"levels"`
}

// option is a rendered option: criterion "" is none.
type option struct{ id, criterion string }

// row is a rendered question, prompt.py's Row.
type row struct {
	typ          string // noul, choice or score
	state        string
	instructions string
	options      []option
}

// valueText is prompt.py's value_text for a string: trimmed, "" for none.
func valueText(s string) string { return strings.TrimSpace(s) }

// levelText is how Godwit's System One client describes a score level.
func levelText(label, description string) string {
	switch {
	case description == "":
		return label
	case label == "":
		return description
	}
	return label + ": " + description
}

// render is from_systemone for one question; ok false when the format cannot
// hold it in one pass (more than 26 options, or an id it cannot show), which
// the plugin answers as a refusal.
func render(state string, q question) (row, bool) {
	r := row{state: state, instructions: valueText(q.Instructions)}
	switch q.Type {
	case "predicate":
		r.typ = "noul"
		r.options = []option{{"true", ""}, {"false", ""}}
		if r.instructions == "" {
			r.instructions = noulInstructions
		}
	case "choice":
		r.typ = "choice"
		for _, c := range q.Choices {
			r.options = append(r.options, option{c.Value, valueText(c.Description)})
		}
	case "score":
		r.typ = "score"
		for i, l := range q.Levels {
			r.options = append(r.options, option{itoa(i), valueText(levelText(l.Label, l.Description))})
		}
	default:
		return r, false
	}
	return r, valid(r)
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

// valid is validate_row: what one pass of the format can show.
func valid(r row) bool {
	if r.instructions == "" || len(r.options) < 2 || len(r.options) > maxOptions || r.typ == "score" && len(r.options) > maxLevels {
		return false
	}
	seen := map[string]bool{}
	for _, o := range r.options {
		if strings.TrimSpace(o.id) == "" || strings.ContainsAny(o.id, "\n\r") || len([]rune(o.id)) > 200 || seen[o.id] {
			return false
		}
		seen[o.id] = true
	}
	return true
}

var (
	bareID = regexp.MustCompile(`(?i)^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$`)
	// Python's \s is Unicode whitespace; Go's is ASCII, hence the classes.
	normRE = regexp.MustCompile(`[\s\p{Z}\x{85}\x{1c}-\x{1f}_\-]+`)
)

func norm(s string) string {
	return strings.ToLower(strings.TrimSpace(normRE.ReplaceAllString(s, " ")))
}

// optionLines is option_lines: lettered, in the question's order.
func optionLines(r row) []string {
	hide := r.typ == "choice"
	for _, o := range r.options {
		hide = hide && bareID.MatchString(o.id) && o.criterion != ""
	}
	lines := make([]string, len(r.options))
	for k, o := range r.options {
		name := o.id
		if r.typ == "noul" {
			name = map[string]string{"true": "yes", "false": "no"}[o.id]
		}
		body := name + ": " + o.criterion
		switch {
		case hide:
			body = o.criterion
		case o.criterion == "" || norm(o.criterion) == norm(name):
			body = name
		}
		lines[k] = letters[k:k+1] + ") " + body
	}
	return lines
}

// promptText is the chat prompt with thinking off, as the model's chat
// template renders messages() with add_generation_prompt and enable_thinking false.
func promptText(r row) string {
	state := r.state
	if strings.TrimSpace(state) == "" {
		state = "(none)"
	}
	user := "Evidence:\n" + state + "\n\nQuestion: " + r.instructions + "\nOptions:\n" + strings.Join(optionLines(r), "\n")
	return "<|im_start|>system\n" + systemLine + "<|im_end|>\n<|im_start|>user\n" + user +
		"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
}
