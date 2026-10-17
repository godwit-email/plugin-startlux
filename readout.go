package main

// From letter logits to Godwit answers, ported from the reference
// startlux_decide/readout.py. The reference reads probabilities over the k
// letters (a +100 bias on each, so the softmax is over them alone), takes
// their logs and applies softmax(z / T): the same as softmax(logit / T) here.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
)

// decisionConfig is a model's decision_config.json: the letter token ids
// and the temperature of each question type.
type decisionConfig struct {
	LetterTokenIDs    []int32            `json:"letter_token_ids"`
	TemperatureByType map[string]float64 `json:"temperature_by_type"`
}

func readDecisionConfig(dir string) (decisionConfig, error) {
	b, err := os.ReadFile(filepath.Join(dir, "decision_config.json"))
	if err != nil {
		return decisionConfig{}, err
	}
	var c decisionConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return decisionConfig{}, err
	}
	if len(c.LetterTokenIDs) != len(letters) {
		return decisionConfig{}, errBadConfig
	}
	for _, t := range []string{"noul", "choice", "score"} {
		if v := c.TemperatureByType[t]; !(v > 0) || math.IsInf(v, 0) {
			return decisionConfig{}, errBadConfig
		}
	}
	return c, nil
}

// probabilities is softmax(z / temperature).
func probabilities(z []float32, temperature float64) []float64 {
	m := float64(slices.Max(z))
	p := make([]float64, len(z))
	sum := 0.0
	for i, x := range z {
		p[i] = math.Exp((float64(x) - m) / temperature)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// choiceConfidence is (n·p_max − 1) / (n − 1), clamped to [0, 1].
func choiceConfidence(p []float64) float64 {
	n := float64(len(p))
	if n < 2 {
		return 1
	}
	return min(1, max(0, (n*slices.Max(p)-1)/(n-1)))
}

// scoreConfidence is 1 − the expected distance from the likeliest level,
// relative to that of an even spread from the middle, floored at 0.
func scoreConfidence(p []float64) float64 {
	n := len(p)
	m := 0
	for i := range p {
		if p[i] > p[m] {
			m = i
		}
	}
	spread, even := 0.0, 0.0
	for i, v := range p {
		spread += v * math.Abs(float64(i-m))
		even += math.Abs(float64(i) - float64(n-1)/2)
	}
	even /= float64(n)
	if even == 0 {
		return 1
	}
	return max(0, 1-spread/even)
}

type probability struct {
	Value       any     `json:"value"`
	Probability float64 `json:"probability"`
}

// answer is Godwit's ai.Answer as JSON.
type answer struct {
	Type          string        `json:"type"`
	Name          string        `json:"name"`
	Probability   *float64      `json:"probability,omitempty"`
	Choice        string        `json:"choice,omitempty"`
	Score         *float64      `json:"score,omitempty"`
	Probabilities []probability `json:"probabilities,omitempty"`
	Confidence    *float64      `json:"confidence,omitempty"`
}

// toAnswer is readout.py's answer for one question of Godwit's type, with
// option probabilities p in option order.
func toAnswer(q question, r row, p []float64) answer {
	a := answer{Type: q.Type, Name: q.Name}
	switch r.typ {
	case "noul":
		a.Probability = &p[0] // options are true, then false
	case "choice":
		best := 0
		for i := range p {
			if p[i] > p[best] { // a tie goes to the option listed first
				best = i
			}
		}
		a.Choice = r.options[best].id
		c := choiceConfidence(p)
		a.Confidence = &c
		for i, o := range r.options {
			a.Probabilities = append(a.Probabilities, probability{o.id, p[i]})
		}
	case "score":
		s := 0.0
		for i, v := range p {
			s += float64(i) * v
			a.Probabilities = append(a.Probabilities, probability{i, v})
		}
		c := scoreConfidence(p)
		a.Score, a.Confidence = &s, &c
	}
	return a
}
