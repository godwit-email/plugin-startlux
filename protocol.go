package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"io"
	"unicode/utf8"
)

//go:embed plugin.json
var manifestJSON []byte

type pluginError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

func failure(code, message string, retryable bool) *pluginError {
	return &pluginError{code, message, retryable}
}

func envelope(result any, err *pluginError) []byte {
	var value any
	if err != nil {
		value = struct {
			OK    bool         `json:"ok"`
			Error *pluginError `json:"error"`
		}{false, err}
	} else {
		value = struct {
			OK     bool `json:"ok"`
			Result any  `json:"result"`
		}{true, result}
	}
	b, marshalErr := json.Marshal(value)
	if marshalErr != nil || len(b) > 4<<20 {
		return []byte(`{"ok":false,"error":{"code":"internal","message":"Response exceeds ABI limits.","retryable":false}}`)
	}
	return b
}

// decodeObject reads one JSON object with only value's fields, none null.
func decodeObject(data []byte, value any) bool {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || len(data) > 1<<20 || data[0] != '{' || !utf8.Valid(data) {
		return false
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if d.Decode(value) != nil || d.Decode(new(any)) != io.EOF {
		return false
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(data, &fields) != nil {
		return false
	}
	for _, raw := range fields {
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return false
		}
	}
	return true
}

var methods = []string{"models.list", "models.load", "models.judge"}

func dispatch(method string, data []byte) []byte {
	bad := func() []byte {
		return envelope(nil, failure("invalid_request", "Invalid method or request JSON.", false))
	}
	switch method {
	case "describe":
		if !decodeObject(data, &struct{}{}) {
			return bad()
		}
		var m struct {
			Name         string
			Version      string
			Native       struct{ ABI int }
			Capabilities []string
		}
		if json.Unmarshal(manifestJSON, &m) != nil {
			return envelope(nil, failure("internal", "Invalid embedded manifest.", false))
		}
		return envelope(struct {
			Name         string   `json:"name"`
			Version      string   `json:"version"`
			ABI          int      `json:"abi"`
			Capabilities []string `json:"capabilities"`
			Methods      []string `json:"methods"`
		}{m.Name, m.Version, m.Native.ABI, m.Capabilities, methods}, nil)
	case "configure": // nothing to set; assets come with models.load
		var req struct {
			Config map[string]json.RawMessage `json:"config"`
		}
		if !decodeObject(data, &req) || req.Config == nil || len(req.Config) != 0 {
			return bad()
		}
		return envelope(struct{}{}, nil)
	case "models.list":
		if !decodeObject(data, &struct{}{}) {
			return bad()
		}
		return envelope(struct {
			Models []catalogModel `json:"models"`
		}{catalog}, nil)
	case "models.load":
		var req struct {
			ModelID string            `json:"modelId"`
			Assets  map[string]string `json:"assets"`
		}
		if !decodeObject(data, &req) || req.Assets["engine"] == "" || req.Assets["model"] == "" {
			return bad()
		}
		if err := load(req.ModelID, req.Assets["engine"], req.Assets["model"]); err != nil {
			return envelope(nil, err)
		}
		return envelope(struct{}{}, nil)
	case "models.judge":
		var req struct {
			ModelID   string     `json:"modelId"`
			Input     string     `json:"input"`
			Questions []question `json:"questions"`
		}
		// Lenient on fields Godwit may add to questions later.
		if data = bytes.TrimSpace(data); len(data) == 0 || data[0] != '{' || !utf8.Valid(data) || json.Unmarshal(data, &req) != nil || len(req.Questions) == 0 {
			return bad()
		}
		answers, err := judge(req.ModelID, req.Input, req.Questions)
		if err != nil {
			return envelope(nil, err)
		}
		return envelope(struct {
			Answers []answer `json:"answers"`
		}{answers}, nil)
	default:
		return bad()
	}
}
