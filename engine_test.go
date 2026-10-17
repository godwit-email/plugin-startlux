package main

import (
	"os"
	"strings"
	"testing"
)

// The pinned engine loads from a directory laid out as Godwit stores it
// (scripts/engine.py), with its backends: CI checks this on every platform.
func TestEngineOpens(t *testing.T) {
	dir := os.Getenv("STARTLUX_ENGINE")
	if dir == "" {
		t.Skip("STARTLUX_ENGINE not set")
	}
	if err := openEngine(dir); err != nil {
		t.Fatal(err)
	}
	if info := systemInfo(); !strings.Contains(info, "CPU") {
		t.Fatalf("no CPU backend loaded: %q", info)
	}
	if err := openEngine(dir); err != nil { // once per process: a second open is a no-op
		t.Fatal(err)
	}
}
