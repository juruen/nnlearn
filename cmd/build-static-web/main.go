package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"nnlearn/internal/mnist"
	"nnlearn/internal/webdata"
)

func main() {
	var (
		dataDir   = flag.String("data-dir", "data", "directory containing MNIST data files")
		modelPath = flag.String("model-path", filepath.Join("model", "model.json"), "path to the serialized model JSON")
		webDir    = flag.String("web-dir", "web", "directory containing the static web app")
		limit     = flag.Int("samples", 100, "number of training samples to export")
	)
	flag.Parse()

	if err := buildStaticWeb(*dataDir, *modelPath, *webDir, *limit); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func buildStaticWeb(dataDir, modelPath, webDir string, limit int) error {
	if err := os.MkdirAll(filepath.Join(webDir, "model"), 0o755); err != nil {
		return fmt.Errorf("failed to create model directory: %w", err)
	}

	if err := copyFile(modelPath, filepath.Join(webDir, "model", "model.json")); err != nil {
		return fmt.Errorf("failed to copy model.json: %w", err)
	}

	wasmExecPath, err := findWasmExecJS()
	if err != nil {
		return err
	}

	if err := copyFile(wasmExecPath, filepath.Join(webDir, "wasm_exec.js")); err != nil {
		return fmt.Errorf("failed to copy wasm_exec.js: %w", err)
	}

	dataset, err := mnist.Load(dataDir, "train")
	if err != nil {
		return fmt.Errorf("failed to load MNIST training data: %w", err)
	}

	samples, err := webdata.BuildTrainingSamples(dataset, limit)
	if err != nil {
		return fmt.Errorf("failed to build training samples: %w", err)
	}

	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal training samples: %w", err)
	}

	if err := os.WriteFile(filepath.Join(webDir, "training-samples.json"), data, 0o644); err != nil {
		return fmt.Errorf("failed to write training-samples.json: %w", err)
	}

	return nil
}

func copyFile(src, dst string) error {
	input, err := os.Open(src)
	if err != nil {
		return err
	}
	defer input.Close() //nolint:errcheck

	output, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer output.Close() //nolint:errcheck

	if _, err := io.Copy(output, input); err != nil {
		return err
	}

	if err := output.Close(); err != nil {
		return err
	}

	return nil
}

func findWasmExecJS() (string, error) {
	candidates := []string{
		filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"),
		filepath.Join(runtime.GOROOT(), "misc", "wasm", "wasm_exec.js"),
	}

	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("could not find wasm_exec.js under %s", runtime.GOROOT())
}
