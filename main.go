package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"nnlearn/internal/mnist"
	"nnlearn/internal/types"
)

func main() {
	wasmExecPath, err := findWasmExecJS()
	if err != nil {
		log.Fatal(err)
	}

	addr := ":8080"
	if envAddr := os.Getenv("NNLEARN_ADDR"); envAddr != "" {
		addr = envAddr
	}
	dataDir := "data"
	if envDataDir := os.Getenv("NNLEARN_DATA_DIR"); envDataDir != "" {
		dataDir = envDataDir
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", serveIndex)
	mux.HandleFunc("/app.js", serveWebAsset("app.js", "text/javascript; charset=utf-8"))
	mux.HandleFunc("/styles.css", serveWebAsset("styles.css", "text/css; charset=utf-8"))
	mux.HandleFunc("/app.wasm", serveFile(filepath.Join("web", "app.wasm"), "application/wasm"))
	mux.HandleFunc("/model/model.json", serveFile(filepath.Join("model", "model.json"), "application/json; charset=utf-8"))
	mux.HandleFunc("/wasm_exec.js", serveFile(wasmExecPath, "text/javascript; charset=utf-8"))
	mux.HandleFunc("/training-samples", trainingSamplesHandler(dataDir, 100))

	log.Printf("nnlearn digit recognizer available at http://localhost%s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatal(err)
	}
}

func serveIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/index.html" {
		http.NotFound(w, r)
		return
	}

	http.ServeFile(w, r, filepath.Join("web", "index.html"))
}

func serveWebAsset(name, contentType string) http.HandlerFunc {
	return serveFile(filepath.Join("web", name), contentType)
}

func serveFile(path, contentType string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		http.ServeFile(w, r, path)
	}
}

type trainingSample struct {
	Index  int       `json:"index"`
	Label  int       `json:"label"`
	Pixels []float64 `json:"pixels"`
}

func trainingSamplesHandler(dataDir string, limit int) http.HandlerFunc {
	var (
		once    sync.Once
		samples []trainingSample
		loadErr error
	)

	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		once.Do(func() {
			dataset, err := mnist.Load(dataDir, "train")
			if err != nil {
				loadErr = fmt.Errorf("failed to load training samples: %w", err)
				return
			}

			samples, loadErr = buildTrainingSamples(dataset, limit)
		})

		if loadErr != nil {
			http.Error(w, loadErr.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		if err := json.NewEncoder(w).Encode(samples); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode training samples: %v", err), http.StatusInternalServerError)
		}
	}
}

func buildTrainingSamples(dataset *mnist.Dataset, limit int) ([]trainingSample, error) {
	if dataset == nil {
		return nil, fmt.Errorf("dataset is nil")
	}

	count := min(limit, len(dataset.Images))
	if len(dataset.Labels) < count {
		return nil, fmt.Errorf("image count %d exceeds label count %d", count, len(dataset.Labels))
	}

	samples := make([]trainingSample, 0, count)
	for i := range count {
		samples = append(samples, trainingSample{
			Index:  i,
			Label:  argmaxLabel(dataset.Labels[i]),
			Pixels: vectorToSlice(dataset.Images[i]),
		})
	}

	return samples, nil
}

func argmaxLabel(v types.Vector) int {
	maxIdx := 0
	maxVal := v.AtVec(0)
	for i := 1; i < v.Len(); i++ {
		val := v.AtVec(i)
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

func vectorToSlice(v types.Vector) []float64 {
	data := make([]float64, v.Len())
	for i := range v.Len() {
		data[i] = v.AtVec(i)
	}
	return data
}

func findWasmExecJS() (string, error) {
	candidates := []string{
		filepath.Join(runtime.GOROOT(), "lib", "wasm", "wasm_exec.js"),
		filepath.Join(runtime.GOROOT(), "misc", "wasm", "wasm_exec.js"),
	}

	for _, candidate := range candidates {
		info, err := os.Stat(candidate)
		if err == nil && !info.IsDir() {
			return candidate, nil
		}
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("failed to inspect %s: %w", candidate, err)
		}
	}

	return "", fmt.Errorf("could not find wasm_exec.js under %s", runtime.GOROOT())
}
