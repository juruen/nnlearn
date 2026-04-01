package nn

import (
	"encoding/json"
	"fmt"
	"os"

	"nnlearn/internal/types"

	"gonum.org/v1/gonum/mat"
)

type serializedLayer struct {
	WeightRows int       `json:"weight_rows"`
	WeightCols int       `json:"weight_cols"`
	Weights    []float64 `json:"weights"`
	Biases     []float64 `json:"biases"`
}

type serializedNN struct {
	InputLen   int               `json:"input_len"`
	OutputLen  int               `json:"output_len"`
	HiddenLens []int             `json:"hidden_lens"`
	Layers     []serializedLayer `json:"layers"`
}

// Save serializes the neural network to a JSON file at the given path.
func (ff *FeedForward) Save(path string) error {
	s := serializedNN{
		InputLen:   ff.inputLen,
		OutputLen:  ff.outputLen,
		HiddenLens: ff.hiddenLens,
		Layers:     make([]serializedLayer, len(ff.weights)),
	}

	for i := range ff.weights {
		rows, cols := ff.weights[i].Dims()
		weights := make([]float64, rows*cols)
		for r := range rows {
			for c := range cols {
				weights[r*cols+c] = ff.weights[i].At(r, c)
			}
		}

		biasLen := ff.biases[i].Len()
		biases := make([]float64, biasLen)
		for j := range biasLen {
			biases[j] = ff.biases[i].AtVec(j)
		}

		s.Layers[i] = serializedLayer{
			WeightRows: rows,
			WeightCols: cols,
			Weights:    weights,
			Biases:     biases,
		}
	}

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal neural network: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// LoadFeedForwardBytes loads a neural network from serialized JSON data.
// Options (activation, cost) can be provided to configure the loaded network.
func LoadFeedForwardBytes(data []byte, opts ...Option) (*FeedForward, error) {
	var s serializedNN
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("failed to unmarshal neural network: %w", err)
	}

	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}

	weights := make([]types.Matrix, len(s.Layers))
	biases := make([]types.Vector, len(s.Layers))

	for i, layer := range s.Layers {
		weights[i] = mat.NewDense(layer.WeightRows, layer.WeightCols, layer.Weights)
		biases[i] = mat.NewVecDense(len(layer.Biases), layer.Biases)
	}

	return &FeedForward{
		inputLen:     s.InputLen,
		outputLen:    s.OutputLen,
		hiddenLens:   s.HiddenLens,
		weights:      weights,
		biases:       biases,
		activateFunc: o.activation,
		costFunc:     o.cost,
	}, nil
}

// LoadFeedForward loads a neural network from a JSON file at the given path.
// Options (activation, cost) can be provided to configure the loaded network.
func LoadFeedForward(path string, opts ...Option) (*FeedForward, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return LoadFeedForwardBytes(data, opts...)
}
