// Package nn implements neural network construction and training.
package nn

import (
	"math/rand/v2"

	"nnlearn/internal/types"
)

// FeedForward is a feedforward neural network.
type FeedForward struct {
	inputLen   int
	outputLen  int
	hiddenLens []int
	weights    []types.Matrix
	biases     []types.Vector
	activation types.Activation
	cost       types.Cost
}

var _ types.NeuralNetwork = (*FeedForward)(nil)

// NewFeedForward creates a new neural network with the given layer sizes and options.
func NewFeedForward(inputLen int, hiddenLens []int, outputLen int, opts ...Option) *FeedForward {
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}

	var rng *rand.Rand
	if o.seed != nil {
		rng = rand.New(rand.NewPCG(*o.seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	layers := allLayers(inputLen, hiddenLens, outputLen)
	weights := make([]types.Matrix, len(layers)-1)
	biases := make([]types.Vector, len(layers)-1)
	for i := 1; i < len(layers); i++ {
		weights[i-1], biases[i-1] = o.initializer(rng, layers[i], layers[i-1])
	}

	return &FeedForward{
		inputLen:   inputLen,
		outputLen:  outputLen,
		hiddenLens: hiddenLens,
		weights:    weights,
		biases:     biases,
		activation: o.activation,
		cost:       o.cost,
	}
}

// InputLength returns the length of the input layer.
func (ff *FeedForward) InputLength() int {
	return ff.inputLen
}

// HiddenLengths returns the lengths of the hidden layers.
func (ff *FeedForward) HiddenLengths() []int {
	return ff.hiddenLens
}

// OutputLength returns the length of the output layer.
func (ff *FeedForward) OutputLength() int {
	return ff.outputLen
}

// Weights returns the weight matrix for the given layer.
func (ff *FeedForward) Weights(layer int) types.Matrix {
	return ff.weights[layer]
}

// Biases returns the bias vector for the given layer.
func (ff *FeedForward) Biases(layer int) types.Vector {
	return ff.biases[layer]
}

// Train trains a batch of training samples and returns the computed gradient.
func (ff *FeedForward) Train(_ types.TrainingBatch) (gradient types.Gradient) {
	// TODO implement me
	panic("implement me")
}

func allLayers(inputLen int, hiddenLens []int, outputLen int) []int {
	layers := make([]int, 0, 2+len(hiddenLens))
	layers = append(layers, inputLen)
	layers = append(layers, hiddenLens...)
	layers = append(layers, outputLen)
	return layers
}
