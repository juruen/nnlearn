// Package nn implements neural network construction and training.
package nn

import (
	"fmt"
	"math/rand/v2"

	maths "nnlearn/internal/math"
	"nnlearn/internal/types"
)

// FeedForward is a feedforward neural network.
type FeedForward struct {
	inputLen     int
	outputLen    int
	hiddenLens   []int
	weights      []types.Matrix
	biases       []types.Vector
	activateFunc types.Activation
	costFunc     types.Cost
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
		inputLen:     inputLen,
		outputLen:    outputLen,
		hiddenLens:   hiddenLens,
		weights:      weights,
		biases:       biases,
		activateFunc: o.activation,
		costFunc:     o.cost,
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
func (ff *FeedForward) Train(b types.TrainingBatch) (*types.TrainBatchResult, error) {
	if len(b.Inputs) != len(b.Outputs) {
		return nil, fmt.Errorf("inputs and outputs have different lengths")
	}

	for i := range b.Inputs {
		if b.Inputs[i].Len() != ff.inputLen {
			return nil, fmt.Errorf("%w: input length mismatch for sample %d: expected %d, got %d",
				types.ErrDimensionMismatch, i, ff.inputLen, b.Inputs[i].Len())
		}
	}

	for i := range b.Outputs {
		if b.Outputs[i].Len() != ff.outputLen {
			return nil, fmt.Errorf("%w: output length mismatch for sample %d: expected %d, got %d",
				types.ErrDimensionMismatch, i, ff.outputLen, b.Outputs[i].Len())
		}
	}

	result := types.TrainBatchResult{
		Results: make([]types.TrainSingleResult, 0, len(b.Inputs)),
	}

	// Backpropagation algorithm

	// n is the number of training samples in the batch
	n := len(b.Inputs)

	// For each training sample i
	for i := 0; i < n; i++ {
		// aˡ are the activation vectors for each layer
		aVecs := make([]types.Vector, 2+len(ff.hiddenLens))

		// Step 1 — Set a⁰ = input vector for this training sample
		aVecs[0] = b.Inputs[i]

		// Step 2 — Feedforward: compute zˡ and aˡ for each layer l
		// zˡ = Wˡ·aˡ⁻¹ + bˡ
		zVecs := make([]types.Vector, len(ff.hiddenLens)+1)
		for l := 0; l < len(ff.hiddenLens)+1; l++ {
			// Wˡ · aˡ⁻¹
			weightsTimesA := maths.MulMatrixVector(ff.Weights(l), aVecs[l])
			// zˡ = Wˡ·aˡ⁻¹ + bˡ
			zVecs[l] = maths.AddVectors(weightsTimesA, ff.Biases(l))
			// aˡ = σ(zˡ)
			aVecs[l+1] = maths.ApplyFuncToVector(zVecs[l], ff.activateFunc.Activate)
		}

		// Step 3 — Compute output error δᴸ = ∂C/∂a ⊙ σ′(zᴸ)
		deltaVecs := make([]types.Vector, len(ff.hiddenLens)+1)

		// ∂C/∂aᴸ at the output layer
		gradientAL, err := ff.costFunc.PartialCostA(b.Outputs[i], aVecs[len(aVecs)-1])
		if err != nil {
			return nil, fmt.Errorf("failed to compute partial cost derivative: %w", err)
		}

		// σ′(zᴸ)
		sigmaPrimeZL := maths.ApplyFuncToVector(zVecs[len(zVecs)-1], ff.activateFunc.ActivatePrime)

		// δᴸ = ∂C/∂aᴸ ⊙ σ′(zᴸ)
		deltaVecs[len(deltaVecs)-1] = maths.MulElemVec(gradientAL, sigmaPrimeZL)

		// Step 4 — Backpropagate: δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ′(zˡ)  for l = L−1, L−2, …, 1
		for l := len(deltaVecs) - 2; l >= 0; l-- {
			// (Wˡ⁺¹)ᵀ
			weightsTrans := maths.Transpose(ff.Weights(l + 1))

			// (Wˡ⁺¹)ᵀ · δˡ⁺¹
			weightsEtaProd := maths.MulMatrixVector(weightsTrans, deltaVecs[l+1])

			// σ′(zˡ)
			sigmaPrimeZL := maths.ApplyFuncToVector(zVecs[l], ff.activateFunc.ActivatePrime)

			// δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ′(zˡ)
			deltaVecs[l] = maths.MulElemVec(weightsEtaProd, sigmaPrimeZL)
		}

		// Step 5 — Compute gradients: ∂C/∂Wˡ = δˡ ⊗ aˡ⁻¹,  ∂C/∂bˡ = δˡ
		weightGradients := make([]types.Matrix, 0, len(ff.hiddenLens)+1)
		biasGradients := make([]types.Vector, 0, len(ff.hiddenLens)+1)

		for l := 0; l < len(deltaVecs); l++ {
			// ∂C/∂Wˡ = δˡ ⊗ aˡ⁻¹  (outer product)
			weightGradients = append(weightGradients, maths.OuterProduct(deltaVecs[l], aVecs[l]))

			// ∂C/∂bˡ = δˡ
			biasGradients = append(biasGradients, deltaVecs[l])
		}

		result.Results = append(result.Results, types.TrainSingleResult{
			AVectors:        aVecs,
			ZVectors:        zVecs,
			DeltaVectors:    deltaVecs,
			WeightGradients: weightGradients,
			BiasGradients:   biasGradients,
		})
	}

	return &result, nil
}

func allLayers(inputLen int, hiddenLens []int, outputLen int) []int {
	layers := make([]int, 0, 2+len(hiddenLens))
	layers = append(layers, inputLen)
	layers = append(layers, hiddenLens...)
	layers = append(layers, outputLen)
	return layers
}
