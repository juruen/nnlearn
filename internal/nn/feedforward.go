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
	if err := ff.validateBatch(b); err != nil {
		return nil, err
	}

	result := types.TrainBatchResult{
		Results: make([]types.TrainSingleResult, 0, len(b.Inputs)),
	}

	for i := range b.Inputs {
		singleResult, err := ff.trainSingleSample(b.Inputs[i], b.Outputs[i])
		if err != nil {
			return nil, fmt.Errorf("failed to train sample %d: %w", i, err)
		}
		result.Results = append(result.Results, *singleResult)
	}

	ff.applyBatchGradients(b.LearningRate, &result)

	return &result, nil
}

// trainSingleSample runs backpropagation on a single training sample and returns the result.
func (ff *FeedForward) trainSingleSample(input, output types.Vector) (*types.TrainSingleResult, error) {
	// aˡ are the activation vectors for each layer
	aVecs := make([]types.Vector, 2+len(ff.hiddenLens))

	// Step 1 — Set a⁰ = input vector for this training sample
	aVecs[0] = input

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
	gradientAL, err := ff.costFunc.PartialCostA(output, aVecs[len(aVecs)-1])
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

	return &types.TrainSingleResult{
		AVectors:        aVecs,
		ZVectors:        zVecs,
		DeltaVectors:    deltaVecs,
		WeightGradients: weightGradients,
		BiasGradients:   biasGradients,
	}, nil
}

// validateBatch checks that the training batch has consistent dimensions.
func (ff *FeedForward) validateBatch(b types.TrainingBatch) error {
	if b.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %f", b.LearningRate)
	}

	if len(b.Inputs) != len(b.Outputs) {
		return fmt.Errorf("inputs and outputs have different lengths")
	}

	for i := range b.Inputs {
		if b.Inputs[i].Len() != ff.inputLen {
			return fmt.Errorf("%w: input length mismatch for sample %d: expected %d, got %d",
				types.ErrDimensionMismatch, i, ff.inputLen, b.Inputs[i].Len())
		}
	}

	for i := range b.Outputs {
		if b.Outputs[i].Len() != ff.outputLen {
			return fmt.Errorf("%w: output length mismatch for sample %d: expected %d, got %d",
				types.ErrDimensionMismatch, i, ff.outputLen, b.Outputs[i].Len())
		}
	}

	return nil
}

// applyBatchGradients performs the SGD weight update for a mini-batch.
// For each layer l, it computes:
//
//	Wˡ = Wˡ − (η/n) · Σᵢ ∂C/∂Wˡᵢ
//	bˡ = bˡ − (η/n) · Σᵢ ∂C/∂bˡᵢ
//
// where η is the learning rate, n is the batch size, and the sum is over all samples i.
func (ff *FeedForward) applyBatchGradients(rate float64, result *types.TrainBatchResult) {
	n := len(result.Results)
	if n == 0 {
		return
	}

	// η/n — learning rate scaled by batch size
	rateByN := rate / float64(n)

	result.UpdatedWeightGradients = make([]types.Matrix, 0, len(ff.hiddenLens)+1)
	result.UpdatedBiasGradients = make([]types.Vector, 0, len(ff.hiddenLens)+1)

	for i := range len(ff.weights) {
		// Σᵢ ∂C/∂Wˡᵢ  and  Σᵢ ∂C/∂bˡᵢ  — sum gradients across all samples
		w := result.Results[0].WeightGradients[i]
		b := result.Results[0].BiasGradients[i]
		for j := 1; j < len(result.Results); j++ {
			w = maths.AddMatrix(w, result.Results[j].WeightGradients[i])
			b = maths.AddVectors(b, result.Results[j].BiasGradients[i])
		}

		// −(η/n) · gradient
		scale := func(v float64) float64 {
			return -v * rateByN
		}

		// Wˡ = Wˡ − (η/n) · Σ ∂C/∂Wˡ
		w = maths.ApplyFuncToMatrix(w, scale)
		w = maths.AddMatrix(ff.weights[i], w)

		// bˡ = bˡ − (η/n) · Σ ∂C/∂bˡ
		b = maths.ApplyFuncToVector(b, scale)
		b = maths.AddVectors(ff.biases[i], b)

		ff.weights[i] = w
		ff.biases[i] = b
		result.UpdatedWeightGradients = append(result.UpdatedWeightGradients, w)
		result.UpdatedBiasGradients = append(result.UpdatedBiasGradients, b)
	}
}

func allLayers(inputLen int, hiddenLens []int, outputLen int) []int {
	layers := make([]int, 0, 2+len(hiddenLens))
	layers = append(layers, inputLen)
	layers = append(layers, hiddenLens...)
	layers = append(layers, outputLen)
	return layers
}
