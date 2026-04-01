package nn

import (
	"testing"

	"nnlearn/internal/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestNewFeedForward(t *testing.T) {
	t.Run("layer sizes", func(t *testing.T) {
		ff := NewFeedForward(784, []int{30, 20}, 10, WithSeed(42))

		assert.Equal(t, 784, ff.InputLength())
		assert.Equal(t, []int{30, 20}, ff.HiddenLengths())
		assert.Equal(t, 10, ff.OutputLength())
	})

	t.Run("weight dimensions per layer", func(t *testing.T) {
		ff := NewFeedForward(784, []int{30, 20}, 10, WithSeed(42))

		// Layer 0: 30 neurons, 784 inputs
		r, c := ff.Weights(0).Dims()
		assert.Equal(t, 30, r)
		assert.Equal(t, 784, c)

		// Layer 1: 20 neurons, 30 inputs
		r, c = ff.Weights(1).Dims()
		assert.Equal(t, 20, r)
		assert.Equal(t, 30, c)

		// Layer 2: 10 neurons, 20 inputs
		r, c = ff.Weights(2).Dims()
		assert.Equal(t, 10, r)
		assert.Equal(t, 20, c)
	})

	t.Run("bias dimensions per layer", func(t *testing.T) {
		ff := NewFeedForward(784, []int{30, 20}, 10, WithSeed(42))

		assert.Equal(t, 30, ff.Biases(0).Len())
		assert.Equal(t, 20, ff.Biases(1).Len())
		assert.Equal(t, 10, ff.Biases(2).Len())
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		ff1 := NewFeedForward(4, []int{3}, 2, WithSeed(99))
		ff2 := NewFeedForward(4, []int{3}, 2, WithSeed(99))

		for layer := range 2 {
			r, c := ff1.Weights(layer).Dims()
			for i := range r {
				for j := range c {
					require.Equal(t, ff1.Weights(layer).At(i, j), ff2.Weights(layer).At(i, j))
				}
			}
			for i := range ff1.Biases(layer).Len() {
				require.Equal(t, ff1.Biases(layer).AtVec(i), ff2.Biases(layer).AtVec(i))
			}
		}
	})

	t.Run("no hidden layers", func(t *testing.T) {
		ff := NewFeedForward(3, nil, 2, WithSeed(42))

		assert.Equal(t, 3, ff.InputLength())
		assert.Empty(t, ff.HiddenLengths())
		assert.Equal(t, 2, ff.OutputLength())

		r, c := ff.Weights(0).Dims()
		assert.Equal(t, 2, r)
		assert.Equal(t, 3, c)
		assert.Equal(t, 2, ff.Biases(0).Len())
	})
}

func vec(data ...float64) *mat.VecDense {
	return mat.NewVecDense(len(data), data)
}

func TestTrain(t *testing.T) {
	t.Run("mismatched input/output count", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1), vec(0)},
			LearningRate: 0.1,
		})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "different lengths")
	})

	t.Run("wrong input dimension", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2, 3)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		})

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})

	t.Run("wrong output dimension", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1, 2)},
			LearningRate: 0.1,
		})

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})

	t.Run("single sample returns correct structure", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(0.5, 0.8)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		})

		require.NoError(t, err)
		require.Len(t, result.Results, 1)

		r := result.Results[0]
		// aVecs: input + hidden + output = 3
		assert.Len(t, r.AVectors, 3)
		// zVecs: hidden + output = 2
		assert.Len(t, r.ZVectors, 2)
		// deltaVecs: hidden + output = 2
		assert.Len(t, r.DeltaVectors, 2)
		// gradients: one per layer (hidden + output) = 2
		assert.Len(t, r.WeightGradients, 2)
		assert.Len(t, r.BiasGradients, 2)
	})

	t.Run("gradient dimensions match weights", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(0.5, 0.8)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		})

		require.NoError(t, err)
		r := result.Results[0]

		for l := range r.WeightGradients {
			wr, wc := ff.Weights(l).Dims()
			gr, gc := r.WeightGradients[l].Dims()
			assert.Equal(t, wr, gr, "weight gradient rows mismatch at layer %d", l)
			assert.Equal(t, wc, gc, "weight gradient cols mismatch at layer %d", l)

			assert.Equal(t, ff.Biases(l).Len(), r.BiasGradients[l].Len(),
				"bias gradient length mismatch at layer %d", l)
		}
	})

	t.Run("batch of multiple samples", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(0.5, 0.8), vec(0.1, 0.9), vec(0.3, 0.4)},
			Outputs:      []types.Vector{vec(1), vec(0), vec(1)},
			LearningRate: 0.1,
		})

		require.NoError(t, err)
		assert.Len(t, result.Results, 3)
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		batch := types.TrainingBatch{
			Inputs:       []types.Vector{vec(0.5, 0.8)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		}

		ff1 := NewFeedForward(2, []int{3}, 1, WithSeed(42))
		ff2 := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		r1, err1 := ff1.Train(batch)
		r2, err2 := ff2.Train(batch)

		require.NoError(t, err1)
		require.NoError(t, err2)

		// Compare delta vectors
		for l := range r1.Results[0].DeltaVectors {
			for i := range r1.Results[0].DeltaVectors[l].Len() {
				assert.Equal(t,
					r1.Results[0].DeltaVectors[l].AtVec(i),
					r2.Results[0].DeltaVectors[l].AtVec(i),
				)
			}
		}
	})

	t.Run("no hidden layers", func(t *testing.T) {
		ff := NewFeedForward(2, nil, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{vec(0.5, 0.8)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		})

		require.NoError(t, err)
		require.Len(t, result.Results, 1)

		r := result.Results[0]
		assert.Len(t, r.AVectors, 2)
		assert.Len(t, r.ZVectors, 1)
		assert.Len(t, r.DeltaVectors, 1)
		assert.Len(t, r.WeightGradients, 1)
		assert.Len(t, r.BiasGradients, 1)
	})

	t.Run("empty batch", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:       []types.Vector{},
			Outputs:      []types.Vector{},
			LearningRate: 0.1,
		})

		require.NoError(t, err)
		assert.Empty(t, result.Results)
	})
}

// TestNoHiddenLayers tests a minimal 2→2 network (no hidden layers) end-to-end.
//
// Network: 2 inputs → 2 outputs, sigmoid activation, quadratic cost.
//
// Given:
//
//	x = [1, 2]ᵀ
//
//	W = ┌  1  -1 ┐    b = ┌ 0 ┐    y = ┌ 1 ┐
//	    └  0   2 ┘        └ 1 ┘        └ 0 ┘
//
// Step 1 — Forward pass:
//
//	z = W·x + b = [-1, 5]ᵀ
//	a = σ(z) ≈ [0.26894, 0.99331]ᵀ
//
// Step 2 — Cost:
//
//	C = ½ Σ(y − a)² ≈ 0.760
//
// Step 3 — Backprop (output error):
//
//	σ′(−1) ≈ 0.1966,  σ′(5) ≈ 0.00665
//	a − y = [−0.73106, 0.99331]ᵀ
//	δ = (a − y) ⊙ σ′(z) ≈ [−0.1437, 0.00660]ᵀ
//
// Step 4 — Gradients:
//
//	∂C/∂W = δ ⊗ xᵀ ≈ ┌ −0.1437  −0.2874 ┐
//	                 └  0.00660  0.0132 ┘
//
//	∂C/∂b = δ ≈ [−0.1437, 0.00660]ᵀ
//
// This test covers: matrix multiply, sigmoid (including saturation at z=5),
// element-wise ops (Hadamard), and outer product.
func TestNoHiddenLayers(t *testing.T) {
	ff := NewFeedForward(2, nil, 2, WithSeed(42))

	ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{
		1, -1,
		0, 2,
	})}

	ff.biases = []types.Vector{mat.NewVecDense(2, []float64{0, 1})}

	res, err := ff.trainSingleSample(vec(1, 2), vec(1, 0))
	require.NoError(t, err)

	tol := 1e-3

	// Verify z = W·x + b = [−1, 5]ᵀ
	require.Len(t, res.ZVectors, 1)
	assert.InDelta(t, -1.0, res.ZVectors[0].AtVec(0), tol)
	assert.InDelta(t, 5.0, res.ZVectors[0].AtVec(1), tol)

	// Verify a = σ(z) ≈ [0.26894, 0.99331]ᵀ
	require.Len(t, res.AVectors, 2)
	assert.InDelta(t, 0.26894, res.AVectors[1].AtVec(0), tol)
	assert.InDelta(t, 0.99331, res.AVectors[1].AtVec(1), tol)

	// Verify δ ≈ [−0.1437, 0.00660]ᵀ
	require.Len(t, res.DeltaVectors, 1)
	assert.InDelta(t, -0.1437, res.DeltaVectors[0].AtVec(0), tol)
	assert.InDelta(t, 0.00660, res.DeltaVectors[0].AtVec(1), tol)

	// Verify ∂C/∂W ≈ [−0.1437, −0.2874; 0.00660, 0.0132]
	require.Len(t, res.WeightGradients, 1)
	wg := res.WeightGradients[0]
	assert.InDelta(t, -0.1437, wg.At(0, 0), tol)
	assert.InDelta(t, -0.2874, wg.At(0, 1), tol)
	assert.InDelta(t, 0.00660, wg.At(1, 0), tol)
	assert.InDelta(t, 0.0132, wg.At(1, 1), tol)

	// Verify ∂C/∂b = δ ≈ [−0.1437, 0.00660]ᵀ
	require.Len(t, res.BiasGradients, 1)
	assert.InDelta(t, -0.1437, res.BiasGradients[0].AtVec(0), tol)
	assert.InDelta(t, 0.00660, res.BiasGradients[0].AtVec(1), tol)
}

// TestTwoLayersTrain tests a 2→2→1 network (hidden + output) end-to-end.
//
// Architecture: 2 inputs → 2 hidden → 1 output, sigmoid activation, quadratic cost.
//
// Given:
//
//	x = [1, 0]ᵀ    y = [1]
//
//	W¹ = ┌ 1  -1 ┐    b¹ = ┌ 0 ┐
//	     └ 0   1 ┘         └ 0 ┘
//
//	W² = [ 1  1 ]      b² = [ 0 ]
//
// Step 1 — Forward pass:
//
//	z¹ = W¹·x + b¹ = [1, 0]ᵀ
//	a¹ = σ(z¹) ≈ [0.73106, 0.5]ᵀ
//	z² = W²·a¹ + b² ≈ [1.23106]
//	a² = σ(z²) ≈ [0.77400]
//
// Step 2 — Cost:
//
//	C = ½(y − a²)² ≈ 0.02554
//
// Step 3 — Output delta:
//
//	σ′(z²) = a²(1−a²) ≈ 0.17492
//	δ² = (a² − y) ⊙ σ′(z²) ≈ [−0.03953]
//
// Step 4 — Output gradients:
//
//	∂C/∂W² = δ² ⊗ (a¹)ᵀ ≈ [−0.02890, −0.01977]
//	∂C/∂b² = δ² ≈ [−0.03953]
//
// Step 5 — Hidden delta (backprop through hidden layer):
//
//	(W²)ᵀ·δ² = [−0.03953, −0.03953]ᵀ
//	σ′(z¹) ≈ [0.19661, 0.25]ᵀ
//	δ¹ = ((W²)ᵀ·δ²) ⊙ σ′(z¹) ≈ [−0.00777, −0.00988]ᵀ
//
// Step 6 — Hidden gradients:
//
//	∂C/∂W¹ = δ¹ ⊗ xᵀ ≈ ┌ −0.00777  0 ┐
//	                   └ −0.00988  0 ┘
//	∂C/∂b¹ = δ¹ ≈ [−0.00777, −0.00988]ᵀ
//
// Note: second column of ∂C/∂W¹ is exactly 0 because x₂ = 0 (good sanity check).
//
// This test covers: multi-layer forward pass, sigmoid at hidden and output,
// quadratic cost, transpose in hidden backprop, and outer products for weight gradients.
func TestTwoLayersTrain(t *testing.T) {
	ff := NewFeedForward(2, []int{2}, 1, WithSeed(42))

	ff.weights = []types.Matrix{
		// W¹: hidden layer weights
		mat.NewDense(2, 2, []float64{
			1, -1,
			0, 1,
		}),

		// W²: output layer weights
		mat.NewDense(1, 2, []float64{1, 1}),
	}

	ff.biases = []types.Vector{
		// b¹: hidden layer biases
		mat.NewVecDense(2, []float64{0, 0}),

		// b²: output layer biases
		mat.NewVecDense(1, []float64{0}),
	}

	res, err := ff.trainSingleSample(vec(1, 0), vec(1))
	require.NoError(t, err)

	tol := 1e-3

	// --- Forward pass ---

	// z¹ = [1, 0]ᵀ
	require.Len(t, res.ZVectors, 2)
	assert.InDelta(t, 1.0, res.ZVectors[0].AtVec(0), tol)
	assert.InDelta(t, 0.0, res.ZVectors[0].AtVec(1), tol)

	// a¹ ≈ [0.73106, 0.5]ᵀ
	require.Len(t, res.AVectors, 3)
	assert.InDelta(t, 0.73106, res.AVectors[1].AtVec(0), tol)
	assert.InDelta(t, 0.5, res.AVectors[1].AtVec(1), tol)

	// z² ≈ [1.23106]
	assert.InDelta(t, 1.23106, res.ZVectors[1].AtVec(0), tol)

	// a² ≈ [0.77400]
	assert.InDelta(t, 0.77400, res.AVectors[2].AtVec(0), tol)

	// --- Deltas ---

	// δ² ≈ [−0.03953]
	require.Len(t, res.DeltaVectors, 2)
	assert.InDelta(t, -0.03953, res.DeltaVectors[1].AtVec(0), tol)

	// δ¹ ≈ [−0.00777, −0.00988]ᵀ
	assert.InDelta(t, -0.00777, res.DeltaVectors[0].AtVec(0), tol)
	assert.InDelta(t, -0.00988, res.DeltaVectors[0].AtVec(1), tol)

	// --- Output layer gradients ---

	// ∂C/∂W² ≈ [−0.02890, −0.01977]
	require.Len(t, res.WeightGradients, 2)
	assert.InDelta(t, -0.02890, res.WeightGradients[1].At(0, 0), tol)
	assert.InDelta(t, -0.01977, res.WeightGradients[1].At(0, 1), tol)

	// ∂C/∂b² ≈ [−0.03953]
	require.Len(t, res.BiasGradients, 2)
	assert.InDelta(t, -0.03953, res.BiasGradients[1].AtVec(0), tol)

	// --- Hidden layer gradients ---

	// ∂C/∂W¹ ≈ [−0.00777, 0; −0.00988, 0]
	assert.InDelta(t, -0.00777, res.WeightGradients[0].At(0, 0), tol)
	assert.InDelta(t, 0.0, res.WeightGradients[0].At(0, 1), tol)
	assert.InDelta(t, -0.00988, res.WeightGradients[0].At(1, 0), tol)
	assert.InDelta(t, 0.0, res.WeightGradients[0].At(1, 1), tol)

	// ∂C/∂b¹ ≈ [−0.00777, −0.00988]ᵀ
	assert.InDelta(t, -0.00777, res.BiasGradients[0].AtVec(0), tol)
	assert.InDelta(t, -0.00988, res.BiasGradients[0].AtVec(1), tol)
}

func TestValidateBatch(t *testing.T) {
	ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

	t.Run("valid batch", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2), vec(3, 4)},
			Outputs:      []types.Vector{vec(1), vec(0)},
			LearningRate: 0.1,
		})
		require.NoError(t, err)
	})

	t.Run("zero learning rate", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "learning rate must be positive")
	})

	t.Run("negative learning rate", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: -0.5,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "learning rate must be positive")
	})

	t.Run("mismatched input/output count", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1), vec(0)},
			LearningRate: 0.1,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "different lengths")
	})

	t.Run("wrong input dimension", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2, 3)},
			Outputs:      []types.Vector{vec(1)},
			LearningRate: 0.1,
		})
		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
		assert.Contains(t, err.Error(), "input length mismatch")
	})

	t.Run("wrong output dimension", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2)},
			Outputs:      []types.Vector{vec(1, 2)},
			LearningRate: 0.1,
		})
		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
		assert.Contains(t, err.Error(), "output length mismatch")
	})

	t.Run("empty batch", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{},
			Outputs:      []types.Vector{},
			LearningRate: 0.1,
		})
		require.NoError(t, err)
	})

	t.Run("error on second sample", func(t *testing.T) {
		err := ff.validateBatch(types.TrainingBatch{
			Inputs:       []types.Vector{vec(1, 2), vec(1, 2, 3)},
			Outputs:      []types.Vector{vec(1), vec(0)},
			LearningRate: 0.1,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sample 1")
	})
}

func TestApplyBatchGradients(t *testing.T) {
	t.Run("single sample updates weights correctly", func(t *testing.T) {
		// W = [1, -1; 0, 2], b = [0, 1], η = 1.0, n = 1
		// With η/n = 1, update is: W_new = W - gradient
		ff := NewFeedForward(2, nil, 2, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{1, -1, 0, 2})}
		ff.biases = []types.Vector{mat.NewVecDense(2, []float64{0, 1})}

		wGrad := mat.NewDense(2, 2, []float64{0.1, 0.2, 0.3, 0.4})
		bGrad := mat.NewVecDense(2, []float64{0.5, 0.6})

		result := &types.TrainBatchResult{
			Results: []types.TrainSingleResult{
				{
					WeightGradients: []types.Matrix{wGrad},
					BiasGradients:   []types.Vector{bGrad},
				},
			},
		}

		ff.applyBatchGradients(1.0, result)

		// W_new = W - (1/1) * grad = [1-0.1, -1-0.2; 0-0.3, 2-0.4]
		assert.InDelta(t, 0.9, ff.weights[0].At(0, 0), 1e-10)
		assert.InDelta(t, -1.2, ff.weights[0].At(0, 1), 1e-10)
		assert.InDelta(t, -0.3, ff.weights[0].At(1, 0), 1e-10)
		assert.InDelta(t, 1.6, ff.weights[0].At(1, 1), 1e-10)

		// b_new = b - (1/1) * grad = [0-0.5, 1-0.6]
		assert.InDelta(t, -0.5, ff.biases[0].AtVec(0), 1e-10)
		assert.InDelta(t, 0.4, ff.biases[0].AtVec(1), 1e-10)
	})

	t.Run("averages gradients over batch", func(t *testing.T) {
		// Two samples with η = 1.0: update = W - (1/2) * (grad1 + grad2)
		ff := NewFeedForward(1, nil, 1, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(1, 1, []float64{5.0})}
		ff.biases = []types.Vector{mat.NewVecDense(1, []float64{3.0})}

		result := &types.TrainBatchResult{
			Results: []types.TrainSingleResult{
				{
					WeightGradients: []types.Matrix{mat.NewDense(1, 1, []float64{2.0})},
					BiasGradients:   []types.Vector{mat.NewVecDense(1, []float64{4.0})},
				},
				{
					WeightGradients: []types.Matrix{mat.NewDense(1, 1, []float64{6.0})},
					BiasGradients:   []types.Vector{mat.NewVecDense(1, []float64{8.0})},
				},
			},
		}

		ff.applyBatchGradients(1.0, result)

		// W = 5 - (1/2)*(2+6) = 5 - 4 = 1
		assert.InDelta(t, 1.0, ff.weights[0].At(0, 0), 1e-10)
		// b = 3 - (1/2)*(4+8) = 3 - 6 = -3
		assert.InDelta(t, -3.0, ff.biases[0].AtVec(0), 1e-10)
	})

	t.Run("scales by learning rate", func(t *testing.T) {
		// Single sample with η = 0.5, n = 1: update = W - 0.5 * gradient
		ff := NewFeedForward(1, nil, 1, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(1, 1, []float64{10.0})}
		ff.biases = []types.Vector{mat.NewVecDense(1, []float64{4.0})}

		result := &types.TrainBatchResult{
			Results: []types.TrainSingleResult{
				{
					WeightGradients: []types.Matrix{mat.NewDense(1, 1, []float64{2.0})},
					BiasGradients:   []types.Vector{mat.NewVecDense(1, []float64{6.0})},
				},
			},
		}

		ff.applyBatchGradients(0.5, result)

		// W = 10 - 0.5 * 2 = 9
		assert.InDelta(t, 9.0, ff.weights[0].At(0, 0), 1e-10)
		// b = 4 - 0.5 * 6 = 1
		assert.InDelta(t, 1.0, ff.biases[0].AtVec(0), 1e-10)
	})

	t.Run("empty batch is a no-op", func(t *testing.T) {
		ff := NewFeedForward(2, nil, 2, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{1, 2, 3, 4})}
		ff.biases = []types.Vector{mat.NewVecDense(2, []float64{5, 6})}

		result := &types.TrainBatchResult{Results: nil}
		ff.applyBatchGradients(1.0, result)

		assert.Equal(t, 1.0, ff.weights[0].At(0, 0))
		assert.Equal(t, 2.0, ff.weights[0].At(0, 1))
		assert.Equal(t, 5.0, ff.biases[0].AtVec(0))
		assert.Equal(t, 6.0, ff.biases[0].AtVec(1))
	})

	t.Run("populates UpdatedWeightGradients and UpdatedBiasGradients", func(t *testing.T) {
		ff := NewFeedForward(2, nil, 1, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(1, 2, []float64{1, 2})}
		ff.biases = []types.Vector{mat.NewVecDense(1, []float64{3})}

		result := &types.TrainBatchResult{
			Results: []types.TrainSingleResult{
				{
					WeightGradients: []types.Matrix{mat.NewDense(1, 2, []float64{0.1, 0.2})},
					BiasGradients:   []types.Vector{mat.NewVecDense(1, []float64{0.3})},
				},
			},
		}

		ff.applyBatchGradients(1.0, result)

		require.Len(t, result.UpdatedWeightGradients, 1)
		require.Len(t, result.UpdatedBiasGradients, 1)

		// Updated values should match the new weights/biases
		assert.Equal(t, ff.weights[0].At(0, 0), result.UpdatedWeightGradients[0].At(0, 0))
		assert.Equal(t, ff.biases[0].AtVec(0), result.UpdatedBiasGradients[0].AtVec(0))
	})
}

func TestPredict(t *testing.T) {
	t.Run("wrong input dimension", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Predict(vec(1, 2, 3))

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})

	t.Run("no hidden layers", func(t *testing.T) {
		// Same setup as TestNoHiddenLayers: 2→2, known W and b
		// x = [1, 2], W = [1 -1; 0 2], b = [0, 1]
		// z = [-1, 5], a = σ(z) ≈ [0.26894, 0.99331]
		ff := NewFeedForward(2, nil, 2, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{1, -1, 0, 2})}
		ff.biases = []types.Vector{mat.NewVecDense(2, []float64{0, 1})}

		result, err := ff.Predict(vec(1, 2))

		require.NoError(t, err)
		require.Equal(t, 2, result.Len())
		assert.InDelta(t, 0.26894, result.AtVec(0), 1e-3)
		assert.InDelta(t, 0.99331, result.AtVec(1), 1e-3)
	})

	t.Run("two layers", func(t *testing.T) {
		// Same setup as TestTwoLayersTrain: 2→2→1
		// a² ≈ [0.77400]
		ff := NewFeedForward(2, []int{2}, 1, WithSeed(42))
		ff.weights = []types.Matrix{
			mat.NewDense(2, 2, []float64{1, -1, 0, 1}),
			mat.NewDense(1, 2, []float64{1, 1}),
		}
		ff.biases = []types.Vector{
			mat.NewVecDense(2, []float64{0, 0}),
			mat.NewVecDense(1, []float64{0}),
		}

		result, err := ff.Predict(vec(1, 0))

		require.NoError(t, err)
		require.Equal(t, 1, result.Len())
		assert.InDelta(t, 0.77400, result.AtVec(0), 1e-3)
	})

	t.Run("matches trainSingleSample output", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		predicted, err := ff.Predict(vec(0.5, 0.8))
		require.NoError(t, err)

		trainResult, err := ff.trainSingleSample(vec(0.5, 0.8), vec(1))
		require.NoError(t, err)

		// Predict output should equal the last activation vector from training
		lastA := trainResult.AVectors[len(trainResult.AVectors)-1]
		for i := range predicted.Len() {
			assert.Equal(t, lastA.AtVec(i), predicted.AtVec(i))
		}
	})
}
