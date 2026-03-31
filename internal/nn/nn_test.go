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
			Inputs:  []types.Vector{vec(1, 2)},
			Outputs: []types.Vector{vec(1), vec(0)},
		})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "different lengths")
	})

	t.Run("wrong input dimension", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Train(types.TrainingBatch{
			Inputs:  []types.Vector{vec(1, 2, 3)},
			Outputs: []types.Vector{vec(1)},
		})

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})

	t.Run("wrong output dimension", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		_, err := ff.Train(types.TrainingBatch{
			Inputs:  []types.Vector{vec(1, 2)},
			Outputs: []types.Vector{vec(1, 2)},
		})

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})

	t.Run("single sample returns correct structure", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		result, err := ff.Train(types.TrainingBatch{
			Inputs:  []types.Vector{vec(0.5, 0.8)},
			Outputs: []types.Vector{vec(1)},
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
			Inputs:  []types.Vector{vec(0.5, 0.8)},
			Outputs: []types.Vector{vec(1)},
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
			Inputs:  []types.Vector{vec(0.5, 0.8), vec(0.1, 0.9), vec(0.3, 0.4)},
			Outputs: []types.Vector{vec(1), vec(0), vec(1)},
		})

		require.NoError(t, err)
		assert.Len(t, result.Results, 3)
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		batch := types.TrainingBatch{
			Inputs:  []types.Vector{vec(0.5, 0.8)},
			Outputs: []types.Vector{vec(1)},
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
			Inputs:  []types.Vector{vec(0.5, 0.8)},
			Outputs: []types.Vector{vec(1)},
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
			Inputs:  []types.Vector{},
			Outputs: []types.Vector{},
		})

		require.NoError(t, err)
		assert.Empty(t, result.Results)
	})
}

// TestSimpleTrain tests a minimal 2→2 network (no hidden layers) end-to-end.
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
//	                   └  0.00660  0.0132  ┘
//
//	∂C/∂b = δ ≈ [−0.1437, 0.00660]ᵀ
//
// This test covers: matrix multiply, sigmoid (including saturation at z=5),
// element-wise ops (Hadamard), and outer product.
func TestSimpleTrain(t *testing.T) {
	ff := NewFeedForward(2, nil, 2, WithSeed(42))

	ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{
		1, -1,
		0, 2,
	})}

	ff.biases = []types.Vector{mat.NewVecDense(2, []float64{0, 1})}

	trainBatch := types.TrainingBatch{
		Inputs:  []types.Vector{vec(1, 2)},
		Outputs: []types.Vector{vec(1, 0)},
	}

	r, err := ff.Train(trainBatch)
	require.NoError(t, err)
	require.Len(t, r.Results, 1)

	res := r.Results[0]
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
