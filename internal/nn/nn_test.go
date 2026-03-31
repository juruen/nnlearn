package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
