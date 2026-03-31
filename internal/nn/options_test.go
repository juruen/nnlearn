package nn

import (
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestXavierInitializer(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))

	t.Run("correct dimensions", func(t *testing.T) {
		weights, biases := XavierInitializer(rng, 5, 3)

		rows, cols := weights.Dims()
		assert.Equal(t, 5, rows)
		assert.Equal(t, 3, cols)
		assert.Equal(t, 5, biases.Len())
	})

	t.Run("weights in range [0, 1)", func(t *testing.T) {
		weights, _ := XavierInitializer(rng, 10, 8)

		rows, cols := weights.Dims()
		for r := range rows {
			for c := range cols {
				v := weights.At(r, c)
				assert.GreaterOrEqual(t, v, 0.0)
				assert.Less(t, v, 1.0)
			}
		}
	})

	t.Run("biases are zero", func(t *testing.T) {
		_, biases := XavierInitializer(rng, 6, 4)

		for i := range biases.Len() {
			assert.Equal(t, 0.0, biases.AtVec(i))
		}
	})

	t.Run("deterministic with same seed", func(t *testing.T) {
		rng1 := rand.New(rand.NewPCG(99, 0))
		rng2 := rand.New(rand.NewPCG(99, 0))

		w1, b1 := XavierInitializer(rng1, 4, 3)
		w2, b2 := XavierInitializer(rng2, 4, 3)

		rows, cols := w1.Dims()
		for r := range rows {
			for c := range cols {
				require.Equal(t, w1.At(r, c), w2.At(r, c))
			}
		}
		for i := range b1.Len() {
			require.Equal(t, b1.AtVec(i), b2.AtVec(i))
		}
	})
}
