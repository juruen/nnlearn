package cost

import (
	"testing"

	"nnlearn/internal/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func vec(data ...float64) *mat.VecDense {
	return mat.NewVecDense(len(data), data)
}

func TestSingleCost(t *testing.T) {
	t.Run("basic cost", func(t *testing.T) {
		q := New()

		// y = [1, 0, 0], a = [0, 0, 0] → diff = [1,0,0], cost = 0.5
		c, err := q.SingleCost(vec(1, 0, 0), vec(0, 0, 0))
		require.NoError(t, err)
		assert.InDelta(t, 0.5, c, 1e-10)
	})

	t.Run("zero diff", func(t *testing.T) {
		q := New()

		c, err := q.SingleCost(vec(1, 2, 3), vec(1, 2, 3))
		require.NoError(t, err)
		assert.Equal(t, 0.0, c)
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		q := New()

		_, err := q.SingleCost(vec(1, 2), vec(1, 2, 3))
		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})
}

func TestCost(t *testing.T) {
	t.Run("average", func(t *testing.T) {
		q := New()

		// sample 1: diff = [1,0], cost = 0.5
		// sample 2: diff = [0,2], cost = 2.0
		// average = (0.5 + 2.0) / 2 = 1.25
		_, _ = q.SingleCost(vec(1, 0), vec(0, 0))
		_, _ = q.SingleCost(vec(0, 2), vec(0, 0))

		avg, err := q.Cost()
		require.NoError(t, err)
		assert.InDelta(t, 1.25, avg, 1e-10)
	})

	t.Run("no costs tracked", func(t *testing.T) {
		q := New()

		_, err := q.Cost()
		require.Error(t, err)
	})
}

func TestClear(t *testing.T) {
	q := New()

	_, _ = q.SingleCost(vec(1, 0), vec(0, 0))
	q.Clear()

	_, err := q.Cost()
	require.Error(t, err)
}

func TestName(t *testing.T) {
	q := New()
	assert.Equal(t, "quadratic_error", q.Name())
}

func TestPartialCostA(t *testing.T) {
	q := New()

	t.Run("basic derivative", func(t *testing.T) {
		// ∂C/∂a = (a - y)
		y := vec(1, 0, 0)
		a := vec(0.8, 0.1, 0.2)

		result, err := q.PartialCostA(y, a)

		require.NoError(t, err)
		require.Equal(t, 3, result.Len())
		assert.InDelta(t, -0.2, result.AtVec(0), 1e-10)
		assert.InDelta(t, 0.1, result.AtVec(1), 1e-10)
		assert.InDelta(t, 0.2, result.AtVec(2), 1e-10)
	})

	t.Run("zero when equal", func(t *testing.T) {
		y := vec(1, 2, 3)
		a := vec(1, 2, 3)

		result, err := q.PartialCostA(y, a)

		require.NoError(t, err)
		for i := range result.Len() {
			assert.Equal(t, 0.0, result.AtVec(i))
		}
	})

	t.Run("dimension mismatch", func(t *testing.T) {
		_, err := q.PartialCostA(vec(1, 2), vec(1, 2, 3))

		require.Error(t, err)
		assert.ErrorIs(t, err, types.ErrDimensionMismatch)
	})
}
