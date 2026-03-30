package activation

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestActivate(t *testing.T) {
	s := NewSigmoid()

	t.Run("zero", func(t *testing.T) {
		assert.InDelta(t, 0.5, s.Activate(0), 1e-10)
	})

	t.Run("large positive saturates to 1", func(t *testing.T) {
		assert.InDelta(t, 1.0, s.Activate(100), 1e-10)
	})

	t.Run("large negative saturates to 0", func(t *testing.T) {
		assert.InDelta(t, 0.0, s.Activate(-100), 1e-10)
	})

	t.Run("symmetry", func(t *testing.T) {
		// σ(z) + σ(-z) = 1
		z := 2.5
		assert.InDelta(t, 1.0, s.Activate(z)+s.Activate(-z), 1e-10)
	})

	t.Run("known value", func(t *testing.T) {
		// σ(1) ≈ 0.7310585786
		assert.InDelta(t, 0.7310585786, s.Activate(1), 1e-8)
	})
}

func TestActivatePrime(t *testing.T) {
	s := NewSigmoid()

	t.Run("max at zero", func(t *testing.T) {
		// σ'(0) = 0.5 * 0.5 = 0.25
		assert.InDelta(t, 0.25, s.ActivatePrime(0), 1e-10)
	})

	t.Run("approaches zero for large positive", func(t *testing.T) {
		assert.InDelta(t, 0.0, s.ActivatePrime(100), 1e-10)
	})

	t.Run("approaches zero for large negative", func(t *testing.T) {
		assert.InDelta(t, 0.0, s.ActivatePrime(-100), 1e-10)
	})

	t.Run("symmetry", func(t *testing.T) {
		// σ'(z) = σ'(-z)
		z := 3.0
		assert.InDelta(t, s.ActivatePrime(z), s.ActivatePrime(-z), 1e-10)
	})
}

func TestSigmoidName(t *testing.T) {
	s := NewSigmoid()
	assert.Equal(t, "sigmoid", s.Name())
}
