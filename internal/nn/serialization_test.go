package nn

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"

	"nnlearn/internal/types"
)

func TestSaveAndLoad(t *testing.T) {
	t.Run("round trip preserves network", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		dir := t.TempDir()
		path := filepath.Join(dir, "nn.json")

		err := ff.Save(path)
		require.NoError(t, err)

		loaded, err := LoadFeedForward(path)
		require.NoError(t, err)

		assert.Equal(t, ff.inputLen, loaded.inputLen)
		assert.Equal(t, ff.outputLen, loaded.outputLen)
		assert.Equal(t, ff.hiddenLens, loaded.hiddenLens)

		for l := range ff.weights {
			r, c := ff.weights[l].Dims()
			lr, lc := loaded.weights[l].Dims()
			require.Equal(t, r, lr)
			require.Equal(t, c, lc)
			for i := range r {
				for j := range c {
					assert.Equal(t, ff.weights[l].At(i, j), loaded.weights[l].At(i, j))
				}
			}
			for i := range ff.biases[l].Len() {
				assert.Equal(t, ff.biases[l].AtVec(i), loaded.biases[l].AtVec(i))
			}
		}
	})

	t.Run("predict matches after load", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		dir := t.TempDir()
		path := filepath.Join(dir, "nn.json")

		require.NoError(t, ff.Save(path))

		loaded, err := LoadFeedForward(path)
		require.NoError(t, err)

		input := vec(0.5, 0.8)
		expected, err := ff.Predict(input)
		require.NoError(t, err)

		actual, err := loaded.Predict(input)
		require.NoError(t, err)

		for i := range expected.Len() {
			assert.Equal(t, expected.AtVec(i), actual.AtVec(i))
		}
	})

	t.Run("predict matches after byte load", func(t *testing.T) {
		ff := NewFeedForward(2, []int{3}, 1, WithSeed(42))

		dir := t.TempDir()
		path := filepath.Join(dir, "nn.json")
		require.NoError(t, ff.Save(path))

		data, err := os.ReadFile(path)
		require.NoError(t, err)

		loaded, err := LoadFeedForwardBytes(data)
		require.NoError(t, err)

		input := vec(0.5, 0.8)
		expected, err := ff.Predict(input)
		require.NoError(t, err)

		actual, err := loaded.Predict(input)
		require.NoError(t, err)

		for i := range expected.Len() {
			assert.Equal(t, expected.AtVec(i), actual.AtVec(i))
		}
	})

	t.Run("custom weights round trip", func(t *testing.T) {
		ff := NewFeedForward(2, nil, 2, WithSeed(42))
		ff.weights = []types.Matrix{mat.NewDense(2, 2, []float64{1, -1, 0, 2})}
		ff.biases = []types.Vector{mat.NewVecDense(2, []float64{0.5, -0.5})}

		dir := t.TempDir()
		path := filepath.Join(dir, "nn.json")

		require.NoError(t, ff.Save(path))

		loaded, err := LoadFeedForward(path)
		require.NoError(t, err)

		assert.Equal(t, 1.0, loaded.weights[0].At(0, 0))
		assert.Equal(t, -1.0, loaded.weights[0].At(0, 1))
		assert.Equal(t, 0.0, loaded.weights[0].At(1, 0))
		assert.Equal(t, 2.0, loaded.weights[0].At(1, 1))
		assert.Equal(t, 0.5, loaded.biases[0].AtVec(0))
		assert.Equal(t, -0.5, loaded.biases[0].AtVec(1))
	})

	t.Run("load nonexistent file", func(t *testing.T) {
		_, err := LoadFeedForward("/nonexistent/path.json")
		require.Error(t, err)
	})

	t.Run("load invalid json", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "bad.json")
		require.NoError(t, os.WriteFile(path, []byte("not json"), 0644))

		_, err := LoadFeedForward(path)
		require.Error(t, err)
	})

	t.Run("load invalid bytes", func(t *testing.T) {
		_, err := LoadFeedForwardBytes([]byte("not json"))
		require.Error(t, err)
	})
}
