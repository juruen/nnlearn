package webdata

import (
	"testing"

	"nnlearn/internal/mnist"
	"nnlearn/internal/types"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestBuildTrainingSamples(t *testing.T) {
	dataset := &mnist.Dataset{
		Images: []types.Vector{
			mat.NewVecDense(4, []float64{0, 0.5, 1, 0.25}),
			mat.NewVecDense(4, []float64{1, 0, 0, 1}),
		},
		Labels: []types.Vector{
			mat.NewVecDense(10, []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}),
			mat.NewVecDense(10, []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}),
		},
	}

	samples, err := BuildTrainingSamples(dataset, 100)
	require.NoError(t, err)
	require.Len(t, samples, 2)

	assert.Equal(t, 0, samples[0].Index)
	assert.Equal(t, 3, samples[0].Label)
	assert.Equal(t, []float64{0, 0.5, 1, 0.25}, samples[0].Pixels)

	assert.Equal(t, 1, samples[1].Index)
	assert.Equal(t, 7, samples[1].Label)
	assert.Equal(t, []float64{1, 0, 0, 1}, samples[1].Pixels)
}

func TestBuildTrainingSamplesRequiresLabels(t *testing.T) {
	dataset := &mnist.Dataset{
		Images: []types.Vector{mat.NewVecDense(2, []float64{0, 1})},
	}

	_, err := BuildTrainingSamples(dataset, 1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds label count")
}
