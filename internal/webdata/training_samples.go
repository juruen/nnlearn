package webdata

import (
	"fmt"

	"nnlearn/internal/mnist"
	"nnlearn/internal/types"
)

type TrainingSample struct {
	Index  int       `json:"index"`
	Label  int       `json:"label"`
	Pixels []float64 `json:"pixels"`
}

func BuildTrainingSamples(dataset *mnist.Dataset, limit int) ([]TrainingSample, error) {
	if dataset == nil {
		return nil, fmt.Errorf("dataset is nil")
	}

	count := min(limit, len(dataset.Images))
	if len(dataset.Labels) < count {
		return nil, fmt.Errorf("image count %d exceeds label count %d", count, len(dataset.Labels))
	}

	samples := make([]TrainingSample, 0, count)
	for i := range count {
		samples = append(samples, TrainingSample{
			Index:  i,
			Label:  argmaxLabel(dataset.Labels[i]),
			Pixels: vectorToSlice(dataset.Images[i]),
		})
	}

	return samples, nil
}

func argmaxLabel(v types.Vector) int {
	maxIdx := 0
	maxVal := v.AtVec(0)
	for i := 1; i < v.Len(); i++ {
		val := v.AtVec(i)
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

func vectorToSlice(v types.Vector) []float64 {
	data := make([]float64, v.Len())
	for i := range v.Len() {
		data[i] = v.AtVec(i)
	}
	return data
}
