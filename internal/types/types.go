// Package types defines shared types and interfaces for nnlearn.
package types

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

// ErrDimensionMismatch is returned when vectors or matrices have incompatible dimensions.
var ErrDimensionMismatch = errors.New("dimension mismatch")

// Matrix is the matrix type. For now, it's just an alias to the gonum Matrix interface.
type Matrix = mat.Matrix

// Vector is the vector type. For now, it's just an alias to the gonum Vector interface.
type Vector = mat.Vector

// Cost is the interface to compute the cost
type Cost interface {
	// SingleCost computes the cost of a single x training sample and keeps track of it.
	//
	// y - predicted result for the training sample
	// a - activation of the output layer for the training sample
	SingleCost(y, a Vector) (float64, error)

	// Cost returns the actual cost based on the added single costs
	Cost() (float64, error)

	// Clear resets the cost to zero for the next epoch
	Clear()

	// PartialCostA returns the partial derivative of the cost function with respect to the activation of the output layer.
	PartialCostA(y, a Vector) (Vector, error)

	// Name returns the name of the cost function
	Name() string
}

// Activation is the interface of the activation function on a single variable z.
type Activation interface {
	// Activate is the activation function
	Activate(z float64) float64

	// ActivatePrime is the first derivative of the activation function
	ActivatePrime(z float64) float64

	// Name returns the name of the activation function
	Name() string
}

// TrainingBatch is the struct that holds the inputs and outputs for a training batch.
type TrainingBatch struct {
	// Inputs is the input data for the training batch. Each vector in the slice represents a single training sample.
	Inputs []Vector

	// Outputs is the output data for the training batch. Each vector in the slice represents a single training sample's expected output.
	Outputs []Vector

	// Learning Rate
	LearningRate float64
}

// TrainSingleResult is the interim result of a single training sample
type TrainSingleResult struct {
	// AVectors is the activation vectors at layer l
	AVectors []Vector

	// ZVectors is the weighted inputs at neurons at layer l
	ZVectors []Vector

	// DeltaVectors is the delta errors at neurons at layer l
	DeltaVectors []Vector

	// WeightGradients is the weight gradients at layer l
	WeightGradients []Matrix

	// BiasGradients is the bais gradients at layer l
	BiasGradients []Vector
}

// TrainBatchResult holds the results of training a batch of samples.
type TrainBatchResult struct {
	// Results of each individual training sample in the batch
	Results []TrainSingleResult

	// UpdatedWeightGradients is the new weight gradients updated with the training batch result
	UpdatedWeightGradients []Matrix

	// UpdatedBiasGradients is the new  bias gradients updated with the training batch result
	UpdatedBiasGradients []Vector
}

// Gradient represent a descent gradient matrix
type Gradient Matrix

// NeuralNetwork is the interface for a neural network.
type NeuralNetwork interface {
	// InputLength returns the length of the input layer of the neural network.
	InputLength() int

	// HiddenLengths returns the lengths of the hidden layers of the neural network.
	HiddenLengths() []int

	// OutputLength returns the length of the output layer of the neural network.
	OutputLength() int

	// Weights returns the weight matrix for the given layer.
	Weights(layer int) Matrix

	// Biases returns the bias vector for the given layer.
	Biases(layer int) Vector

	// Train trains a batch of training samples. Internally, the NN updates its weights based on the
	// computed gradient. This intermediate gradient is also returned.
	Train(batch TrainingBatch) (result *TrainBatchResult, err error)

	// Predict takes an input and returns the predicted output using the existing weights and biases
	Predict(input Vector) (Vector, error)
}
