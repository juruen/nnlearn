// Package activation implements activation functions for neural network layers.
package activation

import (
	"math"
	"nnlearn/internal/types"
)

// Sigmoid implements the sigmoid activation function: σ(z) = 1 / (1 + e^(-z)).
type Sigmoid struct{}

var _ types.Activation = (*Sigmoid)(nil)

// NewSigmoid creates a new Sigmoid activation function.
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

// Activate computes σ(z) = 1 / (1 + e^(-z)).
func (s *Sigmoid) Activate(z float64) float64 {
	return float64(1) / (1 + math.Exp(-z))
}

// ActivatePrime computes σ'(z) = σ(z) * (1 - σ(z)).
func (s *Sigmoid) ActivatePrime(z float64) float64 {
	sig := s.Activate(z)
	return sig * (1 - sig)
}

// Name returns the name of the activation function.
func (s *Sigmoid) Name() string {
	return "sigmoid"
}
