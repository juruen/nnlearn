// Package cost implements cost functions for neural network training.
package cost

import (
	"fmt"
	maths "nnlearn/internal/math"

	"nnlearn/internal/types"

	"gonum.org/v1/gonum/mat"
)

// Quadratic is the quadratic cost function, also known as mean squared error (MSE).
type Quadratic struct {
	costs []float64
}

var _ types.Cost = (*Quadratic)(nil)

// New creates a Quadratic cost function.
func New() *Quadratic {
	return &Quadratic{}
}

// SingleCost computes the cost of a single x training sample and keeps track of it.
//
// y - predicted result for the training sample
// a - activation of the output layer for the training sample
//
// Uses quadratic error for an individual training sample: 0.5 * ||y - a||^2
func (q *Quadratic) SingleCost(y, a types.Vector) (float64, error) {
	if y.Len() != a.Len() {
		return 0, fmt.Errorf("%w: y has length %d, a has length %d", types.ErrDimensionMismatch, y.Len(), a.Len())
	}
	var diff mat.VecDense
	diff.SubVec(y, a)
	c := mat.Dot(&diff, &diff) / 2
	q.costs = append(q.costs, c)
	return c, nil
}

// Cost computes the averaged cost function of all tracked individual errors (SingleCost calls)
func (q *Quadratic) Cost() (float64, error) {
	if len(q.costs) == 0 {
		return 0, fmt.Errorf("no costs have been tracked")
	}
	var total float64
	for _, c := range q.costs {
		total += c
	}
	return total / float64(len(q.costs)), nil
}

// Clear clears the tracked training error samples
func (q *Quadratic) Clear() {
	q.costs = nil
}

// PartialCostA returns the derivative of the cost function with respect to the activation of the output layer,
// which is simply (a - y) for the quadratic cost function.
func (q *Quadratic) PartialCostA(y, a types.Vector) (types.Vector, error) {
	if y.Len() != a.Len() {
		return nil, fmt.Errorf("%w: y has length %d, a has length %d", types.ErrDimensionMismatch, y.Len(), a.Len())
	}

	return maths.SubVectors(a, y), nil
}

// Name returns the name of the cost function
func (q *Quadratic) Name() string {
	return "quadratic_error"
}
