package nn

import (
	"math"
	"math/rand/v2"

	"nnlearn/internal/activation"
	"nnlearn/internal/cost"
	"nnlearn/internal/types"

	"gonum.org/v1/gonum/mat"
)

// InitializerFunc initializes weight matrix and bias vector for a layer
// with the given number of neurons and inputs.
type InitializerFunc func(rng *rand.Rand, neurons, inputs int) (types.Matrix, types.Vector)

// Option configures the neural network constructor.
type Option func(*options)

type options struct {
	initializer InitializerFunc
	seed        *uint64
	activation  types.Activation
	cost        types.Cost
}

func defaultOptions() options {
	return options{
		initializer: XavierInitializer,
		activation:  activation.NewSigmoid(),
		cost:        cost.New(),
	}
}

// XavierInitializer initializes weights with Xavier/Glorot uniform initialization
// and biases to zero.
func XavierInitializer(rng *rand.Rand, neurons, inputs int) (types.Matrix, types.Vector) {
	limit := math.Sqrt(6.0 / float64(inputs+neurons))
	weights := make([]float64, neurons*inputs)
	for i := range weights {
		weights[i] = (rng.Float64()*2 - 1) * limit
	}
	return mat.NewDense(neurons, inputs, weights), mat.NewVecDense(neurons, nil)
}

// WithInitializer sets the function used to initialize weights and biases.
func WithInitializer(fn InitializerFunc) Option {
	return func(o *options) {
		o.initializer = fn
	}
}

// WithSeed sets the random seed for weight initialization. If not set, a random seed is used.
func WithSeed(seed uint64) Option {
	return func(o *options) {
		o.seed = &seed
	}
}

// WithActivation sets the activateFunc function. Defaults to sigmoid.
func WithActivation(a types.Activation) Option {
	return func(o *options) {
		o.activation = a
	}
}

// WithCost sets the costFunc function. Defaults to quadratic (MSE).
func WithCost(c types.Cost) Option {
	return func(o *options) {
		o.cost = c
	}
}
