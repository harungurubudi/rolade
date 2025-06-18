package activation

import (
	"math"
)

// Tanh (Hyperbolic Tangent) is an activation function that maps input
// values to the range (-1, 1). It is defined as (exp(x) - exp(-x)) /
// (exp(x) + exp(-x)). Compared to Sigmoid, Tanh is zero-centered,
// which can help with optimization. Its derivative is 1 - tanh(x)^2.
type Tanh struct{}

func (s *Tanh) Activate(val float64) (result float64) {
	return math.Tanh(val)
}

func (s *Tanh) Derivate(val float64) (result float64) {
	return 1 / math.Pow(math.Cosh(val), 2)
}

func (l *Tanh) CallMe() string {
	return "tanh"
}
