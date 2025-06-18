package activation

import (
	"math"
)

// Sigmoid is an activation function that maps any real-valued input
// into the range (0, 1). It is defined as 1 / (1 + exp(-x)) and is
// commonly used in binary classification tasks. Its derivative is
// sigmoid(x) * (1 - sigmoid(x)). While simple and smooth, sigmoid
// can suffer from vanishing gradients for very large or small inputs.
type Sigmoid struct{}

func (s *Sigmoid) Activate(val float64) (result float64) {
	return 1 / (1 + math.Exp(float64(-1)*val))
}

func (s *Sigmoid) Derivate(val float64) (result float64) {
	return val * (1 - val)
}

func (l *Sigmoid) CallMe() string {
	return "sigmoid"
}
