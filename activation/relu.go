package activation

import (
	"math"
)

// ReLU (Rectified Linear Unit) is a commonly used activation function
// in neural networks. It outputs the input directly if it is positive,
// otherwise it outputs zero. Its derivative is 1 for positive inputs
// and 0 for non-positive inputs. ReLU is popular due to its simplicity
// and ability to mitigate the vanishing gradient problem.
type ReLU struct{}

var zero float64

func (s *ReLU) Activate(val float64) (result float64) {
	return math.Max(zero, val)
}

func (s *ReLU) Derivate(val float64) (result float64) {
	if val >= zero {
		return 1
	}

	return 0
}

func (l *ReLU) CallMe() string {
	return "relu"
}
