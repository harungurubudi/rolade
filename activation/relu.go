package activation

import (
	"math"
)

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
