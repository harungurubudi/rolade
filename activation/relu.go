package activation

import (
	"math"
)

type Relu struct{}

var zero float64

func (s *Relu) Activate(val float64) (result float64) {
	return math.Max(zero, val)
}

func (s *Relu) Derivate(val float64) (result float64) {
	if val >= zero {
		return 1
	}

	return 0
}