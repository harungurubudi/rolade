package activation

import (
	"math"
)

type Sigmoid struct{}

func (s *Sigmoid) Activate(val float64) (result float64) {
	return 1 / (1 + math.Exp(float64(-1) * val))
}

func (s *Sigmoid) Derivate(val float64) (result float64) {
	return val * (1 - val)
}