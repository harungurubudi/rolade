package activation

import (
	"math"
)

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