package activation

import (
	"testing"
	"math"
)

func getSigmoidExpectedValues() (map[float64]float64) {
	return map[float64]float64{
		0.01: 0.502500, 
		0.02: 0.505000, 
		0.34: 0.584191, 
		0.72: 0.672607,
	}
}

func TestActivateSigmoid(t *testing.T) {
	activation := Sigmoid{}
	vals := getSigmoidExpectedValues()
	for val, expected := range vals {
		got := activation.Activate(val)
		if math.Abs(got - expected) > 0.0001 {
			t.Errorf("Expected %f, get %f", expected, got)
		}
	}
}
