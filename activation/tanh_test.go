package activation

import (
	"testing"
	"math"
)

func getTanhExpectedValues() (map[float64]float64) {
	return map[float64]float64{
		0.01: 0.010000, 
		0.02: 0.019997, 
		0.34: 0.327477, 
		0.72: 0.616909,
	}
}

func TestActivateTanh(t *testing.T) {
	activation := Tanh{}
	vals := getTanhExpectedValues()
	for val, expected := range vals {
		got := activation.Activate(val)
		if math.Abs(got - expected) > 0.0001 {
			t.Errorf("Expected %f, get %f", expected, got)
		}
	}
}
