package activation

import (
	"testing"
	"math"
)

func getReluExpectedValues() (map[float64]float64) {
	return map[float64]float64{
		-0.01: 0, 
		0.02: 0.02, 
		-0.34: 0, 
		0.25: 0.25,
	}
}

func TestActivateRelu(t *testing.T) {
	activation := Relu{}
	vals := getReluExpectedValues()
	for val, expected := range vals {
		got := activation.Activate(val)
		if math.Abs(got - expected) > 0.0001 {
			t.Errorf("Expected %f, get %f", expected, got)
		}
	}
}
